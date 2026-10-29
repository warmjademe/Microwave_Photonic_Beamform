"""原864环境的全部11组冻结控制器敏感性；非零扰动重新测量。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import SOURCE,require_host,sha256,verify_sources,write_json,now,rng_for,atomic_npz
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_batch import guard_load
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS
from study_uniform64.controllers import Feedback64,audit as audit_trace
from our_method_response_control.train import precision
from native_sim.control_engine import APD_GAIN,RESPONSIVITY_A_PER_W,DEMODULATOR_GAIN,NativeControlEngine
from revision_af_20261029.perturbed_receiver import SCENARIOS,configuration,public_input,score

METHODS=['cnn_warm64','covariance_warm64','initial_select64']
STATE={}


def read(path):return json.loads(Path(path).read_text())


def fp(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def initialize(output,p):
    require_host();precision();verify_sources(p['source_sha256'])
    bundle=Path(p['prior']['runtime_bundle']);verify_bundle(bundle)
    with np.load(bundle/'public.npz') as f:public={k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    with patch('numpy.load',guard_load(np.load)):
        learned=OnlineController(bundle,'complex_response_cnn',public)
        conventional=OnlineController(bundle,'covariance_response',public)
        scan=OnlineController(bundle,'ttd_das',public)
        models=[Feedback64(learned,public),Feedback64(conventional,public),Feedback64(scan,public,fixed_catalog=True)]
    STATE.update(output=Path(output),protocol=p,public=public,models=models)
    write_json(Path(output)/'workers'/f'{os.getpid()}.json',dict(pid=os.getpid(),at=now(),gpu=torch.cuda.get_device_name(),joint_correction=False,
        artifact_sha256={k:v for m in models for k,v in m.artifacts.items()}))


def anchor(row,carrier):
    source=Path(STATE['protocol']['anchor']);folder=source/'records'/f"environment_{row['index']:05d}"
    marker=read(folder/'complete.json')
    assert marker['environment_id']==row['environment_id']
    original_folder=Path(STATE['protocol']['environment_source'])/'records'/folder.name
    original_marker=read(original_folder/'complete.json')
    assert original_marker['environment_id']==row['environment_id']
    assert sha256(original_folder/'environment.json')==original_marker['environment_sha256']
    assert read(folder/'environment.json')==read(original_folder/'environment.json')
    entry=next(e for e in marker['carriers'] if e['carrier_ghz']==carrier)
    path=source/entry['path'];assert sha256(path)==entry['sha256']
    original_names=STATE['protocol']['anchor_methods'];indices=[original_names.index(n) for n in METHODS]
    with np.load(path,allow_pickle=False) as f:
        a={k:f[k].copy() for k in ['public_X']}
        a.update(control_code=f['control_code'][indices].copy(),metrics=f['metrics'][indices].copy())
        for name in METHODS:
            for suffix in ['trace_control_code','trace_scores','proposal_control_code']:
                key=name+'__'+suffix
                if key in f:a[key]=f[key].copy()
    audit_trace(a,METHODS,STATE['public'],carrier)
    return read(folder/'environment.json'),a,entry['sha256']


def compute(row,carrier,name):
    p,public=STATE['protocol'],STATE['public']
    original,old,anchor_hash=anchor(row,carrier)
    env,config=configuration(original,name)
    if name=='nominal' and not p['preflight']:
        return old,dict(origin='verified_nominal_anchor',anchor_sha256=anchor_hash,joint_correction=False,trace_audit=True),env,config
    raw,engine=public_input(env,carrier,public,p['prior']['backend'],config)
    arrays={'public_X':raw};controls=[];costs=[]
    with patch('numpy.load',guard_load(np.load)):
        for method,model in zip(METHODS,STATE['models']):
            def measure(u,call):return engine.measure_detailed(u,rng_for(row['seed'],carrier,620,call))['score']
            got=model.decide(raw.copy(),rng_for(0,row['seed'],carrier,630),measure)
            controls.append(got['control_code']);costs.append([got['feedback_calls'],got['software_seconds'],got['feedback_simulator_seconds']])
            for suffix in ['trace_control_code','trace_scores','proposal_control_code']:
                if suffix in got:arrays[method+'__'+suffix]=got[suffix]
    decisions=dict(methods=METHODS,control_code=np.asarray(controls),costs=np.asarray(costs))
    arrays.update(control_code=decisions['control_code'],metrics=score(env,carrier,public,p['prior']['backend'],config,decisions))
    audit_trace(arrays,METHODS,public,carrier)
    if name=='nominal':
        for key in arrays:
            if key=='metrics':np.testing.assert_allclose(arrays[key][:,:10],old[key][:,:10],rtol=1e-12,atol=1e-28)
            else:np.testing.assert_array_equal(arrays[key],old[key])
        np.testing.assert_array_equal(arrays['metrics'][:,[0,1,3,4,5,6,10]],old['metrics'][:,[0,1,3,4,5,6,10]])
    trans=engine.transmissions[decisions['control_code'][:,64:]]
    actual_psd,actual_var=engine.noise_for_transmission(trans)
    optical=trans@engine.branch_dc_w
    nominal_psd,nominal_var=NativeControlEngine._noise(engine,optical)
    extra=config['rin_linear_per_hz']*(APD_GAIN*RESPONSIVITY_A_PER_W)**2*np.sum((trans*engine.branch_dc_w)**2,axis=-1)
    np.testing.assert_allclose(actual_psd,nominal_psd+extra,rtol=1e-14,atol=0)
    np.testing.assert_allclose(actual_var,nominal_var+DEMODULATOR_GAIN**2*engine.cfg.df_hz*extra,rtol=1e-14,atol=0)
    for model in STATE['models']:model.base.verify()
    return arrays,dict(origin='new_perturbed_evaluation',anchor_sha256=anchor_hash,joint_correction=False,
        trace_audit=True,zero_perturbation_anchor_equal=(name=='nominal'),rin_psd_formula_checked=True),env,config


def one(row):
    output,p=STATE['output'],STATE['protocol'];all_entries=[]
    for name in SCENARIOS:
        folder=output/'records'/name/f"environment_{row['index']:05d}";folder.mkdir(parents=True,exist_ok=True);entries=[]
        for carrier in p['carriers']:
            path=folder/f'carrier_{carrier:02d}.npz';marker=path.with_suffix('.json')
            if marker.exists():
                e=read(marker);assert e['protocol_sha256']==fp(p) and sha256(path)==e['sha256']
            else:
                if path.exists():raise RuntimeError('Inspect uncommitted record '+str(path))
                arrays,checks,env,config=compute(row,carrier,name)
                descriptor=dict(environment=env,perturbation=config)
                if (folder/'condition.json').exists():assert read(folder/'condition.json')==descriptor
                else:write_json(folder/'condition.json',descriptor)
                atomic_npz(path,**arrays)
                e=dict(environment_id=row['environment_id'],carrier_ghz=carrier,scenario=name,path=str(path.relative_to(output)),
                    sha256=sha256(path),condition_sha256=sha256(folder/'condition.json'),protocol_sha256=fp(p),checks=checks)
                write_json(marker,e)
            entries.append(e)
            write_json(output/'workers'/f'{os.getpid()}_progress.json',dict(environment=row['index'],scenario=name,carrier=carrier,at=now()))
        all_entries.append(dict(scenario=name,carriers=entries))
    result=dict(index=row['index'],environment_id=row['environment_id'],scenarios=all_entries)
    write_json(output/'completed_environments'/f"environment_{row['index']:05d}.json",result)
    return result


def run(project,output,workers,preflight):
    require_host();base=project/'dataset_simulation/revision_rq2_af_20261029/full';p0=read(base/'protocol.json')
    verification=read(base/'verification.json');assert verification['status']=='passed'
    assert sha256(base/'protocol.json')==verification['files_sha256']['protocol.json']
    sources=dict(p0['source_sha256'])
    for n in ['evaluate_r.py','perturbed_receiver.py','R_PROTOCOL.md']:
        f=SOURCE/'revision_af_20261029'/n;sources[str(f.relative_to(SOURCE))]=sha256(f)
    verify_sources(sources)
    p=dict(schema='AF-frozen-receiver-sensitivity-v1',preflight=preflight,prior=p0['prior'],anchor=str(base),anchor_protocol_sha256=sha256(base/'protocol.json'),
        anchor_methods=p0['methods'],environment_source=p0['source'],rows=p0['rows'][:1] if preflight else p0['rows'],carriers=[4,12,20] if preflight else list(range(4,21)),
        scenarios=SCENARIOS,methods=METHODS,source_sha256=sources,code_sha256=sha256(Path(__file__)),joint_correction=False)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'complete.json').exists():raise RuntimeError('Already complete')
        if (output/'protocol.json').exists():assert read(output/'protocol.json')==p
        else:write_json(output/'protocol.json',p)
        for n,h in sources.items():
            dest=output/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==h
            else:shutil.copyfile(SOURCE/n,dest)
        (output/'workers').mkdir(exist_ok=True);(output/'completed_environments').mkdir(exist_ok=True)
        if preflight:
            initialize(output,p);rows=[one(p['rows'][0])]
        else:
            proof=read(output.parent/'preflight/complete.json')
            assert proof['preflight'] and proof['source_sha256']==sources
            rows=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(output,p)) as pool:
                for future in as_completed([pool.submit(one,r) for r in p['rows']]):
                    rows.append(future.result());write_json(output/'progress.json',dict(status='running',completed=len(rows),total=len(p['rows']),pid=os.getpid(),at=now()))
                    print(f"R: {len(rows)}/{len(p['rows'])}",flush=True)
        rows.sort(key=lambda r:r['index']);values=[]
        for row in rows:
            conditions=[]
            for scenario in row['scenarios']:
                a=[]
                for entry in scenario['carriers']:
                    file=output/entry['path'];assert sha256(file)==entry['sha256']
                    with np.load(file,allow_pickle=False) as f:a.append(f['metrics'].copy())
                conditions.append(a)
            values.append(conditions)
        values=np.asarray(values);assert values.shape==(len(rows),11,len(p['carriers']),3,13)
        summary=[]
        for k,scenario in enumerate(SCENARIOS):
            point=metrics(sufficient(values[:,k]).sum(0))
            for i,name in enumerate(METHODS):summary.append(dict(scenario=scenario,method=name,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)}))
        write_json(output/'summary.json',summary);write_json(output/'records.json',rows)
        atomic_npz(output/'environment_metrics.npz',values=values,methods=np.asarray(METHODS),scenarios=np.asarray(SCENARIOS),environment_ids=np.asarray([r['environment_id'] for r in rows]))
        verify_sources(sources)
        write_json(output/'complete.json',dict(status='complete',preflight=preflight,environments=len(rows),carriers=len(p['carriers']),scenarios=11,
            source_sha256=sources,code_sha256=p['code_sha256'],protocol_sha256=fp(p),joint_correction=False,at=now(),
            file_sha256={n:sha256(output/n) for n in ['protocol.json','summary.json','records.json','environment_metrics.npz']}))
        write_json(output/'progress.json',dict(status='complete',completed=len(rows),total=len(rows),at=now()))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--project',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--workers',type=int,default=4);a.add_argument('--preflight',action='store_true');args=a.parse_args()
    try:run(args.project.resolve(),args.output.resolve(),args.workers,args.preflight)
    except BaseException:
        args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
