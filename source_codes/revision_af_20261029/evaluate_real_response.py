"""冻结实数响应 CNN 在相同 A/F 接收协议下的补测。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import (SOURCE, LEVELS, require_host, sha256,
    verify_sources, write_json, now, rng_for, atomic_npz)
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_receiver import public_observation, score_decisions
from study_full_baselines.confirmation_batch import guard_load
from study_uniform64.controllers import Feedback64, audit as audit_trace
from our_method_response_control.train import precision
from study_full_baselines.paired_statistics import sufficient, metrics, FIELDS

METHODS = ['real_response_direct16', 'real_response_feedback64']
STATE = {}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def training_audit(project, bundle):
    """确认不同数据视图绑定同一训练群组和同一响应标签。"""
    m = read(bundle/'manifest.json')
    real = Path(m['training_provenance']['real']['path'])
    response = Path(m['training_provenance']['response']['path'])
    rp, ap = read(real/'protocol.json'), read(response/'protocol.json')
    rm, am = read(real/'complete.json'), read(response/'complete.json')
    assert rp['train_ids'] == m['cohort']['train_ids']
    assert len(rp['train_ids']) == ap['train_environments'] == 3456
    for k in ['epochs', 'seed', 'batch_size', 'learning_rate', 'gradient_clip']:
        assert rp[k] == ap[k]
    assert rp['epochs'] == 40 and rp['seed'] == 0 and not rp['validation'] and not ap['validation']
    rt = project/'dataset_simulation/baseline_results/20260925_full_baselines/fair_3456/targets'
    at = project/'dataset_simulation/baseline_results/20260925_full_baselines/scale_3456/targets'
    rmeta, ameta = read(rt/'complete.json'), read(at/'complete.json')
    assert sha256(rt/'complete.json') == rp['target_sha256']
    assert sha256(at/'complete.json') == ap['targets_complete_sha256']
    assert rmeta['source_target_complete_sha256'] == sha256(at/'complete.json')
    assert rmeta['label_sha256'] == ameta['label_sha256']
    for target in [rt, at]:
        assert sha256(target/'train_response.npy') == rmeta['label_sha256']
    for folder, meta in [(real, rm), (response, am)]:
        assert sha256(folder/'weights.pt') == meta['weights_sha256']
    return dict(train_environments=3456, train_samples=58752,
        shared_response_sha256=rmeta['label_sha256'], shared_training_members=True,
        real_parameters=rm['parameters'], complex_parameters=am['parameters'],
        real_weights_sha256=rm['weights_sha256'], complex_weights_sha256=am['weights_sha256'],
        differences=['real/complex convolution structure', 'GELU/magnitude activation', 'conditioning width'],
        note='whole-estimator comparison; no isolated operator attribution')


def initialize(output, protocol):
    require_host(); precision(); verify_sources(protocol['source_sha256'])
    bundle = Path(protocol['prior']['runtime_bundle']); verify_bundle(bundle)
    with np.load(bundle/'public.npz', allow_pickle=False) as f:
        public = {k:f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    base = OnlineController(bundle, 'real_response_cnn', public)
    assert not base.warm and base.kind == 'response'
    STATE.update(output=Path(output), protocol=protocol, public=public,
                 models=[base, Feedback64(base, public)])
    if protocol['preflight']:
        learned = OnlineController(bundle, 'complex_response_cnn', public)
        STATE['anchors'] = [learned, Feedback64(learned, public)]
    write_json(Path(output)/'workers'/f'{os.getpid()}.json', dict(pid=os.getpid(),
        at=now(), gpu=torch.cuda.get_device_name(), artifacts=base.artifacts,
        joint_correction=False))


def load_original(row, carrier):
    p = STATE['protocol']; folder = Path(p['original'])/'records'/f"environment_{row['index']:05d}"
    marker = read(folder/'complete.json')
    assert marker['environment_id'] == row['environment_id']
    assert sha256(folder/'environment.json') == marker['environment_sha256']
    item = next(e for e in marker['carriers'] if e['carrier_ghz'] == carrier)
    path = Path(p['original'])/item['path']; assert sha256(path) == item['sha256']
    with np.load(path, allow_pickle=False) as f:
        old = {k:f[k].copy() for k in f.files}
    return read(folder/'environment.json'), old


def compute(row, carrier):
    p, public = STATE['protocol'], STATE['public']; env, old = load_original(row, carrier)
    raw, observed = public_observation(env, carrier, public, p['prior']['backend'])
    np.testing.assert_array_equal(raw, old['public_X'])
    controls, costs, results = [], [], []; arrays = {'public_X':raw}
    names = METHODS + (['complex_response_cnn', 'cnn_warm64'] if p['preflight'] else [])
    models = STATE['models'] + (STATE['anchors'] if p['preflight'] else [])
    with patch('numpy.load', guard_load(np.load)):
        for i, (name, model) in enumerate(zip(names, models)):
            def measure(control, call):
                return observed.measure_detailed(control, rng_for(row['seed'], carrier, 620, call))['score']
            got = model.decide(raw.copy(), rng_for(0,row['seed'],carrier,630), measure if i % 2 else None)
            results.append(got); controls.append(got['control_code'])
            costs.append([got['feedback_calls'],got['software_seconds'],got['feedback_simulator_seconds']])
            for suffix in ['trace_control_code','trace_scores','proposal_control_code']:
                if suffix in got: arrays[name+'__'+suffix] = got[suffix]
        if p['preflight']:
            again = STATE['models'][0].decide(raw.copy(),rng_for(0,row['seed'],carrier,630),None)
            np.testing.assert_array_equal(again['control_code'],controls[0])
    decisions = dict(methods=names,control_code=np.asarray(controls),costs=np.asarray(costs),
                     feedback={},teacher_objective_evaluations=None)
    scored = score_decisions(env,carrier,public,decisions,p['prior']['backend'],include_mrc=False)
    for i in range(0,len(names),2):
        np.testing.assert_array_equal(controls[i], results[i+1]['proposal_control_code'])
    feedback_names = names[1::2]
    subset = {k:v for k,v in arrays.items() if k.startswith(tuple(n+'__' for n in feedback_names))}
    subset.update(public_X=raw,control_code=scored['control_code'][1::2],metrics=scored['metrics'][1::2])
    audit_trace(subset,feedback_names,public,carrier)
    if p['preflight']:
        for i,name in enumerate(names[2:],2):
            j=p['prior']['methods'].index(name)
            np.testing.assert_array_equal(scored['control_code'][i],old['control_code'][j])
            np.testing.assert_array_equal(scored['metrics'][i,[0,1,3,4,5,6,10]],old['metrics'][j,[0,1,3,4,5,6,10]])
            np.testing.assert_allclose(scored['metrics'][i,:10],old['metrics'][j,:10],rtol=1e-12,atol=1e-28)
            if i%2:
                for suffix in ['trace_control_code','trace_scores']:
                    np.testing.assert_array_equal(arrays[name+'__'+suffix],old[name+'__'+suffix])
        arrays={k:v for k,v in arrays.items() if k=='public_X' or k.startswith(METHODS[1]+'__')}
    arrays.update(control_code=scored['control_code'][:2],metrics=scored['metrics'][:2])
    np.testing.assert_array_equal(arrays['metrics'][:,10],[16,64])
    STATE['models'][0].verify()
    return arrays


def one(row):
    output,p=STATE['output'],STATE['protocol']; folder=output/'records'/f"environment_{row['index']:05d}"
    folder.mkdir(parents=True,exist_ok=True); entries=[]
    for carrier in p['carriers']:
        path=folder/f'carrier_{carrier:02d}.npz'; marker=path.with_suffix('.json')
        if marker.exists():
            e=read(marker); assert e['protocol_sha256']==fingerprint(p) and sha256(path)==e['sha256']
        else:
            if path.exists(): raise RuntimeError('Uncommitted record needs inspection')
            arrays=compute(row,carrier); atomic_npz(path,**arrays)
            e=dict(environment_id=row['environment_id'],carrier_ghz=carrier,path=str(path.relative_to(output)),
                sha256=sha256(path),protocol_sha256=fingerprint(p),
                checks=dict(public_input_equal=True,feedback_trace_audited=True,proposal_equal_direct=True,
                            A_AF_replay_checked=p['preflight'],joint_correction=False))
            write_json(marker,e)
        entries.append(e)
    record=dict(index=row['index'],environment_id=row['environment_id'],carriers=entries)
    write_json(folder/'complete.json',record); return record


def run(project,preflight,workers):
    require_host(); root=project/'dataset_simulation'; anchor=root/'revision_rq2_af_20261029/full'
    previous=read(anchor/'protocol.json'); bundle=Path(previous['prior']['runtime_bundle']); verify_bundle(bundle)
    audit=training_audit(project,bundle)
    sources=dict(previous['source_sha256'])
    for name in ['evaluate_real_response.py','C_RESPONSE_PROTOCOL.md']:
        sources['revision_af_20261029/'+name]=sha256(Path(__file__).with_name(name))
    verify_sources(sources)
    p=dict(schema='real-response-estimator-control-v1',preflight=preflight,
        rows=previous['rows'][:1] if preflight else previous['rows'],
        carriers=[4,12,20] if preflight else previous['carriers'],methods=METHODS,
        prior=previous['prior'],original=previous['source'],anchor=str(anchor),
        training_audit=audit,source_sha256=sources,code_sha256=sha256(Path(__file__)))
    output=root/'revision_af_20261029/C_response'/('preflight' if preflight else 'full')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'complete.json').exists(): raise RuntimeError('Already complete; do not rerun')
        if (output/'protocol.json').exists(): assert read(output/'protocol.json')==p
        else: write_json(output/'protocol.json',p)
        for name,h in sources.items():
            dest=output/'source_snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists(): assert sha256(dest)==h
            else: shutil.copyfile(SOURCE/name,dest)
        (output/'workers').mkdir(exist_ok=True)
        if preflight:
            initialize(output,p); records=[one(p['rows'][0])]
        else:
            check=read(output.parent/'preflight/complete.json')
            assert check['preflight'] and check['source_sha256']==sources and check['training_audit']==audit
            records=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(output,p)) as pool:
                for future in as_completed([pool.submit(one,r) for r in p['rows']]):
                    records.append(future.result())
                    write_json(output/'progress.json',dict(status='running',completed=len(records),total=len(p['rows']),pid=os.getpid(),at=now()))
        records.sort(key=lambda r:r['index']); values=[]
        for row in records:
            vals=[]
            for entry in row['carriers']:
                path=output/entry['path']; assert sha256(path)==entry['sha256']
                with np.load(path,allow_pickle=False) as f: vals.append(f['metrics'].copy())
            values.append(vals)
        values=np.asarray(values); assert values.shape==(len(p['rows']),len(p['carriers']),2,13)
        point=metrics(sufficient(values).sum(0))
        write_json(output/'summary.json',[dict(method=name,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)}) for i,name in enumerate(METHODS)])
        atomic_npz(output/'environment_metrics.npz',values=values,methods=np.asarray(METHODS),environment_ids=np.asarray([r['environment_id'] for r in records]))
        write_json(output/'records.json',records); verify_sources(sources); verify_bundle(bundle)
        assert training_audit(project,bundle)==audit
        write_json(output/'complete.json',dict(status='complete',at=now(),preflight=preflight,environments=len(records),carriers=len(p['carriers']),
            methods=METHODS,source_sha256=sources,code_sha256=p['code_sha256'],protocol_sha256=fingerprint(p),training_audit=audit,
            file_sha256={n:sha256(output/n) for n in ['protocol.json','records.json','summary.json','environment_metrics.npz']}))
        write_json(output/'progress.json',dict(status='complete',completed=len(records),total=len(records),at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--workers',type=int,default=4);a=parser.parse_args()
    try: run(a.project.resolve(),a.preflight,a.workers)
    except BaseException:
        folder=a.project/'dataset_simulation/revision_af_20261029/C_response'/('preflight' if a.preflight else 'full')
        folder.mkdir(parents=True,exist_ok=True);write_json(folder/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
