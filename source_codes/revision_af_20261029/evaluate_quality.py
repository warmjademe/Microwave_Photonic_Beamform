"""质量目标直接控制网络的完整验证/测试；网络只读取公共观测。"""
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
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import SOURCE,LEVELS,NativeConfig,require_host,sha256,verify_sources,write_json,now,rng_for,atomic_npz
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.confirmation_receiver import public_observation,score_decisions
from study_full_baselines.confirmation_batch import guard_load
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS
from study_uniform64.controllers import Feedback64,audit as audit_trace
from our_method_response_control.train import precision
from deep_common.layers import build
from deep_common.preprocessing import apply
from dataset_protocol.registry import SOURCES
from revision_af_20261029.quality_control import legal_codes

CONFIGS=['cnn_scratch','cnn_mse_init','transformer_scratch','transformer_mse_init']
METHODS=[n+suffix for n in CONFIGS for suffix in ['_direct16','_feedback64']]
STATE={}


def read(path):return json.loads(Path(path).read_text())


def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


class QualityController:
    warm=False
    kind='direct'
    def __init__(self,training,name):
        self.cfg=NativeConfig();self.name=name;self.artifacts={}
        for n in ['complete.json','protocol.json','normalization.npz',name+'/complete.json',name+'/weights.pt']:
            self.artifacts[str(training/n)]=sha256(training/n)
        complete=read(training/'complete.json');p=read(training/'protocol.json');meta=read(training/name/'complete.json')
        assert complete['status']=='complete' and name in complete['models']
        assert meta['status']=='complete' and meta['quality_epochs']==40 and meta['protocol_sha256']==fp(p)
        assert sha256(training/name/'weights.pt')==meta['weights_sha256']
        assert sha256(training/'normalization.npz')==complete['file_sha256']['normalization.npz']
        with np.load(training/'normalization.npz',allow_pickle=False) as f:self.stats={k:f[k].copy() for k in f.files}
        self.model=build(meta['method']).cuda()
        self.model.load_state_dict(torch.load(training/name/'weights.pt',map_location='cuda',weights_only=True));self.model.eval()
    def decide(self,raw,rng,measure_callback=None):
        if measure_callback is not None:raise ValueError('Direct network does not receive measurement callback')
        start=time.perf_counter();raw=np.asarray(raw)
        assert raw.shape==(2513,) and np.isfinite(raw).all()
        with torch.inference_mode():code=legal_codes(self.model(torch.from_numpy(apply(raw[None],self.stats)).cuda())).cpu().numpy()[0]
        return dict(control_code=code,control=code/LEVELS,feedback_calls=16,software_seconds=time.perf_counter()-start,feedback_simulator_seconds=0.)
    def verify(self):
        for path,h in self.artifacts.items():assert sha256(path)==h


def initialize(output,p):
    require_host();precision();verify_sources(p['source_sha256'])
    bundle=Path(p['prior']['runtime_bundle']);verify_bundle(bundle)
    with np.load(bundle/'public.npz',allow_pickle=False) as f:public={k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    models=[]
    # 这里只载入已冻结的归一化和权重；没有响应监督标签或测试答案入口。
    for name in CONFIGS:
        model=QualityController(Path(p['training']),name);models.extend([model,Feedback64(model,public)])
    STATE.update(output=Path(output),protocol=p,public=public,models=models)
    write_json(Path(output)/'workers'/f'{os.getpid()}.json',dict(pid=os.getpid(),at=now(),gpu=torch.cuda.get_device_name(),
        artifact_sha256={k:v for m in models for k,v in m.artifacts.items()},joint_correction=False))


def compute(row,carrier):
    p,public=STATE['protocol'],STATE['public'];path=Path(row['environment_path'])
    assert sha256(path)==row['environment_sha256'];env=read(path)
    raw,engine=public_observation(env,carrier,public,p['prior']['backend'])
    if p['role']=='test':
        folder=path.parent;record=read(folder/'complete.json')
        entry=next(e for e in record['carriers'] if e['carrier_ghz']==carrier)
        original=Path(p['input_record_source'])/entry['path'];assert sha256(original)==entry['sha256']
        with np.load(original,allow_pickle=False) as f:np.testing.assert_array_equal(raw,f['public_X'])
    arrays={'public_X':raw};control=[];cost=[]
    with patch('numpy.load',guard_load(np.load)):
        for i,(name,model) in enumerate(zip(METHODS,STATE['models'])):
            def measure(u,call):return engine.measure_detailed(u,rng_for(row['seed'],carrier,620,call))['score']
            got=model.decide(raw.copy(),rng_for(0,row['seed'],carrier,630),measure if i%2 else None)
            control.append(got['control_code']);cost.append([got['feedback_calls'],got['software_seconds'],got['feedback_simulator_seconds']])
            for suffix in ['trace_control_code','trace_scores','proposal_control_code']:
                if suffix in got:arrays[name+'__'+suffix]=got[suffix]
            if p['preflight'] and not i%2:
                repeated=model.decide(raw.copy(),rng_for(0,row['seed'],carrier,630),None)
                np.testing.assert_array_equal(repeated['control_code'],got['control_code'])
    decisions=dict(methods=METHODS,control_code=np.asarray(control),costs=np.asarray(cost),feedback={},teacher_objective_evaluations=None)
    scored=score_decisions(env,carrier,public,decisions,p['prior']['backend'],include_mrc=False)
    arrays.update(control_code=scored['control_code'],metrics=scored['metrics'])
    np.testing.assert_array_equal(arrays['metrics'][:,10],[16,64]*4)
    names=METHODS[1::2];subset={k:v for k,v in arrays.items() if k.startswith(tuple(n+'__' for n in names))}
    subset.update(public_X=raw,control_code=arrays['control_code'][1::2],metrics=arrays['metrics'][1::2]);audit_trace(subset,names,public,carrier)
    for i in range(4):np.testing.assert_array_equal(arrays['control_code'][2*i],arrays[METHODS[2*i+1]+'__proposal_control_code'])
    for model in STATE['models'][::2]:model.verify()
    return arrays


def one(row):
    output,p=STATE['output'],STATE['protocol'];folder=output/'records'/f"environment_{row['index']:05d}";folder.mkdir(parents=True,exist_ok=True);entries=[]
    for carrier in p['carriers']:
        path=folder/f'carrier_{carrier:02d}.npz';marker=path.with_suffix('.json')
        if marker.exists():
            e=read(marker);assert e['protocol_sha256']==fp(p) and sha256(path)==e['sha256']
        else:
            if path.exists():raise RuntimeError('Inspect uncommitted file '+str(path))
            a=compute(row,carrier);atomic_npz(path,**a)
            e=dict(environment_id=row['environment_id'],carrier_ghz=carrier,path=str(path.relative_to(output)),sha256=sha256(path),protocol_sha256=fp(p),
                checks=dict(public_observation_only=True,feedback_trace_audited=True,proposal_equal_direct=True,joint_correction=False))
            write_json(marker,e)
        entries.append(e)
    result=dict(index=row['index'],environment_id=row['environment_id'],carriers=entries);write_json(folder/'complete.json',result);return result


def run(project,role,workers,preflight):
    require_host();root=project/'dataset_simulation';training=root/'revision_af_20261029/C/training';done=read(training/'complete.json');assert done['status']=='complete'
    base=root/'revision_rq2_af_20261029/full';p0=read(base/'protocol.json');rows=[]
    if role=='test':
        for r in p0['rows']:
            folder=Path(p0['source'])/'records'/f"environment_{r['index']:05d}";marker=read(folder/'complete.json')
            rows.append(dict(r,environment_path=str(folder/'environment.json'),environment_sha256=marker['environment_sha256']))
        assert len(rows)==864
    else:
        relative,split,count,digest=SOURCES['validation'];manifest=root/relative;assert sha256(manifest)==digest
        records={r['environment_id']:r for r in read(manifest.parent/'records.json')}
        for r in read(manifest)['environments']:
            if r['split']==split:
                rows.append(dict(r,environment_path=str(manifest.parent/r['path']/'environment.json'),environment_sha256=records[r['environment_id']]['file_sha256']['environment.json']))
        assert len(rows)==216
    sources=dict(done['source_sha256'])
    sources['revision_af_20261029/evaluate_quality.py']=sha256(Path(__file__))
    sources['dataset_protocol/registry.py']=sha256(SOURCE/'dataset_protocol/registry.py')
    verify_sources(sources)
    artifacts={str(training/n):sha256(training/n) for n in ['complete.json','protocol.json','normalization.npz']}
    for name in CONFIGS:
        for n in ['complete.json','weights.pt']:artifacts[str(training/name/n)]=sha256(training/name/n)
    output=root/'revision_af_20261029/C'/(role+'_preflight' if preflight else role)
    p=dict(schema='quality-objective-control-reception-v1',role=role,preflight=preflight,
        rows=rows[:1] if preflight else rows,carriers=[4,12,20] if preflight else list(range(4,21)),
        methods=METHODS,prior=p0['prior'],anchor=str(base),input_record_source=p0['source'],training=str(training),source_sha256=sources,
        artifact_sha256=artifacts,code_sha256=sha256(Path(__file__)),joint_correction=False)
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
        (output/'workers').mkdir(exist_ok=True)
        if preflight:
            initialize(output,p);completed=[one(p['rows'][0])]
        else:
            proof=read(output.parent/(role+'_preflight')/'complete.json');assert proof['preflight'] and proof['source_sha256']==sources
            if role=='test':assert read(output.parent/'validation/complete.json')['environments']==216
            completed=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(output,p)) as pool:
                for future in as_completed([pool.submit(one,r) for r in p['rows']]):
                    completed.append(future.result());write_json(output/'progress.json',dict(status='running',completed=len(completed),total=len(p['rows']),pid=os.getpid(),at=now()))
                    print(f"C {role}: {len(completed)}/{len(p['rows'])}",flush=True)
        completed.sort(key=lambda r:r['index']);values=[]
        for r in completed:
            a=[]
            for e in r['carriers']:
                file=output/e['path'];assert sha256(file)==e['sha256']
                with np.load(file,allow_pickle=False) as f:a.append(f['metrics'].copy())
            values.append(a)
        values=np.asarray(values);assert values.shape==(len(completed),len(p['carriers']),8,13)
        point=metrics(sufficient(values).sum(0))
        write_json(output/'summary.json',[dict(method=name,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)}) for i,name in enumerate(METHODS)])
        atomic_npz(output/'environment_metrics.npz',values=values,methods=np.asarray(METHODS),environment_ids=np.asarray([r['environment_id'] for r in completed]));write_json(output/'records.json',completed)
        verify_sources(sources)
        for path,h in artifacts.items():assert sha256(path)==h
        write_json(output/'complete.json',dict(status='complete',role=role,preflight=preflight,environments=len(completed),carriers=len(p['carriers']),methods=METHODS,
            source_sha256=sources,code_sha256=p['code_sha256'],protocol_sha256=fp(p),joint_correction=False,at=now(),
            file_sha256={n:sha256(output/n) for n in ['protocol.json','summary.json','records.json','environment_metrics.npz']}))
        write_json(output/'progress.json',dict(status='complete',completed=len(completed),total=len(completed),at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--role',choices=['validation','test'],required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    try:run(a.project.resolve(),a.role,a.workers,a.preflight)
    except BaseException:
        out=a.project/'dataset_simulation/revision_af_20261029/C'/(a.role+'_preflight' if a.preflight else a.role);out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
