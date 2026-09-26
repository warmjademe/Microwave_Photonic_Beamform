"""联合频率候选的旧216全量探索；保持原物理模型、权重和预算。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE,require_host,sha256,source_record,verify_sources,write_json,now,LEVELS
from study_full_baselines.runtime_bundle import verify,fingerprint
from study_full_baselines import confirmation_batch as batch
from study_full_baselines.analyze_confirmation import check_values
from our_method_response_control.train import precision
from our_method_response_control.physics import decode
from our_method_joint_refinement.online_full import bind_joint,models_for,METHODS
from our_method_joint_refinement.method import refine_joint
from our_method_joint_refinement.analyze_full import overlap


def read(path):return json.loads(Path(path).read_text())


def identity_for(project,preflight):
    old=project/'dataset_simulation/baseline_results/20260925_full_baselines/evaluation_fair3456_online'
    previous=read(old/'protocol.json');complete=read(old/'analysis/complete.json')
    if (complete['status']!='complete_exploratory_analysis'
            or previous['common_training_environments']!=3456 or len(previous['rows'])!=216):
        raise ValueError('共同规模全基线尚未完成。')
    for name,digest in complete['file_sha256'].items():
        if sha256(old/'analysis'/name)!=digest:raise ValueError('原全基线分析产物改变。')
    package=verify(Path(previous['runtime_bundle']));bound=bind_joint(project,package)
    sources={**previous['source_sha256'],**bound['source_sha256']}
    sources.update(source_record(['our_method_joint_refinement/'+n for n in
        ['evaluate_full.py','online_full.py','analyze_full.py','FULL_PROTOCOL.md']]))
    verify_sources(sources)
    identity={**previous,'schema':'joint-frequency-full-exploratory-v1',
        'scope':'old216_preflight' if preflight else 'old216_exploratory',
        'research_stage':'joint_frequency_refinement',
        'rows':previous['rows'][:1] if preflight else previous['rows'],
        'carriers':[4,12,20] if preflight else list(range(4,21)),
        'ordinary_methods':METHODS,'methods':METHODS+['teacher','mrc'],
        'measurement_budgets':{n:16 for n in METHODS},'source_sha256':sources,'joint_binding':bound,
        'previous_evaluation':str(old),'previous_protocol_sha256':sha256(old/'protocol.json'),
        'previous_analysis_complete_sha256':sha256(old/'analysis/complete.json')}
    return identity


def initialize(output,identity):
    require_host();precision();verify_sources(identity['source_sha256'])
    bundle=Path(identity['runtime_bundle'])
    if sha256(bundle/'manifest.json')!=identity['runtime_manifest_sha256']:raise ValueError('模型包改变。')
    verify(bundle)
    with np.load(bundle/'public.npz') as f:public={k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    with patch('numpy.load',batch.guard_load(np.load)):models=models_for(bundle,public,identity['joint_binding'])
    batch.STATE.update(output=Path(output),identity=identity,public=public,models=models)
    write_json(Path(output)/'workers'/('%d.json'%os.getpid()),dict(pid=os.getpid(),at=now(),
        methods=list(models),gpu=torch.cuda.get_device_name(),threads=torch.get_num_threads()))


def preflight_checks(output,identity):
    initialize(output,identity);public=batch.STATE['public'];models=batch.STATE['models'];checked=0
    prior=overlap(output,identity)
    with torch.inference_mode():
        for fc in identity['carriers']:
            with np.load(output/'records/environment_00000'/('carrier_%02d.npz'%fc)) as f:arrays={k:f[k].copy() for k in f.files}
            check_values(arrays,fc,identity,public);x=arrays['public_X']
            for a,base in [('covariance','covariance_response'),('cnn','complex_response_cnn')]:
                h=models[base].estimate(x)
                for suffix in ['joint_block_noise','joint_full_noise']:
                    name=a+'__'+suffix;model=models[name]
                    corrected,_=refine_joint(x,public['pilot_qpsk'],h,model.error_spatial,model.error_frequency,model.full_noise)
                    start=public['probe_controls'][int(x[1985:2001].argmax())]
                    u,_=decode(corrected,fc,start,sweeps=2)
                    np.testing.assert_array_equal(np.rint(u*LEVELS),arrays['control_code'][identity['methods'].index(name)])
                    checked+=1
    write_json(output/'preflight_checks.json',dict(status='passed',at=now(),cases=30,
        explicit_joint_path_checks=checked,overlap=prior,source_sha256=identity['source_sha256'],
        runtime_manifest_sha256=identity['runtime_manifest_sha256'],
        joint_fit_complete_sha256=sha256(Path(identity['joint_binding']['path'])/'fit_complete.json')))


def run(a):
    require_host();project=a.project.resolve();output=a.output.resolve();identity=identity_for(project,a.preflight)
    if not a.preflight:
        if a.preflight_evidence is None:raise ValueError('须先完成新在线适配检查。')
        proof=read(a.preflight_evidence)
        if (proof['status']!='passed' or proof['cases']!=30 or proof['source_sha256']!=identity['source_sha256']
                or proof['runtime_manifest_sha256']!=identity['runtime_manifest_sha256']):
            raise ValueError('前置检查身份不同。')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for n in ['records','workers','attempts']:(output/n).mkdir(exist_ok=True)
        protocol=output/'protocol.json'
        if protocol.exists():
            if read(protocol)!=identity:raise ValueError('续跑身份不同。')
        else:
            write_json(protocol,identity)
            for n in identity['source_sha256']:
                dest=output/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/n,dest)
        completed=[];pending=[]
        for row in identity['rows']:
            old=batch.committed_environment(output,row,identity)
            if old is None:pending.append(row)
            else:completed.append(old)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if pending and free<a.workers*3072:raise RuntimeError('显存不足，保留断点。')
        state=dict(status='running',at=now(),pid=os.getpid(),workers=a.workers,completed=len(completed),
            total=len(identity['rows']),scope=identity['scope'],final_confirmation=False,
            identity_sha256=fingerprint(identity))
        attempt=output/'attempts'/('%d_%d.json'%(time.time_ns(),os.getpid()))
        write_json(attempt,state);write_json(output/'progress.json',state)
        if pending:
            with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(str(output),identity)) as pool:
                for job in as_completed([pool.submit(batch.one,row) for row in pending]):
                    completed.append(job.result());state.update(completed=len(completed),at=now())
                    write_json(output/'progress.json',state);print(json.dumps(state),flush=True)
        verify_sources(identity['source_sha256']);verify(Path(identity['runtime_bundle']))
        write_json(output/'records.json',sorted(completed,key=lambda r:r['index']))
        write_json(output/'complete.json',dict(status='complete_exploratory_reception',at=now(),
            environments=len(completed),carriers=len(identity['carriers']),methods=len(identity['methods']),
            cases=len(completed)*len(identity['carriers'])*len(identity['methods']),
            identity_sha256=fingerprint(identity),records_sha256=sha256(output/'records.json'),
            final_confirmation=False,scope=identity['scope']))
        state.update(status='complete',at=now());write_json(output/'progress.json',state);write_json(attempt,state)
        if a.preflight:preflight_checks(output,identity)
        else:subprocess.run([sys.executable,'-u','our_method_joint_refinement/analyze_full.py',
                             '--evaluation',str(output),'--output',str(output/'analysis')],cwd=SOURCE,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['project','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=4)
    p.add_argument('--preflight',action='store_true');p.add_argument('--preflight-evidence',type=Path);a=p.parse_args()
    try:run(a)
    except BaseException:
        if a.output.exists():write_json(a.output/('failure_%d.json'%time.time()),dict(at=now(),traceback=traceback.format_exc()))
        raise
