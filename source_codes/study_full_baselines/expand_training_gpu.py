"""核查通过后的GPU FFT扩展入口；可接续CPU完整环境，不改采样或删除旧结果。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.expand_training import plan
from study_full_baselines.replay_gpu_generation import generate_arrays
from baseline_common.channel import make_environment,serialize_environment


EXTRA_SOURCES=['study_full_baselines/expand_training_gpu.py',
    'study_full_baselines/expand_training.py','study_full_baselines/common.py',
    'study_full_baselines/replay_gpu_generation.py','study_full_baselines/GPU_FFT_PROTOCOL.md',
    'diagnostics/gpu_fft_hybrid.py','generate_dataset.py',
    'baseline_common/channel.py','baseline_common/config.py']


def validate_backend(gates):
    result=json.loads((gates/'summary.json').read_text())
    protocol=json.loads((gates/'protocol.json').read_text())
    if (result['status']!='complete' or not result['passed_declared_gates']
            or not result['all_arrays_bitwise_equal'] or len(result['records'])!=6
            or protocol['carriers']!=list(range(4,21)) or protocol['frames']!=list(range(5))):
        raise ValueError('完整六分层、17频率和五帧重放必须通过，存储数组须逐位一致。')
    verify_sources(protocol['source_sha256'])
    for row in result['records']:
        if sha256(gates/row['result_file'])!=row['result_sha256']:
            raise ValueError('GPU核查输出文件改变。')
    return dict(protocol_sha256=sha256(gates/'protocol.json'),
                summary_sha256=sha256(gates/'summary.json'))


def conflicting_generators(output):
    """禁止两个生成入口同时写同一输出，包括原CPU入口。"""
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        pid=int(path.parent.name)
        if pid==os.getpid():continue
        try:args=path.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        texts=[x.decode(errors='replace') for x in args]
        if (str(output) in texts and any(x.endswith(('expand_training.py','expand_training_gpu.py')) for x in texts)):
            found.append(pid)
    return found


def one(root_string,row):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('要求GPU，不静默回退。')
    root=Path(root_string);started=time.perf_counter()
    with np.load(root/'public.npz') as f:public={k:f[k].copy() for k in f.files}
    env=serialize_environment(make_environment(row['seed'],row['factors']))
    arrays=generate_arrays(env,public)
    folder=root/row['path'];folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'environment.json',env);atomic_npz(folder/'data.npz',**arrays)
    record=dict(environment_id=row['environment_id'],path=row['path'],samples=17,
        file_sha256={name:sha256(folder/name) for name in ['environment.json','data.npz']},
        seconds=time.perf_counter()-started,execution_backend='cuda_fft_cpu_rk4_v1')
    write_json(folder/'complete.json',record);return record


def run(project,base,output,target,seed,workers,gates,plan_only=False):
    require_host();proof=validate_backend(gates)
    old,retained,added,rows,designs=plan(project,base,target,seed)
    sources={**old['source_sha256'],**source_record(EXTRA_SOURCES)}
    identity=dict(base_manifest_sha256=sha256(base/'manifest.json'),target_train_environments=target,
        additional_master_seed=seed,source_sha256=sources,public_sha256=sha256(base/'public.npz'),
        backend='cuda_fft_cpu_rk4_v1',backend_validation=proof)
    if plan_only:
        print(json.dumps(dict(status='plan_passed',retained=len(retained),added=len(added),
            total=target,strata=216,seed=seed,validation=proof,
            currently_conflicting_generators=conflicting_generators(output))),flush=True)
        return
    conflicts=conflicting_generators(output)
    if conflicts:raise RuntimeError('同目录仍有运行的生成器：'+str(conflicts))
    output.mkdir(parents=True,exist_ok=True)
    gpu_identity=output/'gpu_expansion_identity.json'
    if gpu_identity.exists():
        if json.loads(gpu_identity.read_text())!=identity:raise ValueError('GPU续跑来源改变。')
        manifest=json.loads((output/'manifest.json').read_text())
    else:
        if (output/'manifest.json').exists():
            manifest=json.loads((output/'manifest.json').read_text())
            if manifest['status']=='complete':raise ValueError('已完整完成的CPU扩展不应改写。')
            if manifest['environments']!=rows:raise ValueError('CPU/GPU扩展环境计划不同。')
            previous=manifest['expansion_identity']
            if (previous['base_manifest_sha256']!=identity['base_manifest_sha256'] or
                    previous['additional_master_seed']!=seed or previous['target_train_environments']!=target):
                raise ValueError('CPU扩展来源身份不符。')
            verify_sources(manifest['source_sha256'])
            shutil.copyfile(output/'manifest.json',output/'manifest_before_gpu.json')
            if (output/'progress.json').exists():
                shutil.copyfile(output/'progress.json',output/'progress_before_gpu.json')
        else:
            shutil.copyfile(base/'public.npz',output/'public.npz')
            manifest={**old,'status':'running','schema':'quality-rank-hybrid-training-expansion-v1',
                'created_at_utc':now(),'environments':rows,'train_samples':target*17,'test_samples':0,
                'master_seed':seed,'expansion_identity':identity,
                'sampling_design':{'retained_from':str(base),'additional':designs['train']},
                'test_scope':'training-only expansion; final confirmation not generated here'}
            manifest.pop('finished_at_utc',None);manifest.pop('records_sha256',None)
        write_json(gpu_identity,identity)
        manifest.update(status='running',source_sha256=sources,gpu_execution=identity,
            execution_note='Retain verified complete records; new records use validated float64 CUDA FFT and unchanged CPU RK4. See per-record backend and transition record.')
        write_json(output/'manifest.json',manifest)
        for name,digest in sources.items():
            dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and sha256(dest)!=digest:
                raise ValueError('禁止覆盖既有不同源码快照：'+name)
            if not dest.exists():shutil.copyfile(SOURCE/name,dest)
    if sha256(output/'public.npz')!=identity['public_sha256']:raise ValueError('公开设置改变。')
    previous={r['environment_id']:r for r in json.loads((base/'records.json').read_text())}
    records=[];todo=[];started=time.perf_counter()
    for row in rows:
        folder=output/row['path']
        if row['index']<len(retained) and not (folder/'complete.json').exists():
            if folder.exists():raise ValueError('不完整目录需先核查：'+str(folder))
            record=previous[row['environment_id']];source=base/row['path'];folder.mkdir(parents=True)
            for name,digest in record['file_sha256'].items():
                if sha256(source/name)!=digest:raise ValueError('保留成员来源哈希改变。')
                shutil.copyfile(source/name,folder/name)
            write_json(folder/'complete.json',record)
        marker=folder/'complete.json'
        if marker.exists():
            record=json.loads(marker.read_text())
            if record['environment_id']!=row['environment_id'] or record['path']!=row['path']:
                raise ValueError('成员身份不同。')
            for name,digest in record['file_sha256'].items():
                if sha256(folder/name)!=digest:raise ValueError('完整成员哈希改变。')
            records.append(record)
        else:
            if folder.exists():raise ValueError('不完整目录需先核查：'+str(folder))
            todo.append(row)
    def progress(status):
        state=dict(status=status,completed_environments=len(records),total_environments=len(rows),
            retained_environments=len(retained),added_environments=len(added),workers=workers,
            seconds=time.perf_counter()-started,pid=os.getpid(),backend='cuda_fft_cpu_rk4_v1',at=now())
        write_json(output/'progress.json',state);return state
    progress('running')
    # CUDA不在fork后的进程复用父上下文，每个worker由spawn建立独立上下文。
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(one,str(output),row) for row in todo]
        for job in as_completed(jobs):
            records.append(job.result());state=progress('running')
            if len(records)%20==0:print(json.dumps(state),flush=True)
    verify_sources(sources);validate_backend(gates);check_data(base)
    write_json(output/'records.json',sorted(records,key=lambda r:r['environment_id']))
    manifest.update(status='complete',finished_at_utc=now(),records_sha256=sha256(output/'records.json'),
        expansion_seconds=time.perf_counter()-started)
    write_json(output/'manifest.json',manifest);check_data(output);progress('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','base','output','gates']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--target',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    try:run(a.project,a.base,a.output,a.target,a.seed,a.workers,a.gates,a.plan_only)
    except BaseException:
        if not a.plan_only and a.output.exists():
            write_json(a.output/('gpu_failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
