"""冻结后生成新留出信号并实际执行全部方法；旧案例演练不解封新计划。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (SOURCE, require_host, check_data, sha256, source_record,
    verify_sources, write_json, now, atomic_npz, LEVELS)
from study_full_baselines.runtime_bundle import load_models, verify as verify_bundle, fingerprint
from study_full_baselines.confirmation_freeze import register, validate, budget
from study_full_baselines.confirmation_receiver import public_observation, decide_all, score_decisions, METRIC_ORDER
from study_full_baselines.confirmation_batch import check_carrier, committed_environment, guard_load
from study_full_baselines.expand_training_gpu import validate_backend
from baseline_common.channel import make_environment, serialize_environment
from our_method_response_control.train import precision

STATE = {}


def read(path): return json.loads(Path(path).read_text())


def rehearsal_identity(project):
    """此入口固定旧测试0及4/12/20 GHz，不能传入新的环境或任意数据路径。"""
    root = project/'dataset_simulation'; data = root/'outputs/quality_rank_hybrid_20260925'
    manifest = check_data(data); rows = [r for r in manifest['environments'] if r['split']=='test' and r['index']==0]
    if len(rows) != 1: raise ValueError('旧演练环境缺失。')
    bundle = root/'diagnostics/20260926_runtime_bundle_preflight_run02/bundle'
    package = verify_bundle(bundle)
    if package['cohort']['train_environments'] != 864: raise ValueError('演练必须使用已核验原864模型包。')
    proof = read(root/'diagnostics/20260926_confirmation_receiver_preflight/summary.json')
    sources = {**proof['source_sha256'], **package['source_sha256'], **source_record([
        'study_full_baselines/'+n for n in ['confirmation_freeze.py','confirmation_execute.py',
            'FINAL_CONFIRMATION_PROTOCOL.md','confirmation_batch.py','paired_statistics.py']])}
    record = next(r for r in read(data/'records.json') if r['environment_id']==rows[0]['environment_id'])
    return dict(schema='runtime-confirmation-old-rehearsal-v1', scope='old_runtime_rehearsal',
        project=str(project), runtime_bundle=str(bundle), runtime_manifest_sha256=sha256(bundle/'manifest.json'),
        source_sha256=sources, rows=rows, carriers=[4,12,20], ordinary_methods=package['methods'],
        methods=package['methods']+['teacher','mrc'], metric_order=METRIC_ORDER,
        measurement_budgets={n:budget(n) for n in package['methods']},
        backend='cuda_fft_cpu_rk4_v1', backend_proof=validate_backend(root/'diagnostics/20260926_gpu_generation_replay'),
        data_path=str(data), input_sha256=record['file_sha256'],
        final_confirmation=False, new_environment_signals_generated=0)


def initialize_worker(output_string, identity):
    require_host(); precision()
    if not torch.cuda.is_available(): raise RuntimeError('完整在线执行要求华硕GPU。')
    verify_sources(identity['source_sha256']); bundle=Path(identity['runtime_bundle'])
    if sha256(bundle/'manifest.json') != identity['runtime_manifest_sha256']: raise ValueError('运行包改变。')
    with np.load(bundle/'public.npz') as f:
        public={k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    with patch('numpy.load', guard_load(np.load)): models=load_models(bundle,public)
    if list(models) != identity['ordinary_methods']: raise ValueError('实际方法集合改变。')
    output=Path(output_string)
    STATE.update(output=output,identity=identity,models=models,public=public)
    write_json(output/'workers'/('%d.json'%os.getpid()),dict(pid=os.getpid(),at=now(),
        gpu=torch.cuda.get_device_name(),ordinary_methods=len(models),threads=torch.get_num_threads(),
        runtime_manifest_sha256=identity['runtime_manifest_sha256'],identity_sha256=fingerprint(identity)))


def one_environment(row):
    identity=STATE['identity']; output=STATE['output']; public=STATE['public']; models=STATE['models']
    started=time.perf_counter(); folder=output/'records'/('environment_%05d'%row['index']);folder.mkdir(exist_ok=True)
    # 仅有正式冻结/登记之后，执行器才会得到新计划的row。演练row被入口固定为旧索引0。
    environment=serialize_environment(make_environment(row['seed'],row['factors']))
    old_inputs=None
    if identity['scope']=='old_runtime_rehearsal':
        source=Path(identity['data_path'])/row['path']
        for name,digest in identity['input_sha256'].items():
            if sha256(source/name)!=digest: raise ValueError('演练来源改变。')
        if read(source/'environment.json')!=environment: raise ValueError('环境工厂未复现旧参数。')
        with np.load(source/'data.npz') as f: old_inputs=f['X'].copy()
    elif identity['scope']!='fresh_confirmation': raise ValueError('不支持的执行范围。')
    env_path=folder/'environment.json'
    if env_path.exists():
        if read(env_path)!=environment: raise ValueError('续跑的传播环境不同。')
    else: write_json(env_path,environment)
    records=[]; computed=0; reused=0
    for carrier in identity['carriers']:
        committed=check_carrier(folder,carrier,row,identity)
        if committed is not None: records.append(committed);reused+=1;continue
        raw,observed=public_observation(environment,carrier,public,identity['backend'])
        if old_inputs is not None: np.testing.assert_array_equal(raw,old_inputs[carrier-4])
        with patch('numpy.load',guard_load(np.load)):
            decisions=decide_all(models,raw,observed,row['seed'],carrier,public)
        scored=score_decisions(environment,carrier,public,decisions,identity['backend'])
        if scored['methods']!=identity['methods']: raise ValueError('接收方法集合改变。')
        codes=scored['control_code'];metrics=scored['metrics']
        if codes.shape!=(len(identity['methods']),128) or metrics.shape!=(len(identity['methods']),len(METRIC_ORDER)):
            raise ValueError('输出维度不同。')
        if np.any(codes[:-1]<0) or np.any(codes[:-1]>LEVELS) or np.any(codes[-1]!=-1):
            raise ValueError('接收控制码非法。')
        np.testing.assert_array_equal(metrics[:,[1,4,6]],np.tile([496,248,8],(len(metrics),1)))
        for i,name in enumerate(identity['ordinary_methods']):
            if metrics[i,10]!=identity['measurement_budgets'][name]: raise ValueError('实际测量预算改变。')
        arrays=dict(public_X=raw,control_code=codes,metrics=metrics,
            mrc_physical_reference_snr_db=np.asarray(scored['mrc_physical_reference_snr_db']),
            teacher_objective_evaluations=np.asarray(scored['teacher_objective_evaluations']))
        for name,trace in scored['feedback'].items():
            for key,value in trace.items(): arrays[name+'__'+key]=value
        path=folder/('carrier_%02d.npz'%carrier);atomic_npz(path,**arrays)
        record=dict(environment_id=row['environment_id'],carrier_ghz=carrier,sha256=sha256(path),
            identity_sha256=fingerprint(identity),methods=len(identity['methods']),path=str(path.relative_to(output)),
            public_input_source='newly generated public16 measurements',old_input_bitwise_check=old_inputs is not None,
            worker_pid=os.getpid(),at=now())
        write_json(path.with_suffix('.json'),record);records.append(record);computed+=1
        write_json(output/'workers'/('%d_progress.json'%os.getpid()),dict(environment=row['index'],
            carrier=carrier,status='running',at=now()))
    verify_sources(identity['source_sha256'])
    result=dict(index=row['index'],environment_id=row['environment_id'],identity_sha256=fingerprint(identity),
        carriers=records,environment_sha256=sha256(env_path),worker_pid=os.getpid(),
        seconds=time.perf_counter()-started,newly_computed_carriers=computed,reused_carriers=reused,at=now())
    write_json(folder/'complete.json',result);return result


def run(project,output,workers,stop_after,freeze=None,rehearse=False):
    require_host()
    if workers not in [1,2,3,4] or stop_after<0: raise ValueError('worker须1到4，停止边界须非负。')
    if rehearse:
        if freeze is not None: raise ValueError('旧演练不能同时接收新冻结文件。')
        identity=rehearsal_identity(project)
    else:
        if freeze is None: raise ValueError('正式运行须提供经过验证的冻结文件。')
        identity=validate(freeze)
        if Path(identity['project']).resolve()!=project.resolve(): raise ValueError('冻结项目路径不同。')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('同一确认输出已有执行器。')
        if not rehearse:
            registered=register(freeze,output)
            if registered!=identity: raise ValueError('登记时冻结身份改变。')
        for name in ['records','workers','attempts']: (output/name).mkdir(exist_ok=True)
        protocol=output/'protocol.json'
        if protocol.exists():
            if read(protocol)!=identity: raise ValueError('续跑身份不同。')
        else:
            write_json(protocol,identity)
            for name in identity['source_sha256']:
                dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
        complete=[];pending=[]
        for row in identity['rows']:
            old=committed_environment(output,row,identity)
            if old is None: pending.append(row)
            else: complete.append(old)
        selected=pending[:stop_after] if stop_after else pending
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if selected and free<workers*3072: raise RuntimeError('本次worker数量的显存余量不足；保留断点。')
        attempt=output/'attempts'/('%d_%d.json'%(time.time_ns(),os.getpid()))
        state=dict(status='running',pid=os.getpid(),workers=workers,total=len(identity['rows']),completed=len(complete),
            reused_environments=len(complete),scheduled_indices=[r['index'] for r in selected],
            identity_sha256=fingerprint(identity),final_confirmation=identity['final_confirmation'],at=now())
        write_json(attempt,state);write_json(output/'progress.json',state)
        if selected:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize_worker,initargs=(str(output),identity)) as pool:
                jobs=[pool.submit(one_environment,row) for row in selected]
                for job in as_completed(jobs):
                    complete.append(job.result());write_json(output/'progress.json',{**state,'completed':len(complete),'at':now()})
        verify_sources(identity['source_sha256']);verify_bundle(Path(identity['runtime_bundle']))
        state.update(status='complete' if len(complete)==len(identity['rows']) else 'stopped_at_environment_boundary',
            completed=len(complete),newly_computed_environments=len(selected),at=now())
        write_json(attempt,state);write_json(output/'progress.json',state)
        write_json(output/'records.json',sorted(complete,key=lambda r:r['index']))
        if len(complete)==len(identity['rows']):
            write_json(output/'complete.json',dict(status='complete_reception_records',environments=len(complete),
                methods=len(identity['methods']),carriers=len(identity['carriers']),
                cases=len(complete)*len(identity['methods'])*len(identity['carriers']),
                identity_sha256=fingerprint(identity),records_sha256=sha256(output/'records.json'),
                final_confirmation=identity['final_confirmation'],scope=identity['scope'],
                independent_audit_and_statistics_and_timing_pending=True,at=now()))
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','output']: p.add_argument('--'+name,type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',type=Path);group.add_argument('--rehearse-old',action='store_true')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--stop-after',type=int,default=0)
    a=p.parse_args()
    try: run(a.project.resolve(),a.output.resolve(),a.workers,a.stop_after,a.freeze,a.rehearse_old)
    except BaseException:
        if a.output.exists(): write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
