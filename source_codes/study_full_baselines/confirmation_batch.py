"""确认接收器的并行与续跑执行层；当前CLI只允许固定旧环境演练。

最终方法、共同训练成员、统计计划尚未冻结，故本版本拒绝新留出计划。
逐载频提交输入、控制、反馈轨迹和接收指标，续跑核对来源后复用已提交记录。
"""
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
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_name] = '1'
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.online_controller import OnlineController, method_names
from study_full_baselines.confirmation_receiver import public_observation, decide_all, score_decisions, METRIC_ORDER
from our_method_response_control.train import precision
from study_full_baselines.expand_training_gpu import validate_backend

STATE = {}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def guard_load(original):
    def guarded(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name in [
            'data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy']:
            raise RuntimeError('在线控制阶段禁止读取测试缓存或隐藏数据。')
        return original(path, *args, **kwargs)
    return guarded


def verify_artifacts(artifacts):
    unique = {}
    for files in artifacts.values():
        for path, digest in files.items():
            if path in unique and unique[path] != digest: raise ValueError('同一训练产物出现两个身份。')
            unique[path] = digest
    for path, digest in unique.items():
        if sha256(path) != digest: raise ValueError('模型权重或统计参数改变。')


def prepare_worker(project_string, output_string, identity):
    require_host(); precision()
    if not torch.cuda.is_available(): raise RuntimeError('要求华硕GPU。')
    project = Path(project_string); output = Path(output_string)
    verify_sources(identity['source_sha256']); verify_artifacts(identity['model_artifacts'])
    public_path = Path(identity['public_path'])
    if sha256(public_path) != identity['public_sha256']: raise ValueError('公开设置改变。')
    with np.load(public_path) as f:
        public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    with patch('numpy.load', guard_load(np.load)):
        models = {name: OnlineController(project, name, public) for name in identity['ordinary_methods']}
    for name, model in models.items():
        if model.artifacts != identity['model_artifacts'][name]: raise ValueError('worker未加载约定权重：'+name)
    STATE.update(project=project, output=output, identity=identity, public=public, models=models)
    write_json(output/'workers'/('%d.json' % os.getpid()), dict(pid=os.getpid(), at=now(),
        cuda_device=torch.cuda.get_device_name(), methods=len(models), cpu_threads=torch.get_num_threads(),
        identity_sha256=fingerprint(identity), model_loading='once per spawned worker',
        model_artifact_sets_exact=True))


def check_carrier(folder, carrier, row, identity):
    path = folder/('carrier_%02d.npz' % carrier); marker = path.with_suffix('.json')
    if marker.exists():
        record = json.loads(marker.read_text())
        if (record['identity_sha256'] != fingerprint(identity) or record['environment_id'] != row['environment_id']
                or record['carrier_ghz'] != carrier or sha256(path) != record['sha256']):
            raise ValueError('已提交载频记录身份不同。')
        return record
    if path.exists(): raise ValueError('发现未提交载频文件，必须先核查：'+str(path))
    return None


def one(row):
    output = STATE['output']; identity = STATE['identity']; public = STATE['public']; models = STATE['models']
    data = Path(identity['data_path']); start = time.perf_counter()
    folder = output/'records'/('environment_%05d' % row['index']); folder.mkdir(exist_ok=True)
    expected = identity['input_sha256'][row['environment_id']]
    for name, digest in expected.items():
        if sha256(data/row['path']/name) != digest: raise ValueError('演练输入改变。')
    # 演练只读取旧环境，不提供生成新环境的入口。
    environment = json.loads((data/row['path']/'environment.json').read_text())
    env_path = folder/'environment.json'
    if env_path.exists():
        if json.loads(env_path.read_text()) != environment: raise ValueError('续跑环境改变。')
    else: write_json(env_path, environment)
    with np.load(data/row['path']/'data.npz') as f: old_inputs = f['X'].copy()
    committed = []; newly_computed = 0; reused = 0
    for carrier in identity['carriers']:
        record = check_carrier(folder, carrier, row, identity)
        if record is not None:
            committed.append(record); reused += 1; continue
        raw, observed = public_observation(environment, carrier, public, identity['backend'])
        np.testing.assert_array_equal(raw, old_inputs[carrier-4])
        with patch('numpy.load', guard_load(np.load)):
            decisions = decide_all(models, raw, observed, row['seed'], carrier, public)
        scored = score_decisions(environment, carrier, public, decisions, identity['backend'])
        if scored['methods'] != identity['methods']: raise ValueError('方法输出顺序不同。')
        arrays = dict(public_X=raw, control_code=scored['control_code'], metrics=scored['metrics'],
            mrc_physical_reference_snr_db=np.asarray(scored['mrc_physical_reference_snr_db']),
            teacher_objective_evaluations=np.asarray(scored['teacher_objective_evaluations']))
        for name, trace in scored['feedback'].items():
            for key, value in trace.items(): arrays[name+'__'+key] = value
        path = folder/('carrier_%02d.npz' % carrier); atomic_npz(path, **arrays)
        record = dict(environment_id=row['environment_id'], carrier_ghz=carrier, sha256=sha256(path),
            identity_sha256=fingerprint(identity), methods=len(scored['methods']),
            path=str(path.relative_to(output)), input_rebuilt_equal=True, worker_pid=os.getpid(), at=now())
        write_json(path.with_suffix('.json'), record); committed.append(record); newly_computed += 1
        write_json(output/'workers'/('%d_progress.json' % os.getpid()), dict(status='running',
            environment=row['index'], carrier=carrier, committed_carriers=len(committed), at=now()))
    verify_sources(identity['source_sha256'])
    result = dict(index=row['index'], environment_id=row['environment_id'], identity_sha256=fingerprint(identity),
        carriers=committed, environment_sha256=sha256(env_path), worker_pid=os.getpid(),
        seconds=time.perf_counter()-start, newly_computed_carriers=newly_computed, reused_carriers=reused, at=now())
    write_json(folder/'complete.json', result)
    return result


def committed_environment(output, row, identity):
    folder = output/'records'/('environment_%05d' % row['index']); marker = folder/'complete.json'
    if not marker.exists(): return None
    value = json.loads(marker.read_text())
    if (value['index'] != row['index'] or value['environment_id'] != row['environment_id']
            or value['identity_sha256'] != fingerprint(identity)
            or value['environment_sha256'] != sha256(folder/'environment.json')):
        raise ValueError('完整环境提交身份不同。')
    records = [check_carrier(folder, fc, row, identity) for fc in identity['carriers']]
    if any(r is None for r in records) or records != value['carriers']: raise ValueError('完整环境缺少载频。')
    return value


def rehearsal_identity(project):
    root = project/'dataset_simulation'; data = root/'outputs/quality_rank_hybrid_20260925'
    manifest = check_data(data); rows = [r for r in manifest['environments'] if r['split'] == 'test' and r['index'] in [0, 1, 2]]
    if [r['index'] for r in rows] != [0, 1, 2]: raise ValueError('演练仅允许固定前三个旧测试环境。')
    proof_path = root/'diagnostics/20260926_confirmation_receiver_preflight'
    proof = json.loads((proof_path/'summary.json').read_text())
    audit = json.loads((proof_path/'integrity_verification.json').read_text())
    if audit['cases'] != 135 or audit['summary_sha256'] != sha256(proof_path/'summary.json'):
        raise ValueError('接收器前置核查尚未通过。')
    verify_sources(proof['source_sha256']); verify_artifacts(proof['model_artifacts'])
    backend_proof = validate_backend(root/'diagnostics/20260926_gpu_generation_replay')
    records = {r['environment_id']: r for r in json.loads((data/'records.json').read_text())}
    ordinary = method_names([216, 432])
    if set(ordinary) != set(proof['model_artifacts']): raise ValueError('演练方法与前置核查不符。')
    sources = {**proof['source_sha256'], **source_record(['study_full_baselines/confirmation_batch.py',
        'study_full_baselines/CONFIRMATION_BATCH_PROTOCOL.md'])}
    return dict(schema='confirmation-batch-old-rehearsal-v1', scope='old_test_rehearsal',
        source_sha256=sources, model_artifacts=proof['model_artifacts'], rows=rows, carriers=[4, 12, 20],
        ordinary_methods=ordinary, methods=ordinary+['teacher', 'mrc'], metric_order=METRIC_ORDER,
        backend='cuda_fft_cpu_rk4_v1', backend_proof=backend_proof,
        data_path=str(data), data_manifest_sha256=sha256(data/'manifest.json'), public_path=str(data/'public.npz'),
        public_sha256=sha256(data/'public.npz'), input_sha256={r['environment_id']: records[r['environment_id']]['file_sha256'] for r in rows},
        final_confirmation=False, new_environment_signals_generated=0,
        receiver_preflight_sha256=sha256(proof_path/'summary.json'))


def run(project, output, workers, stop_after):
    require_host()
    if workers not in [1, 2] or stop_after < 0: raise ValueError('本演练限定1或2个worker。')
    identity = rehearsal_identity(project)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('同一批次已有执行器运行，禁止重复启动。')
        for name in ['records', 'workers', 'attempts']: (output/name).mkdir(exist_ok=True)
        protocol = output/'protocol.json'
        if protocol.exists():
            if json.loads(protocol.read_text()) != identity: raise ValueError('续跑协议、代码或模型改变。')
        else:
            write_json(protocol, identity)
            for name in identity['source_sha256']:
                dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE/name, dest)
        completed = []; pending = []
        for row in identity['rows']:
            committed = committed_environment(output, row, identity)
            if committed is None: pending.append(row)
            else: completed.append(committed)
        reused_environments = len(completed); selected = pending[:stop_after] if stop_after else pending
        attempt = output/'attempts'/('%d_%d.json' % (time.time_ns(), os.getpid()))
        state = dict(status='running', completed=len(completed), total=len(identity['rows']),
            reused_environments=reused_environments, scheduled_indices=[r['index'] for r in selected],
            workers=workers, pid=os.getpid(), identity_sha256=fingerprint(identity), at=now())
        write_json(attempt, state); write_json(output/'progress.json', state)
        if selected:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=prepare_worker, initargs=(str(project), str(output), identity)) as pool:
                tasks = [pool.submit(one, row) for row in selected]
                for job in as_completed(tasks):
                    completed.append(job.result())
                    write_json(output/'progress.json', {**state, 'completed': len(completed), 'at': now()})
        verify_sources(identity['source_sha256']); verify_artifacts(identity['model_artifacts'])
        complete = len(completed) == len(identity['rows'])
        state.update(status='complete' if complete else 'stopped_at_requested_environment_boundary',
            completed=len(completed), newly_computed_environments=len(selected), at=now())
        write_json(attempt, state); write_json(output/'progress.json', state)
        write_json(output/'records.json', sorted(completed, key=lambda r: r['index']))
        if complete:
            write_json(output/'complete.json', dict(status='complete_old_rehearsal', environments=len(completed),
                ordinary_methods=len(identity['ordinary_methods']), reference_methods=2,
                cases=len(completed)*len(identity['carriers'])*len(identity['methods']),
                identity_sha256=fingerprint(identity), records_sha256=sha256(output/'records.json'),
                final_confirmation=False, new_environment_signals_generated=0, at=now()))
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--stop-after', type=int, default=0)
    # 不接受新环境、模型选择或冻结文件参数，防止在最终方案未确定时提前揭晓。
    args = parser.parse_args()
    try: run(args.project.resolve(), args.output.resolve(), args.workers, args.stop_after)
    except BaseException:
        if args.output.exists(): write_json(args.output/('failure_%d.json' % time.time()),
                                           dict(traceback=traceback.format_exc(), at=now()))
        raise
