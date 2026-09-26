"""联合线性对照、两阶段控制与大规模模型的统一旧测试接收评分。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.evaluate_learned import summarize, METRICS
from our_method_response_control.physics import covariance_estimate, decode
from our_method_two_stage.decode_multistart import decode_multistart
_STATE = None


def bundle_for(project, study, suite):
    """绑定固定预测；不按接收结果选候选、不读测试响应标签。"""
    methods = []; artifacts = {}
    def add(name, kind, path, digest, controllers):
        path = Path(path)
        if sha256(path)!=digest:
            raise ValueError('响应预测或协方差文件改变：'+str(path))
        artifacts[str(path)] = digest
        for controller in controllers:
            methods.append(dict(name=name+'__'+controller, kind=kind, path=str(path),
                controller=controller, probes=16))
    if suite=='components':
        targets = project/'dataset_simulation/diagnostics/20260925_response_control_targets'
        meta = json.loads((targets/'complete.json').read_text())
        add('covariance', 'covariance', targets/'training_covariance.npy', meta['covariance_sha256'],
            ['base_2sweeps', 'single_10sweeps', 'multi_mmse'])
        cnn = project/'dataset_simulation/baseline_results/20260925_response_control'
        meta = json.loads((cnn/'inference.json').read_text())
        add('cnn', 'response', cnn/'predicted_response.npy', meta['predicted_response_sha256'],
            ['base_2sweeps', 'single_10sweeps', 'multi_mmse'])
        linear = study/'joint_linear'
        if json.loads((linear/'complete.json').read_text())['status']!='complete':
            raise ValueError('联合线性模型尚未完成。')
        for name in ['per_tone_relative', 'joint_relative', 'joint_absolute']:
            meta = json.loads((linear/name/'complete.json').read_text())
            add(name, 'response', linear/name/'predicted_response.npy', meta['prediction_sha256'],
                ['base_2sweeps', 'multi_mmse'])
    else:
        count = int(suite.split('_')[1]); scale = study/('scale_%d'%count)
        if json.loads((scale/'progress.json').read_text())['status']!='complete':
            raise ValueError('该规模模型尚未完成。')
        meta = json.loads((scale/'targets/complete.json').read_text())
        add('covariance_n%d'%count, 'covariance', scale/'targets/training_covariance.npy',
            meta['covariance_sha256'], ['base_2sweeps', 'multi_mmse'])
        for schedule in ['fixed_epochs', 'equal_updates']:
            name = 'response_n%04d_%s'%(count, schedule); folder = scale/name
            meta = json.loads((folder/'complete.json').read_text())
            add(name, 'response', folder/'predicted_response.npy', meta['prediction_sha256'],
                ['base_2sweeps', 'multi_mmse'])
    return dict(methods=methods, artifact_sha256=artifacts, suite=suite,
        scope='old216 exploratory comparisons; final unseen864 not loaded',
        timing='CPU estimate/decoder only; cached responses exclude actual model inference')


def init(data, bundle):
    global _STATE
    paths = {m['path'] for m in bundle['methods']}
    _STATE = dict(data=Path(data), public=public_data(data), bundle=bundle,
        arrays={p:np.load(p, mmap_mode='r') for p in paths})


def controls_for(x, sample, public, bundle, arrays):
    """在线边界仅包含公开观测、固定模型输出与控制器设置。"""
    fc = int(x[1984]); initial = public['probe_controls'][int(x[1985:2001].argmax())]
    controls = []; seconds = []
    for m in bundle['methods']:
        tick = time.perf_counter()
        if m['kind']=='covariance':
            h = covariance_estimate(x, public['pilot_qpsk'], arrays[m['path']])
        else:
            h = arrays[m['path']][sample]
        if m['controller']=='multi_mmse':
            u, _ = decode_multistart(h, fc, initial)
        else:
            sweeps = 2 if m['controller']=='base_2sweeps' else 10
            u, _ = decode(h, fc, initial, sweeps=sweeps)
        seconds.append(time.perf_counter()-tick); controls.append(u)
    return np.asarray(controls), np.asarray(seconds)


def one(row, output, fingerprint):
    state = _STATE; data = state['data']; public = state['public']; bundle = state['bundle']
    path = Path(output)/'records'/('environment_%05d.npz'%row['index']); marker = path.with_suffix('.json')
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta['fingerprint']!=fingerprint or sha256(path)!=meta['sha256']:
            raise ValueError('已有接收评分身份不同。')
        return meta
    if path.exists():
        raise ValueError('有未提交评分文件，须先检查。')
    with np.load(data/row['path']/'data.npz') as f:
        x = f['X']
    tick = time.perf_counter()
    decisions = [controls_for(x[ci], row['index']*17+ci, public, bundle, state['arrays'])
                 for ci in range(17)]
    env = json.loads((data/row['path']/'environment.json').read_text())
    scores = []; codes = []
    for ci, fc in enumerate(range(4, 21)):
        controls, seconds = decisions[ci]
        engine, payload = frame_engine(env, fc, public['pilot_qpsk'], 5)
        clean = clean_frame_engine(env, fc, public['pilot_qpsk'], payload)
        metrics, code = reception_metrics(engine, clean, controls, payload, row['seed'], fc)
        scores.append(np.column_stack([metrics, np.full(len(controls), 16), seconds])); codes.append(code)
    atomic_npz(path, metrics=np.asarray(scores), control_code=np.asarray(codes),
        environment_id=np.asarray(row['environment_id']))
    record = dict(index=row['index'], environment_id=row['environment_id'], path=path.name,
        sha256=sha256(path), fingerprint=fingerprint, seconds=time.perf_counter()-tick, at=now())
    write_json(marker, record); return record


def run(project, data, study, output, suite, workers):
    require_host(); manifest = check_data(data)
    rows = [r for r in manifest['environments'] if r['split']=='test']
    if len(rows)!=216:
        raise ValueError('该入口仅用于旧216环境的探索性比较。')
    bundle = bundle_for(project, study, suite)
    protocol = dict(bundle=bundle, data_manifest_sha256=sha256(data/'manifest.json'),
        metric_order=METRICS, frame=5, apd_draws=8, source_sha256=source_record([
            'study_full_baselines/evaluate_response_variants.py', 'study_full_baselines/evaluate_learned.py',
            'study_full_baselines/common.py', 'our_method_response_control/physics.py',
            'our_method_two_stage/control.py', 'our_method_two_stage/decode_multistart.py']))
    fp = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True); (output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:
            raise ValueError('接收评分协议改变。')
    else:
        write_json(output/'protocol.json', protocol)
        for name in protocol['source_sha256']:
            dst = output/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE/name, dst)
    records = []; tick = time.perf_counter()
    write_json(output/'progress.json', dict(status='running', completed=0, total=len(rows),
        pid=os.getpid(), at=now()))
    with ProcessPoolExecutor(max_workers=workers, initializer=init, initargs=(str(data), bundle)) as pool:
        jobs = [pool.submit(one, row, str(output), fp) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result())
            progress = dict(status='running', completed=len(records), total=len(rows),
                seconds=time.perf_counter()-tick, pid=os.getpid(), at=now())
            write_json(output/'progress.json', progress)
            if len(records)%10==0:
                print(json.dumps(progress), flush=True)
    write_json(output/'records.json', sorted(records, key=lambda r:r['index']))
    summarize(output, rows, records, bundle['methods'])
    verify_sources(protocol['source_sha256']); check_data(data)
    for path, digest in bundle['artifact_sha256'].items():
        if sha256(path)!=digest:
            raise ValueError('响应产物在评分中改变。')
    write_json(output/'progress.json', dict(status='complete', completed=len(rows), total=len(rows), at=now()))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project','data','study','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--suite', choices=['components','scaling_1728','scaling_3456'], required=True)
    p.add_argument('--workers', type=int, default=4); a = p.parse_args()
    try:
        run(a.project, a.data, a.study, a.output, a.suite, a.workers)
    except BaseException:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
