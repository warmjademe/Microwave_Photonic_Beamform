"""仅为统一训练成员生成原生教师控制标签；不打开测试数据。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (SOURCE, LEVELS, require_host, check_data,
    public_data, frame_engine, rng_for, atomic_npz, source_record, verify_sources,
    sha256, write_json, now)
from native_sim.control_engine import optimize_teacher


def one_environment(data_string, output_string, row, fingerprint):
    data, output = Path(data_string), Path(output_string)
    path = output / 'records' / ('environment_%05d.npz' % row['index'])
    marker = path.with_suffix('.json')
    if marker.exists():
        m = json.loads(marker.read_text())
        if m['fingerprint'] != fingerprint or sha256(path) != m['sha256']:
            raise ValueError('既有标签来源或哈希不同。')
        return m
    if path.exists():
        raise ValueError('未提交标签产物必须人工核查：' + str(path))
    if row['split'] != 'train':
        raise ValueError('标签生成器禁止测试环境。')
    public = public_data(data)
    env = json.loads((data / row['path'] / 'environment.json').read_text())
    started = time.perf_counter(); controls, objectives, times, calls = [], [], [], []
    for carrier in range(4, 21):
        engine, _ = frame_engine(env, carrier, public['pilot_qpsk'], 0)
        tick = time.perf_counter()
        u, info = optimize_teacher(engine, public['probe_controls'],
            rng_for(0, row['seed'], carrier, 610), starts=2, sweeps=1)
        times.append(time.perf_counter() - tick)
        controls.append(engine.codes(u))
        objectives.append([info['initial_best_objective'], info['objective']])
        calls.append(info['objective_evaluations'])
    atomic_npz(path, control_code=np.asarray(controls, np.int16),
        objectives=np.asarray(objectives), optimization_seconds=np.asarray(times),
        objective_evaluations=np.asarray(calls), environment_id=np.asarray(row['environment_id']))
    m = dict(index=row['index'], environment_id=row['environment_id'],
        sha256=sha256(path), fingerprint=fingerprint, path=path.name,
        seconds=time.perf_counter() - started, completed_at_utc=now())
    write_json(marker, m)
    return m


def run(data, output, workers):
    require_host(); manifest = check_data(data)
    rows = [r for r in manifest['environments'] if r['split'] == 'train']
    names = ['study_full_baselines/common.py', 'study_full_baselines/prepare_labels.py',
        'study_full_baselines/PROTOCOL.md', 'native_sim/control_engine.py',
        'our_method_quality_rank/generate.py', 'diagnostics/hybrid_centered.py']
    protocol = dict(data_manifest_sha256=sha256(data / 'manifest.json'),
        train_ids=[r['environment_id'] for r in rows], starts=2, sweeps=1,
        frame=0, uses_payload=False, uses_test=False, seed=0,
        source_sha256=source_record(names))
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True); (output / 'records').mkdir(exist_ok=True)
    if (output / 'protocol.json').exists():
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('标签协议改变，拒绝覆盖。')
    else:
        write_json(output / 'protocol.json', protocol)
        for name in names:
            dest = output / 'source_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, dest)
    started = time.perf_counter(); records = []
    write_json(output / 'progress.json', dict(status='running', completed=0,
        total=len(rows), workers=workers, pid=os.getpid(), at=now()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(one_environment, str(data), str(output), row, fingerprint) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result())
            progress = dict(status='running', completed=len(records), total=len(rows),
                workers=workers, pid=os.getpid(), seconds=time.perf_counter()-started, at=now())
            write_json(output / 'progress.json', progress)
            if len(records) % 20 == 0:
                print(json.dumps(progress), flush=True)
    result = []
    for r in sorted(records, key=lambda r: r['index']):
        path = output / 'records' / r['path']
        if sha256(path) != r['sha256']:
            raise ValueError('原始标签文件改变。')
        with np.load(path) as f:
            if np.any(f['objectives'][:, 1] < f['objectives'][:, 0] - 1e-12):
                raise ValueError('教师代理目标下降。')
            result.append(f['control_code'])
    codes = np.concatenate(result)
    if codes.shape != (len(rows)*17, 128) or np.any(codes < 0) or np.any(codes > LEVELS):
        raise ValueError('标签尺寸或档位错误。')
    np.save(output / 'train_Y_code.npy', codes, allow_pickle=False)
    np.save(output / 'train_Y.npy', (codes/LEVELS).astype(np.float32), allow_pickle=False)
    write_json(output / 'records.json', sorted(records, key=lambda r: r['index']))
    verify_sources(protocol['source_sha256']); check_data(data)
    write_json(output / 'complete.json', dict(status='complete', fingerprint=fingerprint,
        samples=len(codes), train_environments=len(rows), seconds=time.perf_counter()-started,
        file_sha256={n: sha256(output/n) for n in ['train_Y.npy', 'train_Y_code.npy', 'records.json']},
        finished_at_utc=now()))
    write_json(output / 'progress.json', dict(status='complete', completed=len(rows), total=len(rows), at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    try:
        run(a.data, a.output, a.workers)
    except BaseException:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(a.output/('failure_%d.json' % time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
