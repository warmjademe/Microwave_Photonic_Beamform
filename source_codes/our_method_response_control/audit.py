"""评分完成后从公开输入独立重放全部控制；不读取传播真值或评分标签。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import platform
from pathlib import Path
import sys
import numpy as np
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_quality_rank.train import check_data
from our_method_response_control.train import verify_targets
from our_method_response_control.evaluate import choose_controls, check_run_source
from our_method_response_control.physics import LEVELS
from compact_dataset import sha256
from generate_native_dataset import write_json, now


def replay(data_string, targets_string, run_string, row):
    data, targets, run = map(Path, [data_string, targets_string, run_string])
    with np.load(data/row['path']/'data.npz') as f:
        x = f['X']
    with np.load(data/'public.npz') as f:
        pilots, probes = f['pilot_qpsk'], f['probe_controls']
    covariance = np.load(targets/'training_covariance.npy')
    predictions = np.load(run/'predicted_response.npy', mmap_mode='r')
    path = run/'evaluation'/'records'/f'environment_{row["index"]:05d}.npz'
    with np.load(path) as f:
        expected = f['control_code']
        if str(f['environment_id']) != row['environment_id']:
            raise ValueError('评分环境身份不同。')
    for ci in range(17):
        controls, _, _, _ = choose_controls(x[ci], pilots, probes, covariance,
            predictions[row['index']*17+ci])
        actual = np.rint(controls*LEVELS).astype(int)
        np.testing.assert_array_equal(actual, expected[ci])
    return dict(environment_id=row['environment_id'], replayed_codes=int(expected.size),
        record_sha256=sha256(path))


def run(data, targets, root, workers):
    if platform.node() != 'qyb-HuaShuo':
        raise RuntimeError('仅华硕执行。')
    manifest = check_data(data); verify_targets(data, targets); check_run_source(root)
    if json.loads((root/'evaluation/summary.json').read_text())['status'] != 'complete':
        raise ValueError('评分未完成。')
    inference = json.loads((root/'inference.json').read_text())
    if sha256(root/'predicted_response.npy') != inference['predicted_response_sha256']:
        raise ValueError('CNN预测文件改变。')
    records = json.loads((root/'evaluation/records.json').read_text())
    for r in records:
        if sha256(root/'evaluation/records'/r['path']) != r['sha256']:
            raise ValueError('评分文件改变。')
    rows = [r for r in manifest['environments'] if r['split'] == 'test']
    if len(rows) != 216 or [r['index'] for r in rows] != list(range(216)):
        raise ValueError('测试成员索引与预测数组不一致。')
    evidence = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(replay, str(data), str(targets), str(root), row) for row in rows]
        for job in as_completed(jobs):
            evidence.append(job.result())
            write_json(root/'evaluation/control_replay_progress.json',
                dict(status='running', completed=len(evidence), total=216))
    check_data(data); verify_targets(data, targets); check_run_source(root)
    result = dict(status='complete', at=now(), environments=len(evidence), carriers=17,
        methods=4, replayed_codes=sum(r['replayed_codes'] for r in evidence),
        exact_match=True, source_sha256=sha256(__file__),
        inputs='public X, pilots/probes, training covariance, frozen CNN predictions; no environment truth or payload',
        evidence=sorted(evidence, key=lambda r: r['environment_id']))
    write_json(root/'evaluation/control_replay_audit.json', result)
    write_json(root/'evaluation/control_replay_progress.json', dict(status='complete', completed=216, total=216))
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'targets', 'run-root']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); run(a.data, a.targets, a.run_root, a.workers)
