"""另建小规模新环境：公开输入、独立帧质量标签、从未用于选标签的评分帧。

每环境先划分train/test再扩展17载频。所有模型共享X和标签文件。
生成过程不读取旧测试结果，不覆盖旧数据；数值检查失败时拒绝启动。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_quality_rank.common import catalog, candidate_metrics, METHODS
from baseline_common.channel import make_environment, serialize_environment
from baseline_common.config import rng_for, qpsk
from compact_dataset import pack_observation, sha256
from generate_dataset import plan_environments
from generate_native_dataset import write_json, now, hashes
from native_sim.config import NativeConfig
from native_sim.control_engine import NativeControlEngine
from hybrid_centered import approximate_cache


def frame_engine(environment, fc, pilots, frame):
    cfg = NativeConfig(); seed = environment['seed']
    payload = qpsk(rng_for(seed, fc, 101, frame), (31,))
    band, dc, details = approximate_cache(environment, fc*1e9, pilots, payload,
        rng_for(seed, fc, 102, frame), cfg, return_details=True)
    engine = NativeControlEngine(cfg, band, dc, fc*1e9, pilots)
    engine.simulation_details = details
    return engine, payload


def one_environment(root_string, row):
    start = time.perf_counter(); root = Path(root_string)
    with np.load(root/'public.npz') as f:
        controls = f['catalog_controls']; pilots = f['pilot_qpsk']
    env = serialize_environment(make_environment(row['seed'], row['factors']))
    xs, singles, robusts, choices, nonlinear_counts = [], [], [], [], []
    for fc in range(4, 21):
        engine, _ = frame_engine(env, fc, pilots, 0)
        counts = [engine.simulation_details['nonlinear_routes']]
        measured = [engine.measure_detailed(u, rng_for(row['seed'], fc, 103, index))
                    for index, u in enumerate(controls)]
        arrays = dict(combined_iq_a=np.stack([v['symbols'] for v in measured[:16]]),
            quality=np.asarray([v['score'] for v in measured[:16]]),
            noise_symbol_var_a2=np.stack([v['noise_symbol_var'] for v in measured[:16]]),
            probe_apd_dc_a=np.asarray([v['apd_dc_a'] for v in measured[:16]]))
        xs.append(pack_observation(arrays, fc))
        choices.append(int(np.argmax([v['score'] for v in measured])))
        labels = []
        for frame in range(1, 5):
            engine, payload = frame_engine(env, fc, pilots, frame)
            counts.append(engine.simulation_details['nonlinear_routes'])
            labels.append(candidate_metrics(engine, controls, payload,
                row['seed'], fc, 104, frame, draws=8)['nmse'])
        singles.append(labels[0]); robusts.append(np.mean(labels, axis=0))
        nonlinear_counts.append(counts)
    folder = root/row['path']; folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'environment.json', env)
    np.savez_compressed(folder/'data.npz', X=np.asarray(xs, np.float32),
        single_nmse=np.asarray(singles, np.float32), robust_nmse=np.asarray(robusts, np.float32),
        measured64_choice=np.asarray(choices, np.int16),
        nonlinear_route_counts=np.asarray(nonlinear_counts, np.int16))
    result = dict(environment_id=row['environment_id'], path=row['path'], samples=17,
        file_sha256={name: sha256(folder/name) for name in ['environment.json', 'data.npz']},
        seconds=time.perf_counter()-start)
    write_json(folder/'complete.json', result)
    return result


def require_checks(project):
    numerical = project/'dataset_simulation/diagnostics/20260925_centered_array'
    approximation = project/'dataset_simulation/diagnostics/20260925_hybrid_centered_full'
    n = json.loads((numerical/'summary.json').read_text())
    a = json.loads((approximation/'summary.json').read_text())
    # 具体数值字段按数值检查器的输出读取，不能把运行完成当作门限通过。
    if a['input_cases'] != 102 or not a['passed_declared_gates']:
        raise ValueError('完整加速核检查尚未通过。')
    if (n['status'] != 'complete' or not n['passed_declared_gates']
            or n['input_cases'] != 102 or n['failures'] != 0):
        raise ValueError('完整非线性参考检查未通过。')
    ap = json.loads((approximation/'protocol.json').read_text())
    if (ap['helper_sha256'] != sha256(SOURCE/'diagnostics/hybrid_centered.py')
            or ap['linear_sha256'] != sha256(SOURCE/'diagnostics/linear_centered.py')
            or ap['staged_sha256'] != sha256(SOURCE/'diagnostics/staged_clock.py')):
        raise ValueError('加速模型与通过检查的源码不同。')
    return dict(numerical_summary=n, approximation_summary=a,
        numerical_summary_sha256=sha256(numerical/'summary.json'),
        approximation_summary_sha256=sha256(approximation/'summary.json'))


def run(project, output, workers):
    if platform.node() != 'qyb-HuaShuo':
        raise RuntimeError('只能在华硕生成实验数据。')
    checks = require_checks(project)
    rows, designs = plan_environments(864, 216, 2026092503, 'joint_stratified')
    old_seeds = set()
    for split in ['train', 'test']:
        old = project/f'dataset_simulation/dataset_{split}/environments.json'
        old_seeds.update(int(r['environment_id'].split('-')[-1]) for r in json.loads(old.read_text()))
    if old_seeds.intersection(int(r['seed']) for r in rows):
        raise ValueError('新环境与原数据种子重合。')
    output.mkdir(parents=True, exist_ok=False)
    cfg = NativeConfig(); controls, angles = catalog(cfg)
    np.savez(output/'public.npz', catalog_controls=controls, catalog_angles_deg=angles,
        probe_controls=controls[:16], pilot_qpsk=qpsk(rng_for(2026092503, 2), (31, 2)))
    files = list(hashes()) + [str(p.relative_to(SOURCE)) for p in
        (SOURCE/'our_method_quality_rank').glob('*') if p.suffix in ('.py', '.md')]
    files += ['diagnostics/linear_centered.py', 'diagnostics/carrier_reference.py',
              'diagnostics/hybrid_centered.py', 'diagnostics/staged_clock.py', 'diagnostics/laser_clock.py',
              'compact_dataset.py', 'baseline_dnn/method.py',
              'deep_common/layers.py', 'deep_common/preprocessing.py']
    source_hashes = {f: sha256(SOURCE/f) for f in sorted(set(files))}
    for f in source_hashes:
        destination = output/'source_snapshot'/f
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/f, destination)
    manifest = dict(status='running', schema='quality-rank-diagnostic-v1',
        created_at_utc=now(), master_seed=2026092503, environments=rows,
        sampling_design=designs, train_samples=14688, test_samples=3672,
        split_unit='propagation environment before carrier expansion', validation=False,
        old_environment_seed_overlap=0, core_source_sha256=hashes(),
        source_sha256=source_hashes, public_sha256=sha256(output/'public.npz'),
        numerical_checks=checks, models=METHODS, python=platform.python_version(),
        numpy=np.__version__, host=platform.node(), initial_probes=16,
        compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
        candidate_count=64, propagation='desired source only, no added interferer or device drift',
        receiver='uniform-stage centered hybrid: small-signal with transient, nonlinear RK4 fallback for large predicted state excursion',
        optical_topology='64 optical routes combined before one final APD',
        frame_roles={'0': 'public observations', '1..4': 'offline supervised quality labels',
                     '5': 'held receiver evaluation'},
        label_apd_draws=8, test_apd_draws=8, epochs=40, training_seed=0,
        note='New same-distribution diagnostic cohort; original formal dataset preserved')
    write_json(output/'manifest.json', manifest)
    started = time.perf_counter(); completed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(one_environment, str(output), row): row for row in rows}
        for job in as_completed(jobs):
            completed.append(job.result())
            progress = dict(status='running', completed_environments=len(completed),
                total_environments=len(rows), seconds=time.perf_counter()-started)
            write_json(output/'progress.json', progress)
            if len(completed) % 20 == 0:
                print(json.dumps(progress), flush=True)
    if hashes() != manifest['core_source_sha256']:
        raise ValueError('冻结核心发生变化。')
    write_json(output/'records.json', sorted(completed, key=lambda r: r['environment_id']))
    manifest.update(status='complete', finished_at_utc=now(),
                    generation_seconds=time.perf_counter()-started,
                    records_sha256=sha256(output/'records.json'))
    write_json(output/'manifest.json', manifest)
    write_json(output/'progress.json', dict(status='complete', completed_environments=len(rows),
        total_environments=len(rows), seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    try:
        run(a.project, a.output, a.workers)
    except BaseException:
        if a.output.exists():
            write_json(a.output/'failure.json', dict(traceback=traceback.format_exc(), at=now()))
        raise
