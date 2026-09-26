"""同一独立帧评价完整128维控制；先决定控制，再读取传播真值评分。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
import platform
from pathlib import Path
import sys
import time
import traceback
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_quality_rank.train import load_split, check_data
from our_method_quality_rank.generate import frame_engine
from our_method_quality_rank.common import candidate_metrics
from our_method_response_control.physics import ridge_estimate, covariance_estimate, decode, training_target, LEVELS
from our_method_response_control.model import Model, conditions
from our_method_response_control.train import verify_targets, precision
from compact_dataset import sha256
from generate_native_dataset import write_json, now

METHODS = ['ridge_response', 'covariance_response', 'complex_response_cnn', 'public16']
METRICS = ['bit_errors', 'bits_tested', 'payload_nmse', 'proxy_initial', 'proxy_final',
           'controller_cpu_seconds', 'relative_response_mse']


def check_run_source(run_root):
    protocol = json.loads((run_root/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if sha256(SOURCE/name) != digest:
            raise ValueError('训练开始后方法源码改变：'+name)


def choose_controls(x, pilots, probes, covariance, cnn_response):
    """在线信息边界：接口中没有环境真值、payload、标签、环境seed或测试索引。"""
    best = int(x[1985:2001].argmax()); initial = probes[best]
    responses, controls, timings, proxy = [], [], [], []
    for method in METHODS[:3]:
        start = time.perf_counter()
        if method == 'ridge_response':
            h = ridge_estimate(x, pilots)[0]
        elif method == 'covariance_response':
            h = covariance_estimate(x, pilots, covariance)
        else:
            h = cnn_response
        control, info = decode(h, int(x[1984]), initial, sweeps=2)
        timings.append(time.perf_counter()-start); responses.append(h); controls.append(control)
        proxy.append([info['initial_proxy'], info['final_proxy']])
    controls.append(initial.copy()); timings.append(0.); proxy.append([np.nan, np.nan])
    return np.stack(controls), responses, np.asarray(timings), np.asarray(proxy)


def one_environment(data_string, targets_string, output_string, row):
    data, targets, output = Path(data_string), Path(targets_string), Path(output_string)
    with np.load(data/row['path']/'data.npz') as f:
        x = f['X']  # 不读取测试监督标签或64反馈选优结果。
    with np.load(data/'public.npz') as f:
        pilots, probes = f['pilot_qpsk'], f['probe_controls']
    covariance = np.load(targets/'training_covariance.npy')
    predictions = np.load(output.parent/'predicted_response.npy', mmap_mode='r')
    decisions = [choose_controls(x[ci], pilots, probes, covariance,
        predictions[row['index']*17+ci]) for ci in range(17)]
    # 全17载频控制均已确定，之后才打开仿真传播真值和独立payload评分。
    environment = json.loads((data/row['path']/'environment.json').read_text())
    metrics, codes = [], []
    for ci, fc in enumerate(range(4, 21)):
        controls, responses, seconds, proxy = decisions[ci]
        engine, payload = frame_engine(environment, fc, pilots, 5)
        result = candidate_metrics(engine, controls, payload, row['seed'], fc, 105, 5, draws=8)
        htrue = training_target(environment, fc)  # 仅在控制冻结后计算诊断误差。
        error = [np.mean(abs(h-htrue)**2)/np.mean(abs(htrue)**2) for h in responses]+[np.nan]
        metrics.append(np.column_stack([result['bit_errors'], np.full(4, result['bits_tested']),
            result['nmse'], proxy, seconds, error]))
        codes.append(np.stack([engine.codes(u) for u in controls]))
    path = output/'records'/f'environment_{row["index"]:05d}.npz'
    np.savez_compressed(path, metrics=np.asarray(metrics), control_code=np.asarray(codes),
        environment_id=np.asarray(row['environment_id']))
    return dict(index=row['index'], path=path.name, sha256=sha256(path), environment_id=row['environment_id'])


def infer(data, run_root):
    x, _, _, rows = load_split(data, 'test', labels=False)
    with np.load(data/'public.npz') as f:
        pilots = f['pilot_qpsk']
    started = time.perf_counter()
    initial = ridge_estimate(x, pilots).astype(np.complex64)
    condition_array = conditions(x, initial, pilots)
    preprocessing_batch_seconds = time.perf_counter()-started
    # 在线逐样本预处理另测；不把批处理均摊时间当作单次控制延迟。
    preprocessing_single = []
    for i in range(256):
        tick = time.perf_counter()
        lifted = ridge_estimate(x[i:i+1], pilots).astype(np.complex64)
        conditions(x[i:i+1], lifted, pilots)
        preprocessing_single.append(time.perf_counter()-tick)
    cond = torch.from_numpy(condition_array).cuda()
    initial = torch.from_numpy(initial).cuda()
    complete = json.loads((run_root/'complete.json').read_text())
    if sha256(run_root/'weights.pt') != complete['weights_sha256']:
        raise ValueError('模型权重改变。')
    model = Model().cuda()
    model.load_state_dict(torch.load(run_root/'weights.pt', map_location='cuda', weights_only=True)); model.eval()
    responses = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            responses.append(model(initial[start:start+128], cond[start:start+128]).cpu().numpy())
        for _ in range(10):
            model(initial[:1], cond[:1])
        torch.cuda.synchronize(); seconds = []
        for i in range(256):
            tick = time.perf_counter(); model(initial[i:i+1], cond[i:i+1]); torch.cuda.synchronize()
            seconds.append(time.perf_counter()-tick)
    predictions = np.concatenate(responses)
    np.save(run_root/'predicted_response.npy', predictions, allow_pickle=False)
    write_json(run_root/'inference.json', dict(mean_gpu_single_seconds=float(np.mean(seconds)),
        median_gpu_single_seconds=float(np.median(seconds)), predicted_response_sha256=sha256(run_root/'predicted_response.npy'),
        batch_cpu_preprocessing_seconds=preprocessing_batch_seconds,
        mean_single_cpu_preprocessing_seconds=float(np.mean(preprocessing_single)),
        median_single_cpu_preprocessing_seconds=float(np.median(preprocessing_single)),
        scope='GPU forward and CPU lift/conditioning timed separately; CPU control decoder in raw evaluation; excludes hardware measurement and host-device transfer'))
    # 重新加载权重，重算全量预测；不复用第一次的模型对象。
    replay = Model().cuda()
    replay.load_state_dict(torch.load(run_root/'weights.pt', map_location='cuda', weights_only=True)); replay.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            values = replay(initial[start:start+128], cond[start:start+128]).cpu().numpy()
            if not np.array_equal(values, predictions[start:start+128]):
                raise ValueError('完整CNN响应回放不一致。')
    return rows


def summarize(data, targets, output, rows, records):
    values, codes = [], []
    for record in sorted(records, key=lambda r: r['index']):
        path = output/'records'/record['path']
        if sha256(path) != record['sha256']:
            raise ValueError('评分文件SHA改变。')
        with np.load(path) as f:
            values.append(f['metrics']); codes.append(f['control_code'])
    values, codes = np.asarray(values), np.asarray(codes)
    if values.shape != (216, 17, 4, 7) or not np.all(np.isfinite(values[..., :3])):
        raise ValueError('独立评分不完整。')
    if np.any(codes < 0) or np.any(codes > LEVELS):
        raise ValueError('控制码越界。')
    if np.any(values[:, :, :3, 4]+1e-12 < values[:, :, :3, 3]):
        raise ValueError('控制代理目标下降。')
    summaries = []
    for mi, method in enumerate(METHODS):
        groups = [('all', None, np.ones(216, bool), None)]
        groups += [('carrier_ghz', fc, np.ones(216, bool), fc-4) for fc in range(4, 21)]
        for factor in ['rays', 'max_delay_ns', 'angular_std_deg', 'power_bin']:
            groups += [(factor, value, np.array([r['factors'][factor] == value for r in rows]), None)
                for value in sorted({r['factors'][factor] for r in rows})]
        for group, value, mask, ci in groups:
            selected = values[mask, :, mi]
            if ci is not None:
                selected = selected[:, ci:ci+1]
            nmse = float(selected[..., 2].mean())
            summaries.append(dict(method=method, group=group, value=value,
                environments=int(mask.sum()), ber=float(selected[..., 0].sum()/selected[..., 1].sum()),
                mean_nmse=nmse, rms_evm_percent=100*np.sqrt(nmse), effective_snr_db=-10*np.log10(nmse),
                mean_cpu_control_seconds=float(selected[..., 5].mean()),
                relative_response_mse=float(selected[..., 6].mean()) if mi < 3 else None,
                probes=16, extra_probes=0))
    rng = np.random.default_rng(2026092504); indices = rng.integers(0, 216, (1000, 216))
    contrasts = []
    for second in ['ridge_response', 'covariance_response', 'public16']:
        j = METHODS.index(second)
        delta = ((values[:, :, 2, 0]-values[:, :, j, 0])/values[:, :, 2, 1]).mean(1)
        contrasts.append(dict(first='complex_response_cnn', second=second,
            mean_ber_difference=float(delta.mean()), ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist()))
    check_data(data); verify_targets(data, targets); check_run_source(output.parent)
    write_json(output/'summary.json', dict(status='complete', records=summaries, contrasts=contrasts,
        scope='preregistered same-distribution new test; linear response proxy for decisions, full hybrid receiver for quality',
        uncertainty='216 independent environments; single training seed; 1000 paired bootstraps',
        finished_at_utc=now()))
    write_json(output/'audit.json', dict(status='complete', complete_weight_response_replay=True,
        legal_control_codes=int(codes.size), all_proxy_updates_nondecreasing=True,
        test_truth_access='all17carrier controls fixed before environment truth read in each evaluation worker',
        limitation='independent full decoder replay required in control_replay_audit.json before final completion',
        core_unchanged=True))
    print(json.dumps([r for r in summaries if r['group'] == 'all']), flush=True)
    print(json.dumps(contrasts), flush=True)


def run(data, targets, run_root, workers):
    if platform.node() != 'qyb-HuaShuo' or not torch.cuda.is_available():
        raise RuntimeError('仅华硕GPU推理。')
    check_data(data); verify_targets(data, targets); check_run_source(run_root); precision()
    if json.loads((run_root/'complete.json').read_text())['status'] != 'complete':
        raise ValueError('模型尚未完成训练。')
    output = run_root/'evaluation'; output.mkdir(exist_ok=False); (output/'records').mkdir()
    write_json(output/'protocol.json', dict(methods=METHODS, metrics=METRICS, created_at_utc=now(),
        source_sha256=sha256(__file__), frame=5, apd_draws=8, extra_feedback=0))
    rows = infer(data, run_root); records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(one_environment, str(data), str(targets), str(output), row) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result())
            write_json(output/'progress.json', dict(status='running', completed=len(records), total=216))
    write_json(output/'records.json', records)
    summarize(data, targets, output, rows, records)
    write_json(output/'progress.json', dict(status='complete', completed=216, total=216))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'targets', 'run-root']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=6); a = p.parse_args()
    try:
        run(a.data, a.targets, a.run_root, a.workers)
    except BaseException:
        write_json(a.run_root/'evaluation_failure.json', dict(traceback=traceback.format_exc(), at=now()))
        raise
