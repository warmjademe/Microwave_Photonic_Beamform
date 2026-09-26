"""冻结模型后统一评分；网络不读取测试标签，额外信息参考单列。

评测保留每环境×载频×方法原始误码、NMSE和实际下发控制码。
置信区间按环境重采样，不将同一环境的17个频率当作独立样本。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
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
from our_method_quality_rank.common import METHODS, LEVELS, make_model, controls_from_output, candidate_metrics
from our_method_quality_rank.train import load_split, check_data
from our_method_quality_rank.generate import frame_engine
from deep_common.preprocessing import apply
from compact_dataset import sha256
from generate_native_dataset import write_json, now

ALL_METHODS = METHODS+['public16', 'feedback64', 'offline_catalog_reference']
CONTRASTS = [('absolute_robust', 'absolute_single'),
             ('candidate_soft_robust', 'candidate_soft_single'),
             ('candidate_soft_robust', 'candidate_ce_robust'),
             ('candidate_ce_robust', 'absolute_robust'),
             ('candidate_soft_robust', 'public16')]


def score_environment(data_string, output_string, row):
    data, output = Path(data_string), Path(output_string)
    env = json.loads((data/row['path']/'environment.json').read_text())
    with np.load(data/'public.npz') as f:
        catalog = f['catalog_controls']; pilots = f['pilot_qpsk']
    with np.load(data/row['path']/'data.npz') as f:
        # 这些字段只为两个单列参考读取，不传给神经控制器。
        feedback = f['measured64_choice']; oracle = f['robust_nmse'].argmin(-1)
        best = f['X'][:, 1985:2001].argmax(-1)
    predicted = {m: np.load(output.parent/m/'predictions.npy', mmap_mode='r') for m in METHODS}
    metrics, codes = [], []
    for ci, fc in enumerate(range(4, 21)):
        sample = row['index']*17+ci
        controls = np.stack([predicted[m][sample] for m in METHODS]
            +[catalog[best[ci]], catalog[feedback[ci]], catalog[oracle[ci]]])
        engine, payload = frame_engine(env, fc, pilots, 5)
        values = candidate_metrics(engine, controls, payload, row['seed'], fc, 105, 5, draws=8)
        metrics.append(np.stack([values['bit_errors'],
            np.full(len(ALL_METHODS), values['bits_tested']), values['nmse']], axis=-1))
        codes.append(np.stack([engine.codes(u) for u in controls]))
    folder = output/'records'; path = folder/f'environment_{row["index"]:05d}.npz'
    np.savez_compressed(path, metrics=np.asarray(metrics), control_code=np.asarray(codes),
        environment_id=np.asarray(row['environment_id']))
    return dict(index=row['index'], environment_id=row['environment_id'],
                path=path.name, sha256=sha256(path))


def infer(data, run_root):
    x, _, _, rows = load_split(data, 'test', labels=False)
    with np.load(run_root/'normalization.npz') as f:
        stats = {k: f[k] for k in ['mean', 'scale']}
    z = torch.from_numpy(apply(x, stats)).cuda()
    with np.load(data/'public.npz') as f:
        controls = torch.as_tensor(f['catalog_controls'], device='cuda', dtype=torch.float32)
    timings = {}
    for method in METHODS:
        folder = run_root/method
        meta = json.loads((folder/'complete.json').read_text())
        if sha256(folder/'weights.pt') != meta['weights_sha256']:
            raise ValueError('权重文件改变。')
        model = make_model(method).cuda()
        model.load_state_dict(torch.load(folder/'weights.pt', map_location='cuda', weights_only=True))
        model.eval(); outputs = []
        with torch.inference_mode():
            for start in range(0, len(z), 512):
                outputs.append(controls_from_output(method, model(z[start:start+512]), controls).cpu().numpy())
            for _ in range(10):
                controls_from_output(method, model(z[:1]), controls)
            torch.cuda.synchronize(); elapsed = []
            for i in range(256):
                start = time.perf_counter()
                controls_from_output(method, model(z[i:i+1]), controls)
                torch.cuda.synchronize(); elapsed.append(time.perf_counter()-start)
        prediction = np.concatenate(outputs)
        np.save(folder/'predictions.npy', prediction, allow_pickle=False)
        timings[method] = dict(mean_single_inference_seconds=float(np.mean(elapsed)),
            median_single_inference_seconds=float(np.median(elapsed)),
            scope='GPU inference plus legal control decoding, standardized input already on GPU',
            predictions_sha256=sha256(folder/'predictions.npy'))
        write_json(folder/'inference.json', timings[method])
    return rows, timings


def summarize(data, run_root, output, rows, records, timings):
    order = sorted(records, key=lambda r: r['index'])
    metrics, codes = [], []
    for record in order:
        path = output/'records'/record['path']
        if sha256(path) != record['sha256']:
            raise ValueError('原始评分文件改变。')
        with np.load(path) as f:
            metrics.append(f['metrics']); codes.append(f['control_code'])
    values = np.asarray(metrics); codes = np.asarray(codes)
    if values.shape != (216, 17, 8, 3) or not np.all(np.isfinite(values)):
        raise ValueError('评分数量或数值不符合协议。')
    audits = {}
    x, _, _, _ = load_split(data, 'test', labels=False)
    with np.load(run_root/'normalization.npz') as f:
        z = torch.from_numpy(apply(x, {k: f[k] for k in ['mean', 'scale']})).cuda()
    with np.load(data/'public.npz') as f:
        catalog = torch.as_tensor(f['catalog_controls'], device='cuda', dtype=torch.float32)
    for index, method in enumerate(METHODS):
        folder = run_root/method; stored = np.load(folder/'predictions.npy')
        expected = np.rint(stored*LEVELS).astype(np.int16)
        if not np.array_equal(codes[:, :, index].reshape(-1, 128), expected):
            raise ValueError('下发码与预测不同：'+method)
        model = make_model(method).cuda()
        model.load_state_dict(torch.load(folder/'weights.pt', map_location='cuda', weights_only=True))
        model.eval(); maximum = 0.
        with torch.inference_mode():
            for start in range(0, len(z), 512):
                replay = controls_from_output(method, model(z[start:start+512]), catalog).cpu().numpy()
                maximum = max(maximum, float(np.max(abs(replay-stored[start:start+512]))))
        if maximum != 0:
            raise ValueError('权重回放不同。')
        audits[method] = dict(replay_max_error=maximum, matched_codes=int(expected.size))
    summaries = []
    for mi, method in enumerate(ALL_METHODS):
        groups = [('all', None, np.ones(len(rows), bool), None)]
        groups += [('carrier_ghz', fc, np.ones(len(rows), bool), fc-4) for fc in range(4, 21)]
        for factor in ['rays', 'max_delay_ns', 'angular_std_deg', 'power_bin']:
            for value in sorted({r['factors'][factor] for r in rows}):
                groups.append((factor, value, np.array([r['factors'][factor] == value for r in rows]), None))
        for group, value, mask, ci in groups:
            a = values[mask, :, mi]
            if ci is not None:
                a = a[:, ci:ci+1]
            nmse = a[..., 2]
            summaries.append(dict(method=method, group=group, value=value,
                environments=int(mask.sum()), ber=float(a[..., 0].sum()/a[..., 1].sum()),
                mean_nmse=float(nmse.mean()), rms_evm_percent=float(100*np.sqrt(nmse.mean())),
                effective_snr_db=float(-10*np.log10(nmse.mean())),
                initial_probes=16, additional_probes=48 if method == 'feedback64' else 0,
                privileged=(method == 'offline_catalog_reference')))
    rng = np.random.default_rng(2026092503)
    indices = rng.integers(0, 216, (1000, 216)); comparisons = []
    for first, second in CONTRASTS:
        a, b = ALL_METHODS.index(first), ALL_METHODS.index(second)
        delta = ((values[:, :, a, 0]-values[:, :, b, 0])/values[:, :, a, 1]).mean(1)
        comparisons.append(dict(first=first, second=second, mean_ber_difference=float(delta.mean()),
            ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist()))
    check_data(data)
    summary = dict(status='complete', records=summaries, contrasts=comparisons,
        inference=timings, uncertainty='1000 paired environment bootstraps; one training seed only',
        scope='new same-distribution environments for preregistered diagnostic protocol',
        evaluation_frame=5, evaluation_apd_draws=8,
        snr_definition='-10log10(mean payload NMSE), includes all distortion, not physical thermal SNR',
        finished_at_utc=now())
    write_json(output/'summary.json', summary)
    write_json(output/'audit.json', dict(status='complete', methods=audits, core_unchanged=True,
        source_sha256=sha256(__file__), data_manifest_sha256=sha256(data/'manifest.json')))
    print(json.dumps([r for r in summaries if r['group'] == 'all']), flush=True)
    print(json.dumps(comparisons), flush=True)


def run(data, run_root, workers):
    if platform.node() != 'qyb-HuaShuo' or not torch.cuda.is_available():
        raise RuntimeError('需华硕GPU推理。')
    check_data(data)
    if json.loads((run_root/'progress.json').read_text())['status'] != 'complete':
        raise ValueError('训练未完成。')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = run_root/'evaluation'; output.mkdir(exist_ok=False); (output/'records').mkdir()
    write_json(output/'protocol.json', dict(methods=ALL_METHODS, contrasts=CONTRASTS,
        primary='held-frame payload BER', created_at_utc=now(), source_sha256=sha256(__file__),
        data_manifest_sha256=sha256(data/'manifest.json')))
    rows, timings = infer(data, run_root); results = []
    # GPU推理完成后，CPU进程只进行物理评分，不创建CUDA对象。
    import multiprocessing
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(score_environment, str(data), str(output), row) for row in rows]
        for job in as_completed(jobs):
            results.append(job.result())
            write_json(output/'progress.json', dict(status='running', completed=len(results), total=len(rows)))
    write_json(output/'records.json', sorted(results, key=lambda r: r['index']))
    summarize(data, run_root, output, rows, results, timings)
    write_json(output/'progress.json', dict(status='complete', completed=len(rows), total=len(rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6); a = p.parse_args()
    try:
        run(a.data, a.run_root, a.workers)
    except BaseException:
        write_json(a.run_root/'evaluation_failure.json', dict(traceback=traceback.format_exc(), at=now()))
        raise
