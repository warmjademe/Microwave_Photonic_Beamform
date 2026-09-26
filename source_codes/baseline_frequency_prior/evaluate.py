"""只从训练质量选择固定控制，在同一独立帧检验是否足以解释网络收益。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from baseline_frequency_prior.method import fit, predict_indices
from our_method_quality_rank.train import check_data, load_split
from our_method_quality_rank.generate import frame_engine
from our_method_quality_rank.common import candidate_metrics
from our_method_quality_rank.evaluate import ALL_METHODS
from our_method_response_control.evaluate import METHODS as RESPONSE_METHODS
from compact_dataset import sha256
from generate_native_dataset import write_json, now

METHODS = ['global_prior', 'frequency_prior']


def one_environment(data_string, output_string, row):
    data, output = Path(data_string), Path(output_string)
    with np.load(output/'model.npz') as f:
        model = {k: f[k] for k in f.files}
    # 控制完全由训练选出的表和载频决定，确定后再打开环境真值。
    choices = predict_indices(model, np.arange(4, 21))
    with np.load(data/'public.npz') as f:
        pilots, catalog = f['pilot_qpsk'], f['catalog_controls']
    controls = catalog[choices]
    environment = json.loads((data/row['path']/'environment.json').read_text())
    metrics, codes = [], []
    for ci, fc in enumerate(range(4, 21)):
        engine, payload = frame_engine(environment, fc, pilots, 5)
        result = candidate_metrics(engine, controls[ci], payload, row['seed'], fc, 105, 5, draws=8)
        metrics.append(np.column_stack([result['bit_errors'], np.full(2, 496), result['nmse']]))
        codes.append(np.stack([engine.codes(u) for u in controls[ci]]))
    path = output/'records'/f'environment_{row["index"]:05d}.npz'
    np.savez_compressed(path, metrics=np.asarray(metrics), control_code=np.asarray(codes),
        chosen_candidates=choices, environment_id=np.asarray(row['environment_id']))
    return dict(index=row['index'], path=path.name, environment_id=row['environment_id'], sha256=sha256(path))


def comparison_records(root, method, order):
    records = sorted(json.loads((root/'evaluation/records.json').read_text()), key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(216)):
        raise ValueError('网络评分索引不完整。')
    values = []
    for record in records:
        path = root/'evaluation/records'/record['path']
        if sha256(path) != record['sha256']:
            raise ValueError('网络评分文件改变。')
        with np.load(path) as f:
            values.append(f['metrics'][:, order.index(method), :3])
    return np.asarray(values)


def run(project, output, workers):
    if platform.node() != 'qyb-HuaShuo':
        raise RuntimeError('仅华硕执行。')
    ds = project/'dataset_simulation'; data = ds/'outputs/quality_rank_hybrid_20260925'
    rank = ds/'baseline_results/20260925_quality_rank_hybrid'
    response = ds/'baseline_results/20260925_response_control'
    for path in [rank/'evaluation/audit.json', response/'evaluation/control_replay_audit.json']:
        if json.loads(path.read_text())['status'] != 'complete':
            raise ValueError('先完成两条网络审计，避免并行计时争用。')
    manifest = check_data(data)
    _, _, quality, training_rows = load_split(data, 'train', labels=True)
    if len(training_rows) != 864:
        raise ValueError('训练环境数量不符。')
    started = time.perf_counter(); model = fit(quality.reshape(864, 17, 64))
    fit_seconds = time.perf_counter()-started
    output.mkdir(exist_ok=False, parents=True); (output/'records').mkdir()
    np.savez(output/'model.npz', **model)
    frozen = {str(p.relative_to(SOURCE)): sha256(p)
        for p in (SOURCE/'baseline_frequency_prior').glob('*') if p.suffix in ['.py', '.md']}
    write_json(output/'protocol.json', dict(at=now(), source_sha256=frozen, methods=METHODS,
        fit_uses_train_only=True, fit_seconds=fit_seconds, frame=5, apd_draws=8,
        available_initial_probes=16, used_initial_probes=0, extra_probes=0,
        data_manifest_sha256=sha256(data/'manifest.json'), model_sha256=sha256(output/'model.npz'),
        bootstrap_seed=2026092510, bootstrap_draws=1000))
    rows = [r for r in manifest['environments'] if r['split'] == 'test']
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(one_environment, str(data), str(output), row) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result())
            write_json(output/'progress.json', dict(status='running', completed=len(records), total=len(rows)))
    write_json(output/'records.json', records)
    values = []
    with np.load(data/'public.npz') as f:
        catalog_codes = np.rint(f['catalog_controls']*np.r_[np.full(64, 76), np.full(64, 24)]).astype(int)
    expected_codes = catalog_codes[predict_indices(model, np.arange(4, 21))]
    for record in sorted(records, key=lambda r: r['index']):
        path = output/'records'/record['path']
        if sha256(path) != record['sha256']:
            raise ValueError('先验评分文件改变。')
        with np.load(path) as f:
            np.testing.assert_array_equal(f['control_code'], expected_codes)
            values.append(f['metrics'])
    values = np.asarray(values)
    if values.shape != (216, 17, 2, 3) or not np.all(np.isfinite(values)):
        raise ValueError('先验评分不完整。')
    summaries = []
    for mi, method in enumerate(METHODS):
        groups = [('all', None, np.ones(216, bool), None)]
        groups += [('carrier_ghz', fc, np.ones(216, bool), fc-4) for fc in range(4, 21)]
        for factor in ['rays', 'max_delay_ns', 'angular_std_deg', 'power_bin']:
            groups += [(factor, v, np.array([r['factors'][factor] == v for r in rows]), None)
                for v in sorted({r['factors'][factor] for r in rows})]
        for group, label, mask, ci in groups:
            selected = values[mask, :, mi]
            if ci is not None:
                selected = selected[:, ci:ci+1]
            nmse = float(selected[..., 2].mean())
            summaries.append(dict(method=method, group=group, value=label, environments=int(mask.sum()),
                ber=float(selected[..., 0].sum()/selected[..., 1].sum()), mean_nmse=nmse,
                rms_evm_percent=float(100*np.sqrt(nmse)), effective_quality_db=float(-10*np.log10(nmse)),
                used_probes=0, available_probes=16, extra_probes=0))
    rng = np.random.default_rng(2026092510); samples = rng.integers(0, 216, (1000, 216))
    contrasts = []
    for name, folder, order in [('candidate_soft_robust', rank, ALL_METHODS),
                                ('complex_response_cnn', response, RESPONSE_METHODS)]:
        alternative = comparison_records(folder, name, order)
        np.testing.assert_array_equal(alternative[..., 1], values[:, :, 1, 1])
        delta = ((alternative[..., 0]-values[:, :, 1, 0])/alternative[..., 1]).mean(1)
        contrasts.append(dict(first=name, second='frequency_prior', mean_ber_difference=float(delta.mean()),
            ci95=np.quantile(delta[samples].mean(1), [.025, .975]).tolist()))
    check_data(data)
    assert all(sha256(SOURCE/p) == digest for p, digest in frozen.items())
    write_json(output/'summary.json', dict(status='complete', records=summaries, contrasts=contrasts,
        fit_seconds=fit_seconds, at=now(), scope='preregistered fixed training prior; no online channel information',
        global_candidate=int(model['global_candidate']), carrier_candidates=model['carrier_candidates'].tolist()))
    write_json(output/'audit.json', dict(status='complete', all_controls_equal_train_prior=True,
        matched_codes=216*17*2*128, source_unchanged=True, model_sha256=sha256(output/'model.npz')))
    write_json(output/'progress.json', dict(status='complete', completed=216, total=216))
    print(json.dumps(contrasts), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); run(a.project, a.output, a.workers)
