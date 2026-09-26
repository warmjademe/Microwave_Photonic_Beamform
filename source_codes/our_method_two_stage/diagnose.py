"""固定24个训练环境的阶段B设计诊断；不接触新留出测试环境。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from our_method_response_control.physics import covariance_estimate, ridge_estimate, decode
from our_method_response_control.model import Model, conditions
from our_method_response_control.train import precision, verify_targets
from our_method_two_stage.control import compare_controls, proxy_quality, phase_starts, PROXY_FIELDS


def selected_rows(manifest):
    rows = [r for r in manifest['environments'] if r['split']=='train']
    selected = {}
    for row in sorted(rows, key=lambda r:r['index']):
        key = (row['factors']['power_bin'], row['factors']['rays'])
        selected.setdefault(key, row)
    if len(selected) != 24:
        raise ValueError('预设六档功率×四档路径数未完整覆盖。')
    return list(selected.values())


def preflight(h, public, x):
    evidence = []
    for fc in [4, 12, 20]:
        initial = public['probe_controls'][int(x[1985:2001].argmax())]
        controls, info = compare_controls(h, fc, initial)
        original, old = decode(h, fc, initial, sweeps=2)
        assert np.array_equal(controls[0], original)
        quality = np.asarray(info['proxy'])
        assert np.isclose(quality[0, 1], -old['final_proxy'], rtol=1e-12, atol=1e-14)
        assert quality[1, 1] <= quality[0, 1]+1e-12
        assert quality[2, 1] <= quality[0, 1]+1e-12
        assert quality[3, 0] <= quality[0, 0]+1e-12
        assert quality[3, 1] <= quality[0, 1]+1e-12
        assert quality[3, 2] >= quality[0, 2]*(1-1e-12)
        zero = proxy_quality(np.zeros_like(h), fc, controls)
        assert np.all(zero == np.asarray([.5, 1., 0.]))
        doubled = proxy_quality(2*h, fc, controls)
        assert np.allclose(doubled[:, 2], 4*quality[:, 2], rtol=1e-12)
        assert np.all(doubled[:, :2] <= quality[:, :2]+1e-14)
        starts = phase_starts(h, fc)
        codes = np.rint(starts*LEVELS)
        assert starts.shape == (4, 128)
        assert np.all((codes >= 0) & (codes <= LEVELS))
        assert np.allclose(starts*LEVELS, codes, rtol=0, atol=1e-12)
        evidence.append(dict(carrier_ghz=fc, baseline_identity=True,
            original_objective_match=True, proxy_fallback=True,
            zero_signal_limit=True, signal_power_scaling=True, legal_codes=True))
    return evidence


def run(data, targets, checkpoint, output, preflight_only=False):
    require_host(); precision()
    if not torch.cuda.is_available():
        raise RuntimeError('阶段A的CNN推理只在华硕GPU执行。')
    manifest = check_data(data); verify_targets(data, targets)
    checkpoint_meta = json.loads((checkpoint/'complete.json').read_text())
    if checkpoint_meta['weights_sha256'] != sha256(checkpoint/'weights.pt'):
        raise ValueError('固定CNN权重改变。')
    rows = selected_rows(manifest); public = public_data(data)
    covariance = np.load(targets/'training_covariance.npy')
    raw = []
    for row in rows:
        with np.load(data/row['path']/'data.npz') as f:
            raw.extend([f['X'][fc-4] for fc in [4, 12, 20]])
    x = np.asarray(raw)
    initial = ridge_estimate(x, public['pilot_qpsk']).astype(np.complex64)
    cond = conditions(x, initial, public['pilot_qpsk'])
    model = Model().cuda()
    model.load_state_dict(torch.load(checkpoint/'weights.pt', map_location='cuda', weights_only=True))
    model.eval()
    with torch.inference_mode():
        cnn = model(torch.from_numpy(initial).cuda(), torch.from_numpy(cond).cuda()).cpu().numpy()
    del model; torch.cuda.empty_cache()
    evidence = preflight(cnn[0], public, x[0])
    if preflight_only:
        if output.exists():
            raise FileExistsError('前置检查不覆盖。')
        write_json(output, dict(status='passed', checks=evidence, at=now(),
            train_environment=rows[0]['environment_id']))
        print(json.dumps(evidence), flush=True); return
    sources = ['our_method_two_stage/'+n for n in ['control.py', 'diagnose.py', 'DIAGNOSTIC_PROTOCOL.md']]
    sources += ['study_full_baselines/common.py', 'our_method_response_control/physics.py',
                'our_method_response_control/model.py']
    protocol = dict(training_environment_ids=[r['environment_id'] for r in rows],
        carriers_ghz=[4, 12, 20], frame=5, apd_draws=8, test_used=False,
        methods=[a+'__'+b for a in ['covariance', 'cnn'] for b in
            ['base_2sweeps', 'single_10sweeps', 'multi_mmse', 'multi_gated_ber']],
        metric_order=QUALITY_METRICS, proxy_order=PROXY_FIELDS,
        data_manifest_sha256=sha256(data/'manifest.json'),
        weight_sha256=sha256(checkpoint/'weights.pt'),
        covariance_sha256=sha256(targets/'training_covariance.npy'),
        source_sha256=source_record(sources))
    fp = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True); (output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text()) != protocol:
            raise ValueError('诊断协议改变。')
    else:
        write_json(output/'protocol.json', protocol)
        write_json(output/'preflight.json', dict(status='passed', checks=evidence, at=now()))
        for name in sources:
            dst = output/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE/name, dst)
    started = time.perf_counter(); all_metrics = []; all_seconds = []; all_proxy = []
    for ri, row in enumerate(rows):
        for ci, fc in enumerate([4, 12, 20]):
            i = 3*ri+ci; path = output/'records'/('%03d.npz'%i); marker = path.with_suffix('.json')
            if marker.exists():
                meta = json.loads(marker.read_text())
                if meta['fingerprint'] != fp or meta['sha256'] != sha256(path):
                    raise ValueError('既有诊断结果发生变化。')
                with np.load(path) as f:
                    metrics = f['metrics']; seconds = f['seconds']; proxies = f['proxy']
            else:
                if path.exists():
                    raise ValueError('未提交诊断记录，须先核查。')
                estimate = covariance_estimate(x[i], public['pilot_qpsk'], covariance)
                h = np.stack([estimate, cnn[i]])
                start = public['probe_controls'][int(x[i, 1985:2001].argmax())]
                decisions = [compare_controls(one, fc, start) for one in h]
                controls = np.concatenate([v[0] for v in decisions])
                seconds = np.concatenate([v[1]['seconds'] for v in decisions])
                proxies = np.concatenate([v[1]['proxy'] for v in decisions])
                # 决策已完成后，才打开传播真值；真值只用于评分。
                env = json.loads((data/row['path']/'environment.json').read_text())
                engine, payload = frame_engine(env, fc, public['pilot_qpsk'], 5)
                clean = clean_frame_engine(env, fc, public['pilot_qpsk'], payload)
                metrics, codes = reception_metrics(engine, clean, controls, payload, row['seed'], fc)
                atomic_npz(path, metrics=metrics, control_code=codes, seconds=seconds,
                    response_estimate=h, proxy=proxies, public_x=x[i],
                    candidate_controls=np.asarray([v[1]['candidate_controls'] for v in decisions]),
                    candidate_proxy=np.asarray([v[1]['candidate_proxy'] for v in decisions]),
                    selected_indices=np.asarray([v[1]['selected_indices'] for v in decisions]))
                write_json(marker, dict(fingerprint=fp, sha256=sha256(path),
                    environment_id=row['environment_id'], carrier_ghz=fc, at=now()))
            all_metrics.append(metrics); all_seconds.append(seconds); all_proxy.append(proxies)
            progress = dict(status='running', completed=i+1, total=len(x), pid=os.getpid(),
                seconds=time.perf_counter()-started, at=now())
            write_json(output/'progress.json', progress)
            if (i+1)%6 == 0:
                print(json.dumps(progress), flush=True)
    scores = np.asarray(all_metrics); records = []
    for mi, method in enumerate(protocol['methods']):
        a = scores[:, mi]; nmse = float(a[:, 2].mean())
        records.append(dict(method=method, ber=float(a[:, 0].sum()/a[:, 1].sum()),
            ser=float(a[:, 3].sum()/a[:, 4].sum()), block_error_rate=float(a[:, 5].sum()/a[:, 6].sum()),
            rms_evm_percent=100*np.sqrt(nmse), mean_nmse=nmse,
            paired_output_snr_db=float(10*np.log10(a[:, 7].sum()/a[:, 8].sum())),
            mean_controller_seconds=float(np.asarray(all_seconds)[:, mi].mean())))
    verify_sources(protocol['source_sha256']); verify_targets(data, targets)
    if sha256(checkpoint/'weights.pt') != protocol['weight_sha256']:
        raise ValueError('检查点在诊断中改变。')
    atomic_npz(output/'all_metrics.npz', metrics=scores, seconds=np.asarray(all_seconds),
               proxy=np.asarray(all_proxy))
    write_json(output/'summary.json', dict(status='complete', scope='training-only design diagnostic',
        independent_test=False, records=records, at=now()))
    write_json(output/'complete.json', dict(status='complete', fingerprint=fp,
        results_sha256=sha256(output/'all_metrics.npz'), at=now()))
    write_json(output/'progress.json', dict(status='complete', completed=len(x), total=len(x), at=now()))
    print(json.dumps(records), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'targets', 'checkpoint', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true'); a = p.parse_args()
    try:
        run(a.data, a.targets, a.checkpoint, a.output, a.preflight_only)
    except BaseException:
        if not a.preflight_only:
            a.output.mkdir(parents=True, exist_ok=True)
            write_json(a.output/('failure_%d.json'%time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
