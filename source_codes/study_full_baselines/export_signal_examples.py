"""固定旧测试索引0：逐方法导出真实接收波形，并与原评分的十项指标核对。

数值工作仅在华硕执行。控制直接读取已经提交的评分记录；不重新优化，
不选择较好的噪声抽样。预检查允许指定少数流水线/载频，正式导出要求全部。
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from native_sim.evaluation import evaluate_record
from baseline_common.config import KB, cn
from baseline_common.channel import deserialize_environment, response
from baseline_common.metrics import evaluate_output

PHASES = ['classic', 'learned_evaluation', 'evaluation_components',
          'evaluation_real_response_cnn', 'evaluation_scaling_1728',
          'evaluation_scaling_3456', 'evaluation_feedback_warm']
SELF = 'study_full_baselines/export_signal_examples.py'
SOURCES = [SELF, 'study_full_baselines/common.py', 'native_sim/evaluation.py',
           'native_sim/config.py', 'native_sim/control_engine.py', 'native_sim/waveforms.py',
           'baseline_common/metrics.py', 'baseline_common/channel.py',
           'baseline_common/config.py', 'our_method_quality_rank/generate.py',
           'diagnostics/hybrid_centered.py', 'diagnostics/linear_centered.py',
           'diagnostics/staged_clock.py']


def complex_list(a):
    """JSON复数显式写作[实部,虚部]，不把相位或幅值冒充I/Q。"""
    a = np.asarray(a)
    return np.stack([a.real, a.imag], axis=-1).tolist()


def checked_equal(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True):
        raise ValueError('独立I/Q回放没有复现原始评分：' + label)


def quality(v):
    """从充分统计量重新计算百分比和dB；MRC电流功率字段为空。"""
    v = np.asarray(v)
    result = dict(ber=float(v[0]/v[1]), ser=float(v[3]/v[4]),
                  block_error_rate=float(v[5]/v[6]), payload_nmse=float(v[2]),
                  rms_evm_percent=float(100*np.sqrt(v[2])),
                  effective_snr_db=float(-10*np.log10(max(v[2], 1e-30))))
    result['raw_metrics'] = {name: float(value) if np.isfinite(value) else None
                             for name, value in zip(QUALITY_METRICS, v)}
    if np.isfinite(v[7:10]).all():
        result['paired_output_snr_db'] = float(10*np.log10(v[7]/v[8]))
    return result


def load_record(study, phase, row, digest, preflight):
    folder = study/phase
    protocol = json.loads((folder/'protocol.json').read_text())
    if (protocol['data_manifest_sha256'] != digest or protocol.get('frame') != 5
            or protocol.get('apd_draws', protocol.get('draws')) != 8
            or protocol['metric_order'][:10] != QUALITY_METRICS):
        raise ValueError('评分协议与固定案例不一致：'+phase)
    if not preflight and json.loads((folder/'progress.json').read_text())['status'] != 'complete':
        raise ValueError('正式案例导出须等待完整评分：'+phase)
    verify_sources(protocol['source_sha256'])
    path = folder/'records'/('environment_%05d.npz' % row['index'])
    meta = json.loads(path.with_suffix('.json').read_text())
    fp = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (meta['fingerprint'] != fp or meta['environment_id'] != row['environment_id']
            or meta['index'] != row['index'] or sha256(path) != meta['sha256']):
        raise ValueError('已提交评分的来源或身份改变：'+phase)
    names = protocol.get('methods') or [m['name'] for m in protocol['bundle']['methods']]
    with np.load(path) as f:
        values, controls = f['metrics'].copy(), f['control_code'].copy()
        digital_snr = f['mrc_physical_reference_snr_db'].copy() if 'mrc_physical_reference_snr_db' in f else None
        if str(f['environment_id']) != row['environment_id']:
            raise ValueError('评分数组的环境身份错误。')
    if values.shape != (17, len(names), len(protocol['metric_order'])) or controls.shape != (17, len(names), 128):
        raise ValueError('评分数组尺寸不完整。')
    reference = dict(phase=phase, protocol_sha256=sha256(folder/'protocol.json'),
                     record_sha256=meta['sha256'], record=path.name)
    return dict(methods=names, values=values, codes=controls, digital_snr=digital_snr,
                metric_order=protocol['metric_order'], reference=reference)


def band_to_iq(coeff, cfg):
    spectrum = np.zeros(np.asarray(coeff).shape[:-1]+(cfg.iq_samples,), complex)
    spectrum[..., cfg.band_offsets % cfg.iq_samples] = coeff
    return np.fft.ifft(spectrum, axis=-1)*cfg.iq_samples


def photonic_trace(engine, clean_engine, code, payload, seed, fc):
    """独立走512点I/Q→三块FFT→导频均衡，与批量投影评分交叉检查。"""
    u = code/LEVELS
    if not np.array_equal(engine.codes(u), code):
        raise ValueError('控制档位重放不一致。')
    iq = []; received = []; raw = []; gains = []; draw_values = []
    for draw in range(8):
        r = evaluate_record(engine, u, payload, rng_for(seed, fc, 105, 5, draw))
        got = r['received_qpsk']
        bad_i = (got.real >= 0) != (payload.real >= 0)
        bad_q = (got.imag >= 0) != (payload.imag >= 0)
        draw_values.append([r['bit_errors'], 62, r['payload_nmse'],
                            int((bad_i | bad_q).sum()), 31, int((bad_i | bad_q).any()), 1])
        iq.append(r['iq_a']); received.append(got); raw.append(r['raw_payload_a'])
        gains.append(r['pilot_estimated_gain_a'])
    a = np.asarray(draw_values)
    value = a.sum(axis=0); value[2] = a[:, 2].mean()
    clean_iq = clean_engine.iq(u, None)
    antenna_noisy_iq = engine.iq(u, None)
    optical_dc = engine.state(u)['optical_dc_w']
    _, variance = engine._noise(optical_dc)
    # 时域均方功率独立核对原评价器的频域Parseval计算。
    clean_power = np.mean(abs(clean_iq)**2)
    noise_power = np.mean(abs(antenna_noisy_iq-clean_iq)**2)+255*variance
    value = np.r_[value, clean_power, noise_power, optical_dc]
    return dict(metrics=value, draw_metrics=a, received_qpsk=np.asarray(received),
                raw_payload_a=np.asarray(raw), pilot_estimated_gain_a=np.asarray(gains),
                iq_a=np.asarray(iq), clean_iq_a=clean_iq,
                antenna_noisy_iq_without_apd_a=antenna_noisy_iq)


def digital_trace(env, fc):
    """MRC沿用独立数字接收链和自身符号，不伪造光子控制或光电流单位。"""
    cfg = NativeConfig(); h = response(deserialize_environment(env), cfg, fc*1e9)
    weights = h.conj()/np.maximum(np.sqrt(np.sum(abs(h)**2, axis=0)), 1e-30)
    gain = np.sqrt(1e-3*10**(env['power_dbm']/10)/31)*np.sum(weights*h, axis=0)
    variance = KB*cfg.temperature_k*10**.2*cfg.bandwidth_hz/31
    traces = []; values = []
    for draw in range(8):
        r = evaluate_output(gain, np.full(31, variance), cfg,
                            rng_for(env['seed'], fc, 105, 5, draw), payload_symbols=1, keep_arrays=True)
        a = r['arrays']; got = a['equalized']; sent = a['sent_qpsk']
        bad = ((got.real >= 0) != (sent.real >= 0)) | ((got.imag >= 0) != (sent.imag >= 0))
        values.append([r['bit_errors'], 62, (r['evm_percent']/100)**2, int(bad.sum()), 31, int(bad.any()), 1])
        traces.append(a)
    a = np.asarray(values); v = a.sum(axis=0); v[2] = a[:, 2].mean()
    arrays = {k: np.asarray([t[k] for t in traces]) for k in traces[0]}
    return dict(metrics=np.r_[v, np.full(3, np.nan)], draw_metrics=a,
                physical_reference_snr_db=r['snr_db'], **arrays)


def export_carrier(output, fc, env, x, public, records):
    cfg = NativeConfig(); seed = env['seed']; ci = fc-4
    engine, payload = frame_engine(env, fc, public['pilot_qpsk'], 5)
    clean_engine = clean_frame_engine(env, fc, public['pilot_qpsk'], payload)
    tx_coeff = transmit_coefficients(public['pilot_qpsk'], payload, cfg)
    channel = channel_at_offsets(env, cfg, fc*1e9)
    antenna_clean = np.sqrt(1e-3*10**(env['power_dbm']/10))*channel*tx_coeff[None]
    antenna_noise = np.sqrt(KB*cfg.temperature_k*cfg.df_hz)*cn(
        rng_for(seed, fc, 102, 5), antenna_clean.shape)
    arrays = dict(public_x_frame0=x, pilot_qpsk=public['pilot_qpsk'], payload_qpsk=payload,
                  tx_iq_normalized=band_to_iq(tx_coeff, cfg),
                  antenna_clean_iq_sqrt_w=band_to_iq(antenna_clean, cfg),
                  antenna_noisy_iq_sqrt_w=band_to_iq(antenna_clean+antenna_noise, cfg),
                  branch_band_w=engine.branch_band_w, branch_dc_w=engine.branch_dc_w,
                  channel_coefficients_scoring_only=channel,
                  time_us=np.arange(cfg.iq_samples)/cfg.iq_sample_rate_hz*1e6)
    cases = []; keys = []; codes = []; metrics = []; draw_metrics = []
    currents = []; clean_currents = []; got_all = []; gains = []; raw = []; cache = {}
    for phase, record in records.items():
        for mi, name in enumerate(record['methods']):
            method = phase+'/'+name; v = record['values'][ci, mi, :10]
            if name == 'mrc':
                digital = digital_trace(env, fc)
                checked_equal(digital['metrics'], v, method)
                checked_equal(digital['physical_reference_snr_db'], record['digital_snr'][ci], method+' SNR')
                for key, value in digital.items():
                    arrays['mrc_'+key] = np.asarray(value)
                cases.append(dict(method=method, photonic=False, privileged=True,
                    own_digital_payload=True, quality=quality(v),
                    digital_physical_reference_snr_db=digital['physical_reference_snr_db'],
                    display_draw=0, sent=complex_list(digital['sent_qpsk'][0, :, 0]),
                    received=complex_list(digital['equalized'][0, :, 0])))
                continue
            code = record['codes'][ci, mi]
            if np.any(code < 0) or np.any(code > LEVELS) or not np.array_equal(code, np.rint(code)):
                raise ValueError('非法控制码：'+method)
            # 同一控制跨流水线复用相同波形，但仍逐一核对各自原始评分。
            key = tuple(code.tolist())
            if key not in cache:
                cache[key] = photonic_trace(engine, clean_engine, code, payload, seed, fc)
            trace = cache[key]; checked_equal(trace['metrics'], v, method)
            calls = record['values'][ci, mi, record['metric_order'].index('feedback_calls')]
            cases.append(dict(method=method, photonic=True, privileged=name=='teacher',
                feedback_calls=int(calls) if np.isfinite(calls) else None,
                display_draw=0, quality=quality(v), received=complex_list(trace['received_qpsk'][0]),
                raw_iq_a=complex_list(trace['iq_a'][0]), delay_code=code[:64].tolist(),
                attenuation_code=code[64:].tolist(), delay_ps=(code[:64]/cfg.sample_rate_hz*1e12).tolist(),
                attenuation_db=(code[64:]*.5).tolist()))
            keys.append(method); codes.append(code); metrics.append(trace['metrics'])
            draw_metrics.append(trace['draw_metrics']); currents.append(trace['iq_a'])
            clean_currents.append(trace['clean_iq_a']); got_all.append(trace['received_qpsk'])
            raw.append(trace['raw_payload_a']); gains.append(trace['pilot_estimated_gain_a'])
    arrays.update(methods=np.asarray(keys), control_code=np.asarray(codes), metrics=np.asarray(metrics),
                  draw_metrics=np.asarray(draw_metrics), received_qpsk=np.asarray(got_all),
                  received_iq_a=np.asarray(currents), clean_iq_a=np.asarray(clean_currents),
                  raw_payload_a=np.asarray(raw), pilot_estimated_gain_a=np.asarray(gains))
    stem = 'carrier_%02d' % fc; path = output/(stem+'.npz'); atomic_npz(path, **arrays)
    # 同一载频的全部光子方法共享坐标范围，只用固定draw0决定绘图范围。
    received0 = np.asarray(got_all)[:, 0]
    limit = max(1.2, 1.08*float(max(np.max(abs(received0.real)), np.max(abs(received0.imag)))))
    page = dict(carrier_ghz=fc, frame=5, scoring_draws=8, display_draw=0,
                source_scope='fixed old test index0; exploratory, not sealed confirmation',
                sent_qpsk=complex_list(payload), tx_iq_normalized=complex_list(arrays['tx_iq_normalized']),
                time_us=arrays['time_us'].tolist(), constellation_axis_limit=limit, methods=cases,
                units=dict(tx_iq='normalized; mean squared magnitude 1',
                           antenna_iq='sqrt(W); RF-equivalent complex envelope', received_iq='A',
                           equalized_constellation='dimensionless', time='microseconds'),
                physics=dict(no_independent_interference=True, no_added_temperature_drift=True,
                    optical_combining='64 branches before one photodetector', fixed_optical_loss_db=5),
                control_meaning='effective software delay/attenuation codes; not verified physical switch wiring',
                npz_sha256=sha256(path), metrics_verified_against_committed_records=True)
    write_json(output/(stem+'.json'), page)
    with (output/(stem+'_controls.csv')).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'antenna_index', 'array_row', 'array_column', 'delay_code',
                         'delay_ps', 'attenuation_code', 'attenuation_db', 'total_optical_loss_db'])
        for key, code in zip(keys, codes):
            for antenna in range(64):
                writer.writerow([key, antenna, antenna//8, antenna%8, code[antenna],
                    code[antenna]/cfg.sample_rate_hz*1e12, code[64+antenna],
                    .5*code[64+antenna], 5+.5*code[64+antenna]])
    return dict(carrier_ghz=fc, photonic_methods=len(keys), all_methods=len(cases),
                unique_controls=len(cache), verified_quality_values=len(cases)*10,
                files={f.name: sha256(f) for f in output.glob(stem+'*')})


def run(project, output, phases, carriers, preflight):
    require_host(); data = project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    study = project/'dataset_simulation/baseline_results/20260925_full_baselines'
    manifest = check_data(data); rows = [r for r in manifest['environments'] if r['split']=='test']
    if len(rows) != 216 or rows[0]['index'] != 0:
        raise ValueError('本入口固定旧216测试的索引0。')
    if not preflight and (phases != PHASES or carriers != list(range(4, 21))):
        raise ValueError('正式导出必须包括七条流水线和全部17载频。')
    row = rows[0]; digest = sha256(data/'manifest.json')
    records = {phase: load_record(study, phase, row, digest, preflight) for phase in phases}
    public = public_data(data); env = json.loads((data/row['path']/'environment.json').read_text())
    with np.load(data/row['path']/'data.npz') as f:
        x = f['X'].copy()
    if x.shape != (17, 2513) or env['seed'] != row['seed']:
        raise ValueError('案例输入尺寸或种子错误。')
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(scope='preflight_subset' if preflight else 'full_old_test_example',
        environment_index=0, environment_id=row['environment_id'], seed=row['seed'],
        phases=phases, carriers_ghz=carriers, data_manifest_sha256=digest,
        references={k: v['reference'] for k, v in records.items()},
        source_sha256=source_record(SOURCES), frame=5, display_draw=0, metric_draws=8,
        quality_relative_tolerance=1e-12, quality_absolute_tolerance=0,
        online_input='frame0 public X2513; frame5 payload/channel used only by offline evaluator')
    write_json(output/'protocol.json', protocol); write_json(output/'environment.json', env)
    for name in SOURCES:
        dst = output/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dst)
    entries = []
    for fc in carriers:
        entries.append(export_carrier(output, fc, env, x[fc-4], public, records))
        state = dict(status='running', completed_carriers=len(entries), total_carriers=len(carriers),
                     pid=os.getpid(), at=now())
        write_json(output/'progress.json', state); print(json.dumps(state), flush=True)
    verify_sources(protocol['source_sha256'])
    for phase in phases:
        # 在整个导出结束时再读一遍提交记录，防止处理中原始评分被换掉。
        current = load_record(study, phase, row, digest, preflight)
        if current['reference'] != records[phase]['reference']:
            raise ValueError('导出期间原始评分改变。')
    write_json(output/'summary.json', dict(status='passed_preflight' if preflight else 'complete',
        entries=entries, protocol_sha256=sha256(output/'protocol.json'),
        all_quality_replays_passed=True, scope=protocol['scope'], at=now()))
    write_json(output/'progress.json', dict(status='complete', scope=protocol['scope'], at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phases', nargs='+', choices=PHASES, default=PHASES)
    p.add_argument('--carriers', nargs='+', type=int, choices=range(4, 21), default=list(range(4, 21)))
    p.add_argument('--preflight', action='store_true'); a = p.parse_args()
    try:
        run(a.project, a.output, a.phases, a.carriers, a.preflight)
    except BaseException:
        if a.output.exists():
            write_json(a.output/('failure_%d.json' % os.getpid()), dict(traceback=traceback.format_exc(), at=now()))
        raise
