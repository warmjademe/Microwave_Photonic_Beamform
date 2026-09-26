"""独立确认接收流程：公开探测→实际在线控制→另一帧接收评分。

本模块不生成环境计划、不选择模型、不启动新留出集。调用端必须先冻结来源。
控制器只收到公开向量和标量反馈接口；传播真值与评分符号留在评价端。
"""
import time
import numpy as np
from study_full_baselines.common import (NativeConfig, NativeControlEngine, LEVELS,
    rng_for, frame_engine, clean_frame_engine, reception_metrics, QUALITY_METRICS)
from compact_dataset import pack_observation
from baseline_common.config import qpsk
from native_sim.waveforms import transmit_coefficients, channel_at_offsets
from native_sim.control_engine import optimize_teacher
from study_full_baselines.evaluate_classic import digital_reference
from gpu_fft_hybrid import approximate_cache_cuda, hybrid_from_coefficients_cuda

METRIC_ORDER = QUALITY_METRICS+['feedback_calls', 'controller_seconds', 'feedback_simulator_seconds']


def frame(environment, carrier, pilots, number, backend):
    if backend == 'cpu': return frame_engine(environment, carrier, pilots, number)
    if backend != 'cuda_fft_cpu_rk4_v1': raise ValueError('未知接收器计算后端。')
    cfg = NativeConfig(); seed = environment['seed']
    payload = qpsk(rng_for(seed, carrier, 101, number), (31,))
    band, dc, detail = approximate_cache_cuda(environment, carrier*1e9, pilots, payload,
        rng_for(seed, carrier, 102, number), cfg, return_details=True)
    engine = NativeControlEngine(cfg, band, dc, carrier*1e9, pilots)
    engine.simulation_details = detail
    return engine, payload


def public_observation(environment, carrier, public, backend):
    """只做原始16次探测；不计算测试标签或其余48个候选的隐藏质量。"""
    engine, _ = frame(environment, carrier, public['pilot_qpsk'], 0, backend)
    measured = [engine.measure_detailed(control, rng_for(environment['seed'], carrier, 103, index))
                for index, control in enumerate(public['probe_controls'])]
    if len(measured) != 16: raise ValueError('共同探测数必须为16。')
    arrays = dict(combined_iq_a=np.stack([v['symbols'] for v in measured]),
        quality=np.asarray([v['score'] for v in measured]),
        noise_symbol_var_a2=np.stack([v['noise_symbol_var'] for v in measured]),
        probe_apd_dc_a=np.asarray([v['apd_dc_a'] for v in measured]))
    return pack_observation(arrays, carrier), engine


def decide_all(models, raw, observed_engine, seed, carrier, public, references=True):
    """在创建评分帧之前完成全部普通控制。返回控制码，不读取缓存预测。"""
    if not models or len(models) != len(set(models)): raise ValueError('控制器集合为空或重复。')
    if raw.shape != (2513,) or raw[1984] != carrier: raise ValueError('输入形状或载频不同。')
    controls = []; costs = []; details = {}; original = raw.copy()
    for name, model in models.items():
        extra = model.warm or (model.kind == 'classic' and name != 'ttd_das')
        def measure(control, call):
            return observed_engine.measure_detailed(control, rng_for(seed, carrier, 620, call))['score']
        got = model.decide(raw.copy(), rng_for(0, seed, carrier, 630), measure if extra else None)
        code = got['control_code']
        if code.shape != (128,) or np.any(code < 0) or np.any(code > LEVELS): raise ValueError('非法控制码。')
        controls.append(code); costs.append([got['feedback_calls'], got['software_seconds'], got['feedback_simulator_seconds']])
        details[name] = {k: v for k, v in got.items() if k in ['trace_control_code', 'trace_scores']}
        if not np.array_equal(raw, original): raise ValueError('算法改变了共享输入。')
    teacher_evaluations = None
    if references:
        tick = time.perf_counter()
        u, metadata = optimize_teacher(observed_engine, public['probe_controls'],
            rng_for(0, seed, carrier, 610), starts=2, sweeps=1)
        controls.append(observed_engine.codes(u)); costs.append([np.nan, time.perf_counter()-tick, 0.])
        teacher_evaluations = metadata['objective_evaluations']
    return dict(methods=list(models)+(['teacher'] if references else []),
        control_code=np.asarray(controls), costs=np.asarray(costs), feedback=details,
        teacher_objective_evaluations=teacher_evaluations)


def scoring_engines(environment, carrier, public, backend):
    """第5帧及配对无天线噪声链路，仅在控制器已输出控制后调用。"""
    scored, payload = frame(environment, carrier, public['pilot_qpsk'], 5, backend)
    if backend == 'cpu':
        clean = clean_frame_engine(environment, carrier, public['pilot_qpsk'], payload)
    else:
        cfg = NativeConfig()
        coeff = transmit_coefficients(public['pilot_qpsk'], payload, cfg)
        channel = channel_at_offsets(environment, cfg, carrier*1e9)
        power = 1e-3*10**(float(environment['power_dbm'])/10)
        band, dc, detail = hybrid_from_coefficients_cuda(500*np.sqrt(power)*channel*coeff[None], carrier*1e9, cfg)
        clean = NativeControlEngine(cfg, band, dc, carrier*1e9, public['pilot_qpsk'])
        clean.simulation_details = detail
    return scored, clean, payload


def score_decisions(environment, carrier, public, decisions, backend, include_mrc=True):
    scored, clean, payload = scoring_engines(environment, carrier, public, backend)
    quality, codes = reception_metrics(scored, clean, decisions['control_code']/LEVELS,
                                       payload, environment['seed'], carrier)
    if not np.array_equal(codes, decisions['control_code']): raise ValueError('接收器改变了合法控制码。')
    metrics = np.column_stack([quality, decisions['costs']]); names = list(decisions['methods'])
    reference_snr = None
    if include_mrc:
        digital, reference_snr = digital_reference(environment, carrier)
        metrics = np.vstack([metrics, digital]); codes = np.vstack([codes, np.full(128, -1, np.int16)])
        names.append('mrc')
    if not np.isfinite(metrics[:, :7]).all(): raise ValueError('接收质量计数缺失。')
    return dict(methods=names, control_code=codes, metrics=metrics,
        mrc_physical_reference_snr_db=reference_snr, feedback=decisions['feedback'],
        teacher_objective_evaluations=decisions['teacher_objective_evaluations'])
