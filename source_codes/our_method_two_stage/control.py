"""阶段B：确定性多起点控制，以及相同代理模型下的保底选择。

输入只有估计复响应、载频和初始合法控制；没有传播真值或未知数据。
代理质量是设计依据，真实接收质量必须由独立完整接收器检验。
"""
import math
import numpy as np
from our_method_response_control.physics import design, decode, LEVELS
from native_sim.control_engine import (THERMAL_PSD_A2_HZ, RESPONSIVITY_A_PER_W,
                                      DARK_CURRENT_A, DEMODULATOR_GAIN)

PROXY_FIELDS = ['ideal_csi_qpsk_ber', 'mmse_nmse', 'tone_signal_noise_ratio']


def proxy_quality(h, carrier_ghz, controls):
    """理想已知响应下的QPSK误码代理、MMSE代理和频点总信噪比。

    gamma是复符号能量/复噪声方差；Gray QPSK的比特错误率为
    0.5*erfc(sqrt(gamma/2))。此公式忽略导频估计误差和实际链路残差。
    """
    h = np.asarray(h, complex)
    if h.shape != (64, 31) or not np.isfinite(h).all():
        raise ValueError('估计响应应为64×31有限复数。')
    u = np.atleast_2d(np.asarray(controls, float))
    if u.shape[1] != 128 or not np.isfinite(u).all():
        raise ValueError('控制应为有限的128维向量。')
    code = np.rint(np.clip(u, 0, 1)*LEVELS).astype(int)
    m = design(carrier_ghz); trans = m['transmission'][code[:, 64:]]
    signal = np.sum(h[None]*m['phase'][code[:, :64]]*trans[:, :, None], axis=1)
    noise = np.einsum('cn,nk->ck', trans**2, m['antenna_payload_variance'])
    dc = trans @ m['dc']
    psd = THERMAL_PSD_A2_HZ+m['shot_factor']*(RESPONSIVITY_A_PER_W*dc+DARK_CURRENT_A)
    noise += (DEMODULATOR_GAIN**2*m['cfg'].df_hz*psd)[:, None]*m['apd_row_norm']
    power = abs(signal)**2; gamma = power/np.maximum(noise, np.finfo(float).tiny)
    ber = np.fromiter((.5*math.erfc(math.sqrt(v/2)) for v in gamma.flat),
                      dtype=float, count=gamma.size).reshape(gamma.shape).mean(axis=1)
    return np.column_stack([ber, np.mean(1/(1+gamma), axis=1),
                            power.sum(axis=1)/noise.sum(axis=1)])


def phase_starts(h, carrier_ghz):
    """四种共同参考相位；逐路选择中心频点相位最接近的合法延时码。

    只改变算法起点，不增加现场测量。相位相同的不同延时在宽带上并不等价。
    """
    h = np.asarray(h, complex); m = design(carrier_ghz)
    result = []
    for rotation in np.arange(4)*np.pi/2:
        phase_error = np.angle(np.exp(1j*(np.angle(h[:, 15])[:, None]
                               +np.angle(m['phase'][:, 15])[None]-rotation)))
        code = np.r_[np.abs(phase_error).argmin(axis=1), np.zeros(64, int)]
        result.append(code/LEVELS)
    return np.asarray(result)


def compare_controls(h, carrier_ghz, initial):
    """返回四个预设控制器；每种实际执行成本单独计时，候选不隐藏。

    B0=1起点×2轮；相同扫描预算对照=1起点×10轮；多起点=5起点×2轮。
    代理保底版从与B0相比MMSE不增、频点总SNR不降的候选中选最低BER。
    保底只针对估计响应代理，不代表实际接收指标一定不下降。
    """
    import time
    tick = time.perf_counter()
    base, info = decode(h, carrier_ghz, initial, sweeps=2)
    base_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    long, _ = decode(h, carrier_ghz, initial, sweeps=10)
    long_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    candidates = np.vstack([base, [decode(h, carrier_ghz, start, sweeps=2)[0]
                                 for start in phase_starts(h, carrier_ghz)]])
    quality = proxy_quality(h, carrier_ghz, candidates)
    # 质量的浮点容差只覆盖舍入，不能成为人为放宽约束的超参数。
    eligible = ((quality[:, 1] <= quality[0, 1]+1e-12)
                & (quality[:, 2] >= quality[0, 2]*(1-1e-12)))
    indices = np.flatnonzero(eligible)
    gated = int(indices[np.argmin(quality[indices, 0])])
    mmse = int(np.argmin(quality[:, 1]))
    multistart_seconds = base_seconds+time.perf_counter()-tick
    controls = np.stack([base, long, candidates[mmse], candidates[gated]])
    return controls, dict(names=['base_2sweeps', 'single_10sweeps',
        'multi_mmse', 'multi_gated_ber'], proxy=proxy_quality(h, carrier_ghz, controls).tolist(),
        candidate_proxy=quality.tolist(), selected_indices=[mmse, gated],
        candidate_controls=candidates.tolist(), eligible=eligible.tolist(),
        seconds=[base_seconds, long_seconds, multistart_seconds, multistart_seconds],
        coordinate_sweeps=[2, 10, 10, 10], extra_feedback=0,
        base_final_objective=info['final_proxy'])
