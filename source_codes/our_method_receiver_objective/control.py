"""在相同搜索预算下，仅替换控制目标；不读取真实信道或数据符号。"""
import math
import numpy as np
from our_method_response_control.physics import design, LEVELS
from native_sim.control_engine import (THERMAL_PSD_A2_HZ,
    RESPONSIVITY_A_PER_W, DARK_CURRENT_A, DEMODULATOR_GAIN)


def score_signal(model, signal, antenna_noise, dc, objective):
    """MMSE与理想已知响应QPSK误码代理，都以分数越大越好定义。"""
    psd = THERMAL_PSD_A2_HZ + model['shot_factor'] * (
        RESPONSIVITY_A_PER_W * np.asarray(dc) + DARK_CURRENT_A)
    noise = antenna_noise + (DEMODULATOR_GAIN**2 * model['cfg'].df_hz * psd)[..., None] * model['apd_row_norm']
    if objective == 'mmse':
        return -np.mean(noise / np.maximum(abs(signal)**2 + noise, np.finfo(float).tiny), axis=-1)
    if objective == 'ber':
        gamma = abs(signal)**2 / np.maximum(noise, np.finfo(float).tiny)
        # gamma为复符号能量/复噪声方差，Gray QPSK的逐比特错误率。
        values = np.fromiter((math.erfc(v) for v in np.sqrt(gamma / 2).flat),
            dtype=float, count=gamma.size).reshape(gamma.shape)
        return -np.mean(.5 * values, axis=-1)
    raise ValueError('目标必须是mmse或ber。')


def decode(h, carrier_ghz, initial_control, objective='ber', sweeps=2):
    """与冻结解码器保持相同起点、128维扫描顺序、档位和两轮预算。"""
    h = np.asarray(h, complex)
    if h.shape != (64, 31) or not np.isfinite(h).all():
        raise ValueError('响应应为64×31有限复数。')
    u = np.asarray(initial_control, float)
    if u.shape != (128,) or not np.isfinite(u).all() or sweeps != 2:
        raise ValueError('本诊断只接受128维合法输入及固定两轮。')
    m = design(carrier_ghz)
    code = np.rint(np.clip(u, 0, 1)*LEVELS).astype(int)
    trans = m['transmission'][code[64:]]
    phase = m['phase']; noise_table = m['antenna_payload_variance']
    branch = h*phase[code[:64]]*trans[:, None]
    signal = branch.sum(0)
    antenna_noise = np.sum(noise_table*trans[:, None]**2, axis=0)
    dc = float(trans @ m['dc'])
    initial = float(score_signal(m, signal, antenna_noise, dc, objective)); score = initial
    evaluations = 1
    for _ in range(sweeps):
        for index in range(128):
            n = index % 64
            if index < 64:
                replacement = h[n][None, :]*phase*trans[n]
                nn = np.broadcast_to(antenna_noise, replacement.shape)
                pp = np.full(77, dc)
            else:
                candidate_trans = m['transmission']
                replacement = h[n][None, :]*phase[code[n]][None, :]*candidate_trans[:, None]
                nn = antenna_noise[None, :] + noise_table[n][None, :] * (candidate_trans[:, None]**2-trans[n]**2)
                pp = dc+(candidate_trans-trans[n])*m['dc'][n]
            ss = signal[None, :] - branch[n][None, :] + replacement
            values = score_signal(m, ss, nn, pp, objective)
            best = int(values.argmax()); evaluations += len(values)
            if values[best] > score + 1e-14:
                code[index] = best; signal = ss[best]; antenna_noise = nn[best].copy()
                dc = float(pp[best]); branch[n] = replacement[best]
                if index >= 64:
                    trans[n] = m['transmission'][best]
                score = float(values[best])
    if not np.isfinite(score) or score < initial-1e-12:
        raise ValueError('代理目标非有限或下降。')
    return code/LEVELS, dict(objective=objective, initial_proxy=initial,
        final_proxy=score, coordinate_sweeps=2, candidate_evaluations=evaluations,
        extra_feedback=0)
