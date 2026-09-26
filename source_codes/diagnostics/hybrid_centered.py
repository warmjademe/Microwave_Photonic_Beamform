"""保守混合计算：小振幅走线性核，强响应逐路回退到非线性RK4。

门限按预测状态的Fourier系数L1上界设定，与控制效果和训练/测试划分无关。
本候选仍须通过完整数值检查，不修改先前线性核失败记录或1%验收门限。
"""
from functools import lru_cache
import os
from pathlib import Path
import numpy as np
from native_sim.laser import load_profile
from native_sim.waveforms import transmit_coefficients, channel_at_offsets
from baseline_common.config import cn, KB
from linear_centered import transfers, from_positive_coefficients
from staged_clock import load_staged_clock, stage_samples

PHOTON_RELATIVE_BOUND = .35
CARRIER_RELATIVE_BOUND = .04


@lru_cache(maxsize=1)
def nonlinear_solver():
    # 每进程独立构建目录，避免多worker同时替换已加载的共享库。
    path = Path(__file__).resolve().parent/'_hybrid_build'/str(os.getpid())
    path.mkdir(parents=True, exist_ok=True)
    return load_staged_clock(path)


def hybrid_from_coefficients(coefficients, carrier_hz, cfg):
    coefficients = np.asarray(coefficients, complex)
    profile = load_profile(); models = transfers(float(carrier_hz), 2)
    band, dc = from_positive_coefficients(coefficients, carrier_hz, cfg, include_transient=True)
    bounds = np.asarray([2*np.sum(abs(m['response']*(p['modulation_peak_current_a']*c)[:, None]), axis=0)
                         / m['state_bias'] for m, p, c in zip(models, profile, coefficients)])
    selected = np.flatnonzero((bounds[:, 0] > CARRIER_RELATIVE_BOUND)
                             | (bounds[:, 1] > PHOTON_RELATIVE_BOUND))
    factor = 2; fs = cfg.sample_rate_hz*factor
    bins = int(round(carrier_hz/cfg.df_hz))+cfg.band_offsets
    align = np.exp(-2j*np.pi*(carrier_hz+cfg.band_offsets*cfg.df_hz)/fs)
    if len(selected):
        solver = nonlinear_solver()
        for start in range(0, len(selected), 8):
            indices = selected[start:start+8]
            stage = stage_samples(coefficients[indices], carrier_hz, cfg, factor)
            field, _ = solver(stage, fs, [profile[i] for i in indices])
            intensity = abs(field)**2
            spectrum = np.fft.rfft(intensity, axis=-1)/intensity.shape[-1]
            band[indices] = spectrum[:, bins]*align[None, :]
            dc[indices] = spectrum[:, 0].real
    return band, dc, dict(nonlinear_routes=int(len(selected)), selected_routes=selected.tolist(),
        state_bound_max=bounds.max(0).tolist(),
        thresholds=dict(carrier=CARRIER_RELATIVE_BOUND, photon=PHOTON_RELATIVE_BOUND))


def approximate_cache(environment, carrier_hz, pilots, payload, rng, cfg, return_details=False):
    coeff = transmit_coefficients(pilots, payload, cfg)
    channel = channel_at_offsets(environment, cfg, carrier_hz)
    power = 1e-3*10**(float(environment['power_dbm'])/10)
    clean = np.sqrt(power)*channel*coeff[None, :]
    noise = np.sqrt(KB*cfg.temperature_k*cfg.df_hz)*cn(rng, clean.shape)
    b, dc, details = hybrid_from_coefficients(500*(clean+noise), carrier_hz, cfg)
    return (b, dc, details) if return_details else (b, dc)
