"""统一接收后端：独立参考块估计增益，独立数据块评价。"""
import numpy as np
from .config import cn, qpsk
from .data import estimate_from_iq


def waveform(frequency_symbols, cfg):
    """频率递增数组→OFDM时域；平均时域功率等于频点功率之和。"""
    x = np.fft.ifft(np.fft.ifftshift(frequency_symbols, axes=0), axis=0)*cfg.tones
    cp = int(round(cfg.cp_s*cfg.bandwidth_hz))
    if cp >= cfg.tones:
        raise ValueError('当前波形例程要求 CP 小于一个有效块。')
    return np.concatenate([x[-cp:], x], axis=0).T.reshape(-1) if cp else x.T.reshape(-1)


def evaluate_output(gain, variance, cfg, rng, payload_symbols=256, keep_arrays=False):
    gain = np.asarray(gain, complex)
    variance = np.broadcast_to(variance, gain.shape)
    pilot = qpsk(rng, (cfg.tones, cfg.pilot_symbols))
    pilot_rx = gain[:, None]*pilot+np.sqrt(variance[:, None])*cn(rng, pilot.shape)
    estimate, noise_est, _ = estimate_from_iq(pilot_rx, pilot)
    sent = qpsk(rng, (cfg.tones, payload_symbols))
    clean = gain[:, None]*sent
    noise = np.sqrt(variance[:, None])*cn(rng, sent.shape)
    received = clean+noise
    # 所有光子控制算法使用同一个基于带噪参考估计的 LMMSE 后端。
    equalizer = np.conj(estimate)/(abs(estimate)**2+noise_est)
    equalized = equalizer[:, None]*received
    errors = int(np.count_nonzero((equalized.real >= 0) != (sent.real >= 0))+
                 np.count_nonzero((equalized.imag >= 0) != (sent.imag >= 0)))
    out = dict(snr_db=float(10*np.log10(max(float(np.sum(abs(gain)**2)/variance.sum()), 1e-30))),
               evm_percent=float(100*np.sqrt(np.mean(abs(equalized-sent)**2))),
               bit_errors=errors, bits_tested=int(2*sent.size), ber=float(errors/(2*sent.size)),
               equalizer='estimated_gain_LMMSE; independent pilot and payload',
               evaluation_pilot_blocks=cfg.pilot_symbols, payload_blocks=payload_symbols)
    if keep_arrays:
        out['arrays'] = dict(sent_qpsk=sent, received=received, clean=clean, noise=noise,
                             equalized=equalized, gain_estimate=estimate,
                             sent_time=waveform(sent/np.sqrt(cfg.tones), cfg),
                             received_time=waveform(received, cfg))
    return out
