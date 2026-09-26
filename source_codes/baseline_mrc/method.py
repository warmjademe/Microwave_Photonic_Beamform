"""额外拥有64路数字接收和完整信道的理想 RF MRC 参考。"""
import numpy as np
from baseline_common.config import KB
from baseline_common.metrics import evaluate_output


def evaluate(channel, power_dbm, cfg, rng, noise_figure_db=2.0, payload_symbols=256):
    h = np.asarray(channel, complex)
    if h.shape != (cfg.n, cfg.tones):
        raise ValueError('MRC 需要各阵元、各频点的完整信道。')
    norm = np.sqrt(np.sum(abs(h)**2, axis=0))
    weights = np.conj(h)/np.maximum(norm, 1e-30)
    # y=sum(weights*x)，因此这里用 conj(h)，而不是再做一次共轭。
    power = 1e-3*10**(power_dbm/10)/cfg.tones
    gain = np.sqrt(power)*np.sum(weights*h, axis=0)
    variance = KB*cfg.temperature_k*10**(noise_figure_db/10)*cfg.bandwidth_hz/cfg.tones
    result = evaluate_output(gain, np.full(cfg.tones, variance), cfg, rng, payload_symbols)
    return dict(result, reference_type='ideal_RF_64_digital_channels_true_CSI',
                uses_photonic_controls=False, privileged_true_channel=True,
                is_proven_upper_bound_for_photonic_hardware=False,
                noise_figure_db=noise_figure_db)
