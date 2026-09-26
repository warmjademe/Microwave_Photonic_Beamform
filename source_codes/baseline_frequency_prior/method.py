"""同一训练候选质量的固定选择器：不用当前测量预测，仅检查训练先验的作用。"""
import numpy as np


def fit(robust_quality):
    quality = np.asarray(robust_quality, float)
    if quality.ndim != 3 or quality.shape[1:] != (17, 64) or not np.all(np.isfinite(quality)):
        raise ValueError('需要训练环境×17载频×64候选的有限NMSE。')
    mean_by_carrier = quality.mean(axis=0)
    global_mean = mean_by_carrier.mean(axis=0)
    return dict(global_candidate=int(global_mean.argmin()),
        carrier_candidates=mean_by_carrier.argmin(axis=-1).astype(int),
        mean_training_nmse_by_carrier=mean_by_carrier, global_mean_training_nmse=global_mean)


def predict_indices(model, carrier_ghz):
    frequencies = np.asarray(carrier_ghz)
    if not np.all((frequencies >= 4) & (frequencies <= 20) & (frequencies == np.rint(frequencies))):
        raise ValueError('载频须为4–20 GHz整数。')
    return np.stack([np.full(frequencies.shape, model['global_candidate']),
        np.asarray(model['carrier_candidates'])[frequencies.astype(int)-4]], axis=-1)
