"""只在训练集拟合标准化；I/Q使用相同尺度，保留幅度与相位信息。"""
import numpy as np


def transform_aux(x):
    a = np.asarray(x[:, 1984:], np.float64).copy()
    a[:, 1:17] = np.sign(a[:, 1:17]) * np.log1p(abs(a[:, 1:17]))
    # 噪声方差和直流为正，固定对数变换压缩量纲跨度，不按样本归一化。
    a[:, 17:] = np.log10(np.maximum(a[:, 17:], 1e-30))
    return a


def fit(dataset):
    if dataset.split != 'train':
        raise ValueError('标准化器只允许使用训练集。')
    total = np.zeros(2513, np.float64)
    for start in range(0, len(dataset), 4096):
        x = np.asarray(dataset.X[start:start + 4096], np.float64).copy()
        x[:, 1984:] = transform_aux(x)
        total += x.sum(0)
    mean = total / len(dataset)
    # 二次遍历计算中心化平方，避免近常量噪声/DC中E[x²]-E[x]²的消去误差。
    square = np.zeros(2513, np.float64)
    for start in range(0, len(dataset), 4096):
        x = np.asarray(dataset.X[start:start + 4096], np.float64).copy()
        x[:, 1984:] = transform_aux(x)
        square += np.square(x - mean).sum(0)
    variance = square / len(dataset)
    variance[:1984] = np.repeat(variance[:1984].reshape(-1, 2).mean(-1), 2)
    scale = np.sqrt(np.maximum(variance, 1e-30))
    # 对数辅助量完全不变时保留为零附近，不放大舍入误差。
    scale[1984:] = np.where(scale[1984:] < 1e-12, 1., scale[1984:])
    return dict(mean=mean, scale=scale)


def apply(x, stats):
    z = np.asarray(x, np.float64).copy()
    z[:, 1984:] = transform_aux(x)
    z = ((z - stats['mean']) / stats['scale']).astype(np.float32)
    if not np.all(np.isfinite(z)):
        raise ValueError('标准化后包含非有限值。')
    return z
