"""所有可实现算法共用同一套限幅、量化与几何控制。"""
import numpy as np
from .config import C0


def direction(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.stack([np.sin(az)*np.cos(el), np.sin(el), np.cos(az)*np.cos(el)], -1)


def project(control, cfg):
    u = np.asarray(control, dtype=float)
    if u.shape != (2*cfg.n,) or not np.all(np.isfinite(u)):
        raise ValueError('控制必须是有限的 [128] 数组。')
    levels = np.r_[np.full(cfg.n, cfg.delay_levels), np.full(cfg.n, cfg.attenuation_levels)]
    return np.floor(np.clip(u, 0, 1)*levels+0.5)/levels


def codes(control, cfg):
    u = project(control, cfg)
    return np.rint(u*np.r_[np.full(cfg.n, cfg.delay_levels),
                           np.full(cfg.n, cfg.attenuation_levels)]).astype(np.int16)


def physical_units(control, cfg):
    u = project(control, cfg)
    return u[:cfg.n]*cfg.delay_max_ps, u[cfg.n:]*cfg.attenuation_max_db


def geometric_control(cfg, az_deg, el_deg):
    advance = cfg.positions@direction(az_deg, el_deg)/C0
    delay = (advance-advance.min())*1e12
    if delay.max() > cfg.delay_max_ps+1e-8:
        raise ValueError('当前方向所需时延超出器件范围。')
    return project(np.r_[delay/cfg.delay_max_ps, np.zeros(cfg.n)], cfg)


def probe_codebook(cfg):
    """所有环境共享的 4×4 方向网格，不读取真实方向或真实信道。"""
    angles = np.array([(az, el) for az in np.linspace(-35, 35, 4)
                       for el in np.linspace(-15, 15, 4)])
    return np.stack([geometric_control(cfg, az, el) for az, el in angles]), angles
