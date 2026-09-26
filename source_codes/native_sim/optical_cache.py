"""每路完整复光场的非循环自相关；保留目标RF带和DC供控制搜索。"""
import numpy as np


def make_cache(fields,centers_hz,carrier_hz,cfg,chunk=8):
    f=np.asarray(fields,complex)
    if f.shape!=(cfg.n,cfg.sample_count): raise ValueError('每路必须有完整时域复光场。')
    spacing=float(np.min(np.diff(np.sort(centers_hz))))
    cross_min=spacing-(cfg.sample_count-1)*cfg.df_hz
    if cross_min<=2*(carrier_hz+cfg.bandwidth_hz/2):
        raise ValueError('跨载波拍频侵入所需信号或噪声和频支持。')
    optical_low, optical_high=192.95e12-3.2e12,192.95e12+3.2e12
    if np.min(centers_hz)-cfg.sample_rate_hz/2<optical_low or np.max(centers_hz)+cfg.sample_rate_hz/2>optical_high:
        raise ValueError('6.4THz探测窗口未覆盖完整光谱。')
    n=cfg.sample_count;nfft=1<<(2*n-2).bit_length()
    bins=int(round(carrier_hz/cfg.df_hz))+cfg.band_offsets
    band=np.empty((cfg.n,len(bins)),complex);dc=np.empty(cfg.n)
    for start in range(0,cfg.n,chunk):
        spectrum=np.fft.fftshift(np.fft.fft(f[start:start+chunk],axis=-1),axes=-1)/n
        transformed=np.fft.fft(spectrum,n=nfft,axis=-1)
        corr=np.fft.ifft(abs(transformed)**2,axis=-1)
        band[start:start+chunk]=corr[:,bins]
        dc[start:start+chunk]=corr[:,0].real
    if not np.all(np.isfinite(band)) or np.any(dc<=0): raise ValueError('非有限/非正光缓存。')
    return band,dc
