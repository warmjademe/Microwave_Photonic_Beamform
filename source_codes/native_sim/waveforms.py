"""QPSK帧→多径→64路正常带内天线噪声→原生电幅度输入。"""
import numpy as np
from baseline_common.channel import deserialize_environment
from baseline_common.config import cn, KB
from baseline_common.controls import direction


def transmit_coefficients(pilots, payload, cfg):
    """三块分别是已知pilot1、已知pilot2、独立payload；固定矩形带限。"""
    symbols=np.column_stack([pilots,payload])
    if symbols.shape!=(31,3): raise ValueError('需要31子载波的两个导频及一个独立数据块。')
    source=np.zeros(cfg.iq_samples,complex)
    tones=np.arange(-15,16)
    block=cfg.useful_samples+cfg.cp_samples
    for index in range(3):
        spectrum=np.zeros(cfg.useful_samples,complex)
        spectrum[tones%cfg.useful_samples]=symbols[:,index]
        useful=cfg.useful_samples*np.fft.ifft(spectrum)
        with_cp=np.r_[useful[-cfg.cp_samples:],useful]
        source[index*block:(index+1)*block]=with_cp
    spectrum=np.fft.fft(source)/len(source)
    keep=cfg.band_offsets%cfg.iq_samples
    coefficients=spectrum[keep]
    power=float(np.sum(abs(coefficients)**2))
    if not power>0: raise ValueError('零功率发射帧。')
    # 单阵元参考功率是进入随机信道前的整帧平均实RF功率；不逐阵元归一化。
    return coefficients/np.sqrt(power)


def channel_at_offsets(environment, cfg, carrier_hz):
    e=deserialize_environment(environment)
    geometric=cfg.positions@direction(e['angles_deg'][:,0],e['angles_deg'][:,1]).T/299792458.0
    frequency=carrier_hz+cfg.band_offsets*cfg.df_hz
    phase=(2*np.pi*geometric[:,:,None]*frequency[None,None,:]
        -2*np.pi*e['delays_s'][None,:,None]*(frequency-e['reference_hz'])[None,None,:])
    return np.sum(e['alpha'][None,:,None]*np.exp(1j*phase),axis=1)


def make_drive(environment, carrier_hz, pilots, payload, rng, cfg):
    coeff=transmit_coefficients(pilots,payload,cfg)
    channel=channel_at_offsets(environment,cfg,carrier_hz)
    power_w=1e-3*10**(float(environment['power_dbm'])/10)
    clean=np.sqrt(power_w)*channel*coeff[None,:]
    # 复包络每个频格方差kTΔf；不对实现后的噪声功率做强制归一化。
    antenna_noise=np.sqrt(KB*cfg.temperature_k*cfg.df_hz)*cn(rng,clean.shape)
    bb=clean+antenna_noise
    carrier_bin=int(round(carrier_hz/cfg.df_hz))
    if abs(carrier_bin*cfg.df_hz-carrier_hz)>1e-3:
        raise ValueError('载频必须准确落在公共DFT频格。')
    bins=carrier_bin+cfg.band_offsets
    if bins.min()<=0 or bins.max()>=cfg.sample_count//2:
        raise ValueError('实RF采样率不满足本频带要求。')
    rf=np.zeros((cfg.n,cfg.sample_count//2+1),complex)
    # sqrt(2)实通带转换 × 1000/sqrt(2)前端放大，再取正频一半，合计500。
    rf[:,bins]=500*bb
    drive=np.fft.irfft(rf,n=cfg.sample_count,axis=-1)*cfg.sample_count
    return drive,dict(transmit_coefficients=coeff,antenna_clean_coefficients=clean,
        antenna_noise_coefficients=antenna_noise,channel_coefficients=channel,
        expected_antenna_noise_w=KB*cfg.temperature_k*len(bins)*cfg.df_hz,
        actual_discrete_bandwidth_hz=len(bins)*cfg.df_hz)
