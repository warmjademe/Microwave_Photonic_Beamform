"""候选计算后端：仅把非线性分支的大FFT放到华硕GPU，保留CPU速率方程。

本模块不替换任何冻结文件。未通过配对数值门限前不得接入正式生成。
双精度FFT仍可能有舍入差异，不能把这一后端称为逐位等价。
"""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_sim.laser import load_profile
from native_sim.waveforms import transmit_coefficients,channel_at_offsets
from baseline_common.config import cn,KB
from linear_centered import transfers,from_positive_coefficients
from hybrid_centered import nonlinear_solver,PHOTON_RELATIVE_BOUND,CARRIER_RELATIVE_BOUND


def stage_samples_cuda(positive_coefficients,carrier_hz,cfg,factor):
    if not isinstance(factor,int) or factor<1:
        raise ValueError('积分倍数须为正整数。')
    values=np.asarray(positive_coefficients,complex)
    if values.shape[-1]!=len(cfg.band_offsets) or not np.isfinite(values).all():
        raise ValueError('正频输入需为255个有限复数。')
    n=cfg.sample_count*factor*2
    bins=int(round(carrier_hz/cfg.df_hz))+cfg.band_offsets
    if (not np.isclose(round(carrier_hz/cfg.df_hz)*cfg.df_hz,carrier_hz,rtol=0,atol=1e-3)
            or bins.min()<=0 or bins.max()>=n//2):
        raise ValueError('载频必须落在统一FFT格点及采样范围内。')
    spectrum=torch.zeros(values.shape[:-1]+(n//2+1,),device='cuda',dtype=torch.complex128)
    indices=torch.as_tensor(bins,device='cuda',dtype=torch.int64)
    spectrum[...,indices]=torch.as_tensor(values,device='cuda',dtype=torch.complex128)
    samples=(torch.fft.irfft(spectrum,n=n,dim=-1)*n).cpu().numpy()
    return np.concatenate([samples,samples[...,:1]],axis=-1)


def hybrid_from_coefficients_cuda(coefficients,carrier_hz,cfg):
    coefficients=np.asarray(coefficients,complex)
    profile=load_profile();models=transfers(float(carrier_hz),2)
    band,dc=from_positive_coefficients(coefficients,carrier_hz,cfg,include_transient=True)
    bounds=np.asarray([2*np.sum(abs(m['response']*(p['modulation_peak_current_a']*c)[:,None]),axis=0)
        /m['state_bias'] for m,p,c in zip(models,profile,coefficients)])
    selected=np.flatnonzero((bounds[:,0]>CARRIER_RELATIVE_BOUND)|(bounds[:,1]>PHOTON_RELATIVE_BOUND))
    factor=2;fs=cfg.sample_rate_hz*factor
    bins=int(round(carrier_hz/cfg.df_hz))+cfg.band_offsets
    align=np.exp(-2j*np.pi*(carrier_hz+cfg.band_offsets*cfg.df_hz)/fs)
    if len(selected):
        solver=nonlinear_solver()
        indices_gpu=torch.as_tensor(bins,device='cuda',dtype=torch.int64)
        for start in range(0,len(selected),8):
            indices=selected[start:start+8]
            stage=stage_samples_cuda(coefficients[indices],carrier_hz,cfg,factor)
            field,_=solver(stage,fs,[profile[i] for i in indices])
            # 沿用原CPU绝对值和平方；只替换FFT实现及执行位置。
            intensity=abs(field)**2
            spectrum=torch.fft.rfft(torch.as_tensor(intensity,device='cuda',dtype=torch.float64),dim=-1)/intensity.shape[-1]
            band[indices]=spectrum[:,indices_gpu].cpu().numpy()*align[None,:]
            dc[indices]=spectrum[:,0].real.cpu().numpy()
    return band,dc,dict(nonlinear_routes=int(len(selected)),selected_routes=selected.tolist(),
        state_bound_max=bounds.max(0).tolist(),
        thresholds=dict(carrier=CARRIER_RELATIVE_BOUND,photon=PHOTON_RELATIVE_BOUND))


def approximate_cache_cuda(environment,carrier_hz,pilots,payload,rng,cfg,return_details=False):
    coeff=transmit_coefficients(pilots,payload,cfg)
    channel=channel_at_offsets(environment,cfg,carrier_hz)
    power=1e-3*10**(float(environment['power_dbm'])/10)
    clean=np.sqrt(power)*channel*coeff[None,:]
    noise=np.sqrt(KB*cfg.temperature_k*cfg.df_hz)*cn(rng,clean.shape)
    band,dc,details=hybrid_from_coefficients_cuda(500*(clean+noise),carrier_hz,cfg)
    return (band,dc,details) if return_details else (band,dc)
