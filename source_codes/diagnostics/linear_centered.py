"""候选加速：同一RK4速率方程的局部线性响应，含有限帧初态瞬变。

与早先失败的近似不同：使用精确半步电输入、统一时间参考和实际初态。
输出是光强度的RF系数，不对未居中的有限光谱泄漏进行拟合。
这是待验证的数值近似，不直接替换非线性器件或正式数据。
"""
from functools import lru_cache
import numpy as np
from native_sim.config import NativeConfig
from native_sim.laser import load_profile
from native_sim.waveforms import transmit_coefficients,channel_at_offsets
from baseline_common.config import cn,KB
from carrier_reference import bias_equilibrium


def route_transfer(route,frequencies_hz,fs):
    p=route['physical'];eq=bias_equilibrium(route)
    s=eq['photons_cm3'];n=eq['carriers_cm3']
    v=p['active_volume_cm3'];eps=p['gain_compression_cm3']
    nt=p['transparency_density_cm_neg3'];beta=p['spontaneous_emission_factor']
    g=p['group_velocity_cm_s']*p['differential_gain_cm2']
    gamma=p['mode_confinement_factor'];tn=p['carrier_lifetime_s'];tp=p['photon_lifetime_s']
    # 状态缩放与C++一致：载流子/1e18、光子/1e15；时间单位ns。
    dn=g*s/(1+eps*s);ds=g*(n-nt)/(1+eps*s)**2
    jac=np.array([[-1/tn-dn,-ds*1e-3],
                  [(gamma*dn+gamma*beta/tn)*1e3,gamma*ds-1/tp]])*1e-9
    drive=np.array([1/(1.602176634e-19*v)*1e-27,0.])
    dt=1e9/fs
    def step(y,u0,um,u1):
        k1=jac@y+drive*u0
        k2=jac@(y+dt*k1/2)+drive*um
        k3=jac@(y+dt*k2/2)+drive*um
        k4=jac@(y+dt*k3)+drive*u1
        return y+dt*(k1+2*k2+2*k3+k4)/6
    matrix=np.column_stack([step(np.eye(2)[:,k],0.,0.,0.) for k in range(2)])
    b0=step(np.zeros(2),1.,0.,0.)
    bm=step(np.zeros(2),0.,1.,0.)
    b1=step(np.zeros(2),0.,0.,1.)
    eig,vectors=np.linalg.eig(matrix)
    if np.max(abs(eig))>=1:raise ValueError('候选离散线性核不稳定。')
    frequency=np.asarray(frequencies_hz,float)
    z=np.exp(2j*np.pi*frequency/fs)
    half=np.exp(1j*np.pi*frequency/fs)
    forcing=b0[None,:]+half[:,None]*bm[None,:]+z[:,None]*b1[None,:]
    response=np.linalg.solve(z[:,None,None]*np.eye(2)-matrix[None,:,:],forcing[...,None])[...,0]
    power_per_scaled_photon=eq['optical_dc_w']/(s/1e15)
    return dict(response=response,matrix=matrix,eigenvalues=eig,eigenvectors=vectors,
                inverse_eigenvectors=np.linalg.inv(vectors),equilibrium=eq,
                power_per_scaled_photon=power_per_scaled_photon,frequency_hz=frequency,z=z,
                state_bias=np.array([n/1e18,s/1e15]),current_jacobian=jac,current_input=drive)


@lru_cache(maxsize=20)
def transfers(carrier_hz,factor=2):
    cfg=NativeConfig()
    frequency=carrier_hz+cfg.band_offsets*cfg.df_hz
    return [route_transfer(r,frequency,cfg.sample_rate_hz*factor) for r in load_profile()]


def from_positive_coefficients(coefficients,carrier_hz,cfg,factor=2,include_transient=True):
    """coefficients 为64路实电驱动的正频系数，不是RF功率。"""
    coefficients=np.asarray(coefficients,complex)
    if coefficients.shape!=(64,255) or not np.all(np.isfinite(coefficients)):
        raise ValueError('需64×255有限正频系数。')
    profile=load_profile();models=transfers(float(carrier_hz),factor)
    count=cfg.sample_count*factor
    band=np.empty_like(coefficients);dc=np.empty(64)
    for k,(route,model) in enumerate(zip(profile,models)):
        current=route['modulation_peak_current_a']*coefficients[k]
        particular=model['response']*current[:,None]
        state_band=particular.copy()
        state_mean=model['state_bias'].copy()
        if include_transient:
            initial_current=route['bias_current_a']+2*current.real.sum()
            if initial_current<=0:raise ValueError('初态电流非正。')
            initial=bias_equilibrium({**route,'bias_current_a':initial_current})
            initial_state=np.array([initial['carriers_cm3']/1e18,initial['photons_cm3']/1e15])
            mismatch=initial_state-model['state_bias']-2*particular.real.sum(axis=0)
            beta=model['inverse_eigenvectors']@mismatch
            eigen=model['eigenvalues'];z=model['z']
            ratio=eigen[None,:]/z[:,None]
            # 原始步末样本的有限几何级数，再按已知dt对齐到共同时间。
            amplitude=(eigen[None,:]/(count*(z[:,None]-eigen[None,:]))
                       *(1-ratio**count))*beta[None,:]
            state_band+=amplitude@model['eigenvectors'].T
            mean_coeff=(eigen*(1-eigen**count)/(count*(1-eigen)))*beta
            state_mean+=np.real(model['eigenvectors']@mean_coeff)
        band[k]=state_band[:,1]*model['power_per_scaled_photon']
        dc[k]=state_mean[1]*model['power_per_scaled_photon']
    return band,dc


def approximate_cache(environment,carrier_hz,pilots,payload,rng,cfg,include_transient=True):
    coeff=transmit_coefficients(pilots,payload,cfg)
    channel=channel_at_offsets(environment,cfg,carrier_hz)
    power=1e-3*10**(float(environment['power_dbm'])/10)
    clean=np.sqrt(power)*channel*coeff[None,:]
    noise=np.sqrt(KB*cfg.temperature_k*cfg.df_hz)*cn(rng,clean.shape)
    return from_positive_coefficients(500*(clean+noise),carrier_hz,cfg,
                                      include_transient=include_transient)
