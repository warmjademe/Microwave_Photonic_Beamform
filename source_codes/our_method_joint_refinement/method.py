"""联合31个频率的正则化校正；输入仅为公开测量和训练统计。"""
from functools import lru_cache
import numpy as np
from our_method_response_control.physics import design,pilot_gain,measurement_noise
from native_sim.control_engine import BAND_OFFSETS
from baseline_common.config import KB


@lru_cache(maxsize=17)
def noise_basis(fc,pilot_bytes):
    m=design(fc); pilots=np.frombuffer(pilot_bytes,np.complex128).reshape(31,2)
    extract=m['extract']
    average=(extract[0]*pilots[:,0].conj()[:,None]+extract[1]*pilots[:,1].conj()[:,None])/2
    frequency=fc*1e9+BAND_OFFSETS*m['cfg'].df_hz
    delays=np.rint(m['probes'][:,:64]*76)/m['cfg'].sample_rate_hz
    branches=(m['transmission'][0]*np.exp(-2j*np.pi*delays[:,:,None]*frequency[None,None,:])
              *m['gain'][None,:,:])
    # 每个Fourier噪声源的探测间相关性；同一天线噪声被16套控制共同观测。
    spatial=np.einsum('pnb,qnb->bpq',branches,branches.conj())
    spectral=average[:,None,:]*average.conj()[None,:,:]
    common=(KB*m['cfg'].temperature_k*m['cfg'].df_hz
            *np.einsum('klb,bpq->kplq',spectral,spatial,optimize=True)).reshape(496,496)
    return common,average@average.conj().T,np.sum(abs(extract[0])**2,axis=-1)


def full_noise(x,pilots):
    fc=int(x[1984]); common,apd_shape,row_norm=noise_basis(fc,np.asarray(pilots,np.complex128).tobytes())
    # X存储APD经过第一块导频提取后的方差，反推出每个探测的Fourier噪声方差。
    per_probe=np.mean(x[2001:2497].reshape(16,31)/row_norm[None,:],axis=1)
    apd=np.einsum('kl,pq,p->kplq',apd_shape,np.eye(16),per_probe).reshape(496,496)
    result=common+apd
    return (result+result.conj().T)/2


def refine_joint(x,pilots,response_estimate,spatial_covariance,frequency_covariance,
                 correlated_noise=True,diagnostics=False):
    h=np.asarray(response_estimate,complex); fc=int(x[1984]); a=design(fc)['sensing']
    if h.shape!=(64,31) or not np.isfinite(h).all(): raise ValueError('响应格式错误。')
    scale=max(float(np.mean(abs(h)**2)),1e-24)
    spatial=np.asarray(spatial_covariance[fc-4],complex)*scale
    frequency=np.asarray(frequency_covariance[fc-4],complex)
    if spatial.shape!=(64,64) or frequency.shape!=(31,31): raise ValueError('训练统计格式错误。')
    if correlated_noise:
        noise=full_noise(x,pilots)
    else:
        blocks=measurement_noise(design(fc),x,pilots)
        noise=np.zeros((31,16,31,16),complex)
        noise[np.arange(31),:,np.arange(31),:]=blocks
        noise=noise.reshape(496,496)
    # 与旧逐频率版本相同的噪声相对加载；没有用真实信道计算此矩阵。
    noise=noise.copy()
    noise4=noise.reshape(31,16,31,16)
    block_trace=np.asarray([np.trace(noise4[k,:,k,:]).real/16 for k in range(31)])
    noise.flat[::497]+=np.repeat(block_trace*1e-12,16)
    af=a.reshape(496,64)
    covariance_y=((af@spatial@af.conj().T).reshape(31,16,31,16)
                  *frequency[:,None,:,None]).reshape(496,496)
    system=covariance_y+noise
    diagonal=system.diagonal().real.reshape(31,16).mean(1)*1e-12
    system.flat[::497]+=np.repeat(diagonal,16)
    before=(pilot_gain(x,pilots).T-np.einsum('kpn,nk->kp',a,h)).reshape(496)
    z=np.linalg.solve(system,before)
    lifted=np.einsum('kpn,kp->nk',a.conj(),z.reshape(31,16))
    result=h+spatial@lifted@frequency.T
    if not np.isfinite(result).all(): raise ValueError('联合校正出现非有限值。')
    detail=dict(extra_feedback=0,correlated_noise=correlated_noise)
    if diagnostics:
        after=(pilot_gain(x,pilots).T-np.einsum('kpn,nk->kp',a,result)).reshape(496)
        weighted=lambda v: float(np.real(np.vdot(v,np.linalg.solve(noise,v))))
        old=weighted(before); new=weighted(after)
        if new>old+1e-8*max(1,old): raise ValueError('加权测量残差增加。')
        detail.update(weighted_before=old,weighted_after=new,
            solve_relative_residual=float(np.linalg.norm(system@z-before)/max(np.linalg.norm(before),1e-30)))
    return result,detail
