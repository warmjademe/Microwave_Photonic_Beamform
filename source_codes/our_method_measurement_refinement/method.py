"""确定性测量校正：输入仅为公开X、已知导频、估计响应及训练残差矩阵。"""
import numpy as np
from our_method_response_control.physics import design, pilot_gain, measurement_noise


def project_response(h, y, a, c, noise):
    """h为64×31；返回正则化最小二乘校正，不声称独立贝叶斯后验。"""
    ca=c[None] @ a.conj().transpose(0,2,1)
    matrix=a @ ca + noise
    matrix += (np.trace(matrix,axis1=1,axis2=2).real/16*1e-12)[:,None,None]*np.eye(16)
    residual=y-np.einsum('kpn,nk->kp',a,h)
    solved=np.linalg.solve(matrix,residual[...,None])[...,0]
    return h+np.einsum('knp,kp->nk',ca,solved)


def refine(x, pilots, response_estimate, covariance, mode):
    h=np.asarray(response_estimate,complex)
    if h.shape!=(64,31) or not np.isfinite(h).all():raise ValueError('响应形状或数值错误。')
    fc=int(x[1984]);m=design(fc)
    c=np.asarray(covariance[fc-4],complex)*max(float(np.mean(abs(h)**2)),1e-24)
    if mode=='isotropic':c=np.eye(64)*np.trace(c).real/64
    elif mode!='spatial':raise ValueError('未知的校正方式。')
    y=pilot_gain(x,pilots).T;a=m['sensing'];r=measurement_noise(m,x,pilots)
    r=r+(np.trace(r,axis1=1,axis2=2).real/16*1e-12)[:,None,None]*np.eye(16)
    result=project_response(h,y,a,c,r)
    before=y-np.einsum('kpn,nk->kp',a,h)
    after=y-np.einsum('kpn,nk->kp',a,result)
    def weighted(v):
        return float(np.real(np.sum(v.conj()*np.linalg.solve(r,v[...,None])[...,0])))
    old=weighted(before);new=weighted(after)
    if not np.isfinite(result).all() or new>old+1e-9*max(1,old):
        raise ValueError('噪声加权测量残差增大或结果非有限。')
    return result,dict(mode=mode,weighted_residual_before=old,weighted_residual_after=new,
        extra_feedback=0,definition='regularized projection of a measurement-dependent estimate')
