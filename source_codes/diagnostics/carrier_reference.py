"""显式记录激光器偏置平衡态的频率参考，避免把静态光载波截窗泄漏当RF。

E(t) 改写为 E'(t)=E(t)exp(-j omega_bias t)，同时把光载波中心改为
nu'=nu+omega_bias/(2pi)。物理光场与瞬时强度均未改变。
有限光谱窗的误差须另行检查，不能仅凭此恒等式认定通过。
"""
import numpy as np


def bias_equilibrium(route):
    p=route['physical']
    q=1.602176634e-19
    h=6.62607015e-34
    v=p['active_volume_cm3']; eta=p['quantum_efficiency']
    beta=p['spontaneous_emission_factor']; eps=p['gain_compression_cm3']
    nt=p['transparency_density_cm_neg3']
    gain=p['group_velocity_cm_s']*p['differential_gain_cm2']
    gamma=p['mode_confinement_factor']; tn=p['carrier_lifetime_s']
    tp=p['photon_lifetime_s']; alpha=p['linewidth_enhancement_factor']
    low,high=0.,1e18
    for _ in range(100):
        photons=(low+high)/2
        a=gain*photons/(1+eps*photons)
        carriers=(a*nt+photons/(gamma*tp))/(a+beta/tn)
        current=q*v*(carriers/tn+gain*(carriers-nt)*photons/(1+eps*photons))
        if current>route['bias_current_a']: high=photons
        else: low=photons
    power=v*eta*h*route['optical_frequency_hz']/(2*gamma*tp)*photons
    omega=alpha/2*(gamma*gain*(carriers-nt)-1/tp)
    return dict(optical_dc_w=power,angular_frequency_offset=omega,
                optical_frequency_hz=route['optical_frequency_hz']+omega/(2*np.pi),
                carriers_cm3=carriers,photons_cm3=photons)


def center_field(fields,profile,fs):
    if fields.shape[0]!=len(profile):raise ValueError('逐路参数数量不同。')
    time=(np.arange(fields.shape[-1])+1)/fs
    offsets=np.array([bias_equilibrium(r)['angular_frequency_offset'] for r in profile])
    return fields*np.exp(-1j*offsets[:,None]*time[None,:])
