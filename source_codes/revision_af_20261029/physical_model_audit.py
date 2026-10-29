"""从冻结参数推导物理范围及噪声定义；不将这些计算作为硬件测量。"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'diagnostics'))
import numpy as np
from study_full_baselines.common import require_host,sha256,verify_sources,write_json
from native_sim.config import NativeConfig
from native_sim.laser import load_profile
from native_sim.control_engine import NativeControlEngine,DEMODULATOR_GAIN,APD_GAIN,RESPONSIVITY_A_PER_W,THERMAL_PSD_A2_HZ,ELECTRON_CHARGE_C,DARK_CURRENT_A,IONIZATION_RATIO
from carrier_reference import bias_equilibrium


def run(project):
    require_host();source=project/'source_codes'
    protocol=json.loads((project/'dataset_simulation/revision_rq2_af_20261029/full/protocol.json').read_text())
    verify_sources(protocol['source_sha256'])
    out=project/'dataset_simulation/revision_af_20261029/P/static'
    if out.exists():raise RuntimeError('Existing physical audit must be inspected, not overwritten')
    out.mkdir(parents=True)
    cfg=NativeConfig();profile=load_profile();eq=[bias_equilibrium(r) for r in profile]
    frequencies=np.asarray([r['optical_frequency_hz'] for r in profile])
    effective=np.asarray([r['optical_frequency_hz'] for r in eq])
    assert len(frequencies)==64 and len(np.unique(frequencies))==64
    pitch=cfg.pitch_m;wavelength=lambda f:299792458./(f*1e9)
    geometry=[]
    for carrier in range(4,21):
        ratio=pitch/wavelength(carrier);aliases=[]
        for m in range(-4,5):
            for n in range(-4,5):
                if m==n==0:continue
                u,v=m/ratio,n/ratio
                if u*u+v*v<=1+1e-12:aliases.append([m,n,u,v])
        geometry.append(dict(carrier_ghz=carrier,pitch_wavelengths=ratio,
            instantaneous_fractional_bandwidth=cfg.bandwidth_hz/(carrier*1e9),
            broadside_isotropic_visible_aliases=aliases))
    powers=np.asarray([r['optical_dc_w'] for r in eq])
    engine=NativeControlEngine(cfg,np.zeros((64,255),complex),powers,12e9,np.ones((31,2),complex))
    excess=IONIZATION_RATIO*APD_GAIN+(1-IONIZATION_RATIO)*(2-1/APD_GAIN)
    cases=[]
    for attenuation_code in [0,12,24]:
        transmission=10**(-(5+.5*attenuation_code)/10)
        optical=float(transmission*powers.sum())
        psd,variance=engine._noise(optical)
        analytic=THERMAL_PSD_A2_HZ+2*ELECTRON_CHARGE_C*APD_GAIN**2*excess*(RESPONSIVITY_A_PER_W*optical+DARK_CURRENT_A)
        np.testing.assert_allclose(psd,analytic,rtol=1e-14,atol=0)
        np.testing.assert_allclose(variance,DEMODULATOR_GAIN**2*cfg.df_hz*analytic,rtol=1e-14,atol=0)
        cases.append(dict(common_attenuation_code=attenuation_code,optical_dc_w=optical,
            stationary_apd_psd_a2_per_hz=float(psd),complex_coefficient_variance=float(variance)))
    observation_duration=2*(cfg.useful_samples+cfg.cp_samples)/cfg.iq_sample_rate_hz
    result=dict(at=datetime.now(timezone.utc).isoformat(),scope='derived from frozen software; no hardware measurement',
        source_sha256=protocol['source_sha256'],audit_code_sha256=sha256(Path(__file__)),
        optical=dict(channels=64,frequencies_hz=frequencies.tolist(),effective_centers_hz=effective.tolist(),
            minimum_nominal_separation_hz=float(np.diff(np.sort(frequencies)).min()),
            minimum_equilibrium_separation_hz=float(np.diff(np.sort(effective)).min()),
            first_order_sideband_crossbeat_lower_bound_hz=float(np.diff(np.sort(effective)).min()-2*(20e9+cfg.bandwidth_hz/2)),
            crossbeat_scope='carrier and first-order modulation sidebands only; excludes high-order nonlinear overlap and linewidth tails'),
        array=dict(pitch_m=pitch,aperture_one_axis_m=7*pitch,geometry=geometry,
            alias_scope='ideal isotropic elements, broadside, front visible hemisphere; not a causal attribution of measured BER'),
        delay=dict(step_s=1/cfg.sample_rate_hz,maximum_s=cfg.delay_max_ps*1e-12,
            maximum_model_multipath_s=200e-9,cyclic_prefix_s=cfg.cp_s,
            interpretation='branch beam control does not cancel all path delays; OFDM equalization is separate'),
        scale=dict(electrical_coefficient_multiplier=500.,multiplication_order='500*(sqrt(P)*H*C + thermal_noise); modulation_peak_current converts drive to current',
            hardware_gain_or_noise_figure_measured=False,demodulation_gain=DEMODULATOR_GAIN,
            demodulation_convention='complex RF envelope is twice the retained positive-frequency coefficient'),
        noise=dict(apd_gain=APD_GAIN,excess_factor=excess,responsivity=RESPONSIVITY_A_PER_W,
            thermal_psd=THERMAL_PSD_A2_HZ,dark_current=DARK_CURRENT_A,formula_checks=cases,
            antenna_noise='propagated through per-branch laser transfer; not added again as detector noise',
            rin_in_main_model=False,explicit_optical_crossbeats_in_main_model=False),
        measurement_time=dict(two_pilots_s=observation_duration,full_simulation_frame_s=cfg.iq_samples/cfg.iq_sample_rate_hz,
            configured_switch_s=cfg.switch_s,configured_switch_measured=False,
            total_16_with_configured_switch_s=16*(observation_duration+cfg.switch_s),
            total_64_with_configured_switch_s=64*(observation_duration+cfg.switch_s),
            scope='sequential pilot acquisition lower-level cost assumption; software and transfer time additional'))
    write_json(out/'audit.json',result)
    text='''# 固定接收模型的物理范围核查

本报告由冻结软件参数推导，数值在 audit.json；不是实测器件校准或新接收实验。

- 64 个激光光载频均不同，标称从 189.8 到 196.1 THz，相邻间隔 100 GHz。偏置平衡频移也逐路计算。只考虑载波和一阶 20 GHz 调制边带时，跨光载波差频的最低位置仍在当前接收带外；这不能证明高阶非线性光谱、线宽尾部和实际滤波均可忽略。主模型没有显式求这些光场交叉项。
- 阵元间距为 4 GHz 的半波长，在 20 GHz 为 2.5 倍波长。报告列出理想各向同性阵元、正侧向波束的可见空间周期峰位置。这是阵列几何约束，不能单独归因为高频 BER 变差；激光响应、噪声、控制量化和多径也随条件变化。
- 最大光延时是 1.484375 ns，而统计多径最大时延为 200 ns。延时器整合各路复合响应，不逐条消除全部多径时延。当前循环前缀为 300 ns，接收均衡与光子合并分别承担作用。
- 输入系数 500 同时乘到期望信号和天线热噪声，再乘激光调制电流尺度。它是固定的软件驱动尺度；没有附带实际放大器的阻抗、噪声系数、饱和功率或实测链路预算。不能将它单独解释为已经实现的低噪声硬件增益。
- 解调系数 2 将保留的实射频正频系数转换为所定义的复包络幅值，并同样用于接收噪声方差。它不是在计算 SNR 时只放大信号。
- 主模型 APD 噪声为热噪声加依赖总光功率的散粒和暗电流项。三个衰减工作点的公式与执行器数值已核对。独立激光 RIN 的接收 PSD 应与各路光功率平方之和相关；它与总光功率平方对应的完全相关噪声假设不同，后续敏感性分析须明确选择哪一种。
- 一次两导频观测为 1.24 微秒，完整三块仿真帧为 2.56 微秒；两者不能混用。配置中 1 微秒的切换时间是计算假设，不是测量值。总在线成本还应增加网络、求解器、选择管理和实际硬件传输时间。
'''
    (out/'REPORT_zh.md').write_text(text)
    verify_sources(protocol['source_sha256'])
    write_json(out/'complete.json',dict(status='complete',at=result['at'],type='static parameter and formula audit',
        exclusive_gpu_latency_measurement_complete=False,rin_sensitivity_complete=False,
        file_sha256={n:sha256(out/n) for n in ['audit.json','REPORT_zh.md']}))
    print(json.dumps(dict(optical_min_separation_GHz=result['optical']['minimum_equilibrium_separation_hz']/1e9,
        first_order_crossbeat_bound_GHz=result['optical']['first_order_sideband_crossbeat_lower_bound_hz']/1e9,
        measurement_time=result['measurement_time'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
