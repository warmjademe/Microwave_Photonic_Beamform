"""可选的 64 路强度合并研究近似模型；不是 OSD 原生单载波探测的复现。

该类供明确选用 research_joint_v1 时使用。不会修改 native_receiver.py
或调用/绕过其端到端校准门禁。每 8 路后不做光电转换，只有最后一个 APD。
"""
from dataclasses import dataclass, asdict
import numpy as np
from .config import KB, cn
from .controls import project, codes
from .data import estimate_from_iq


@dataclass(frozen=True)
class ResearchDevice:
    mode: str = 'research_joint_v1'
    electrical_ports: int = 2
    lna_gain_db: float = 60.0
    lna_noise_figure_db: float = 2.0  # 研究假设，沿用旧模型；不是新测得值。
    impedance_ohm: float = 50.0  # 定义物理复 RMS 电流换算，不重定义 OSD 电功率计。
    laser_bias_a: float = .025  # OSD 工作点记录。
    laser_dc_w: float = .0036255712012  # 原生组件稳态参考，不是动态波形。
    laser_slope_w_a: float = .25  # 线性小信号假设；未校准全动态激光。
    laser_rin_db_hz: float = -144.0
    branch_loss_db: float = 3.0
    wdm_effective_loss_db: float = 0.0
    combiner_loss_db: float = 2.0
    apd_gain: float = 30.0
    responsivity_a_w: float = 1.0
    dark_current_a: float = 4e-9
    ionization_ratio: float = .9
    thermal_current_psd_a2_hz: float = 3.98e-21
    electron_charge_c: float = 1.60217733e-19

    def to_dict(self):
        return asdict(self)


class ResearchReceiver:
    def __init__(self, cfg, channel, power_dbm, carrier_hz, device=None):
        self.cfg = cfg
        self.device = device or ResearchDevice()
        self.h = np.asarray(channel, complex)
        self.power_dbm = float(power_dbm)
        self.carrier_hz = float(carrier_hz)
        if self.h.shape != (cfg.n, cfg.tones) or not np.all(np.isfinite(self.h)):
            raise ValueError('信道应为有限复数 [64,64]。')
        self.freq = carrier_hz+cfg.offsets_hz
        self.power_per_tone = 1e-3*10**(self.power_dbm/10)/cfg.tones
        d = self.device
        self.phase = np.exp(-2j*np.pi*np.arange(cfg.delay_levels+1)[:, None]
                            *(cfg.delay_max_ps/cfg.delay_levels)*1e-12*self.freq)
        fixed_loss = d.branch_loss_db+d.wdm_effective_loss_db+d.combiner_loss_db
        # 衰减器改变光功率的比例，同时也是 RF 光电流幅度的比例。
        self.transmissions = 10**(-(fixed_loss+np.arange(cfg.attenuation_levels+1)
                                  *cfg.attenuation_max_db/cfg.attenuation_levels)/10)
        convert = np.sqrt(10**(d.lna_gain_db/10)/d.impedance_ohm)*d.laser_slope_w_a*d.responsivity_a_w*d.apd_gain
        self.signal_factor = np.sqrt(self.power_per_tone/d.electrical_ports)*convert
        # 天线热噪声经过电合路器；LNA 附加噪声位于其后，避免重复除以2。
        front = convert**2*KB*cfg.temperature_k*(1/d.electrical_ports+10**(d.lna_noise_figure_db/10)-1)
        rin = 10**(d.laser_rin_db_hz/10)*(d.apd_gain*d.responsivity_a_w*d.laser_dc_w)**2
        excess = d.ionization_ratio*d.apd_gain+(1-d.ionization_ratio)*(2-1/d.apd_gain)
        shot_factor = 2*d.electron_charge_c*d.apd_gain**2*excess
        self.noise_coeff = dict(front=front, rin=rin, shot=shot_factor*d.responsivity_a_w*d.laser_dc_w,
                               dark=shot_factor*d.dark_current_a, thermal=d.thermal_current_psd_a2_hz)

    def state(self, control):
        u = project(control, self.cfg)
        c = codes(u, self.cfg)
        t = self.transmissions[c[self.cfg.n:]]
        branches = self.signal_factor*t[:, None]*self.h*self.phase[c[:self.cfg.n]]
        return dict(control=u, transmission=t, branches=branches, gain=branches.sum(axis=0),
                    sum_t=float(t.sum()), sum_t2=float(np.dot(t, t)))

    def noise_variance(self, sum_t, sum_t2):
        c = self.noise_coeff
        return ((c['front']+c['rin'])*np.asarray(sum_t2)+c['shot']*np.asarray(sum_t)
                +c['dark']+c['thermal'])*(self.cfg.bandwidth_hz/self.cfg.tones)

    @staticmethod
    def objective(gain, variance):
        v = np.asarray(variance)
        if v.ndim:
            v = v[..., None]
        return -np.mean(v/(abs(gain)**2+v), axis=-1)

    def evaluate(self, control):
        state = self.state(control)
        v = float(self.noise_variance(state['sum_t'], state['sum_t2']))
        s = abs(state['gain'])**2
        d = self.device
        noise_parts = dict(front_end=self.noise_coeff['front']*state['sum_t2'],
                           rin=self.noise_coeff['rin']*state['sum_t2'],
                           shot_and_dark=self.noise_coeff['shot']*state['sum_t']+self.noise_coeff['dark'],
                           thermal=self.noise_coeff['thermal'])
        return dict(gain_a=state['gain'], noise_per_tone_a2=np.full(self.cfg.tones, v),
                    objective=float(self.objective(state['gain'], v)),
                    snr_db=float(10*np.log10(max(float(s.sum()/(v*self.cfg.tones)), 1e-30))),
                    signal_current_power_a2=float(s.sum()), noise_current_power_a2=float(v*self.cfg.tones),
                    optical_dc_w=float(d.laser_dc_w*state['sum_t']),
                    apd_dc_a=float(d.apd_gain*(d.responsivity_a_w*d.laser_dc_w*state['sum_t']+d.dark_current_a)),
                    physical_operating_range_verified=False,
                    noise_psd_a2_hz=noise_parts)

    def scan_coordinate(self, state, index, values):
        """只替换一个支路贡献，批量精确计算候选；仅用于离线教师。"""
        cfg = self.cfg
        values = np.asarray(values, float)
        n = index % cfg.n
        if index < cfg.n:
            code = np.floor(np.clip(values, 0, 1)*cfg.delay_levels+.5).astype(int)
            values = code/cfg.delay_levels
            replacement = self.signal_factor*state['transmission'][n]*self.h[n]*self.phase[code]
            gain = state['gain']-state['branches'][n]+replacement
            v = self.noise_variance(state['sum_t'], state['sum_t2'])
        else:
            code = np.floor(np.clip(values, 0, 1)*cfg.attenuation_levels+.5).astype(int)
            values = code/cfg.attenuation_levels
            t = self.transmissions[code]
            old = state['transmission'][n]
            gain = state['gain']+state['branches'][n]*(t[:, None]/old-1)
            v = self.noise_variance(state['sum_t']-old+t, state['sum_t2']-old**2+t*t)
        scores = self.objective(gain, v)
        best = int(np.argmax(scores))
        u = state['control'].copy()
        u[index] = values[best]
        return u, float(scores[best]), len(values)

    def measure(self, control, pilots, rng, return_iq=False):
        e = self.evaluate(control)
        y = (e['gain_a'][:, None]*pilots+
             np.sqrt(e['noise_per_tone_a2'][:, None])*cn(rng, pilots.shape)).astype(np.complex64)
        _, _, score = estimate_from_iq(y, pilots)
        return (float(score), y) if return_iq else float(score)

    def trace_antenna(self, pilots, rng):
        """只用于审计：干净与含天线热噪声的64路输入。"""
        clean = np.sqrt(self.power_per_tone)*self.h[:, :, None]*pilots
        noise = np.sqrt(KB*self.cfg.temperature_k*self.cfg.bandwidth_hz/self.cfg.tones)*cn(rng, clean.shape)
        return clean, noise, clean+noise
