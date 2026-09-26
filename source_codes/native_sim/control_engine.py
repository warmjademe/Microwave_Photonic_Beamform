"""冻结原生激光输出的缓存控制引擎；每条记录的激光积分在外部只做一次。

输入是64路无光损光场的非循环强度自相关，不是静态信道H(f)。本文件
只施加逐路有效时延、光功率衰减和最后一个APD的带内响应。APD噪声明确
采用平均总光功率决定的stationary Gaussian近似，不声称复现原生瞬时噪声。
教师只使用前两块公开导频；第三块用于调用方独立评价，绝不用于选控制。
"""
import numpy as np


OPTICAL_SAMPLE_RATE_HZ = 51.2e9
BASEBAND_SAMPLE_RATE_HZ = 200e6
BASEBAND_COUNT = 512
FREQUENCY_STEP_HZ = BASEBAND_SAMPLE_RATE_HZ / BASEBAND_COUNT
BAND_OFFSETS = np.arange(-127, 128, dtype=np.int64)
ACTIVE_TONES = np.arange(-15, 16, dtype=np.int64)
USEFUL_STARTS = (60, 184, 308)
USEFUL_COUNT = 64
APD_GAIN = 30.0
DEMODULATOR_GAIN = 2.0
RESPONSIVITY_A_PER_W = 1.0
ELECTRON_CHARGE_C = 1.60217733e-19
DARK_CURRENT_A = 4e-9
THERMAL_PSD_A2_HZ = 3.98e-21
IONIZATION_RATIO = 0.9
FIXED_LOSS_DB = 5.0


def extraction_matrix():
    """A[b,t,k]把255个基带Fourier系数转换为第b块第t个OFDM符号。

    IQ[n]=sum_k c[k] exp(j2pi*k*n/512)。每块只取64个有效样点，按
    FFT/64归一化。因此独立系数噪声Var=v时，符号噪声方差是
    v*sum_k|A[b,t,k]|²；不能以总IQ噪声功率除以31替代。
    """
    local = np.arange(USEFUL_COUNT)
    wave = np.exp(2j*np.pi*BAND_OFFSETS[:, None]*local[None, :]/BASEBAND_COUNT)
    demod = np.exp(-2j*np.pi*ACTIVE_TONES[:, None]*local[None, :]/USEFUL_COUNT)/USEFUL_COUNT
    base = demod @ wave.T
    return np.stack([base*np.exp(2j*np.pi*BAND_OFFSETS*start/BASEBAND_COUNT)[None, :]
                     for start in USEFUL_STARTS])


def _finite(value, shape, name, dtype):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(name+'形状错误或包含非有限数值。')
    return result


class NativeControlEngine:
    """同一激光记录的所有候选控制共用branch_band_w和branch_dc_w。

    control前64维为0..76有效时延码/76，后64维为0..24衰减码/24。
    可返回第三块符号和完整IQ供独立评价；教师内部始终只读取前两块。
    """
    def __init__(self, cfg, branch_band_w, branch_dc_w, carrier_hz, pilot_qpsk):
        self.cfg = cfg
        if (cfg.n, cfg.tones, cfg.delay_levels, cfg.attenuation_levels) != (64, 31, 76, 24):
            raise ValueError('该固定协议需要64路、31子载波、76时延级和24衰减级。')
        if not np.isclose(cfg.delay_max_ps,76/OPTICAL_SAMPLE_RATE_HZ*1e12,rtol=0,atol=1e-9):
            raise ValueError('最大时延必须对应76个真实光包络采样间隔。')
        if cfg.attenuation_max_db != 12.0:
            raise ValueError('衰减量程固定0..12dB，每码0.5dB。')
        for key,wanted in [('sample_rate_hz',OPTICAL_SAMPLE_RATE_HZ),
                           ('sample_count',131072),
                           ('iq_sample_rate_hz',BASEBAND_SAMPLE_RATE_HZ),
                           ('iq_samples',BASEBAND_COUNT),
                           ('iq_sample_count',BASEBAND_COUNT),
                           ('bandwidth_hz',100e6),('useful_samples',64),('cp_samples',60),
                           ('probes',16),('pilot_symbols',2)]:
            if hasattr(cfg,key) and getattr(cfg,key)!=wanted:
                raise ValueError('配置不符合冻结网格：'+key)
        self.carrier_hz = float(carrier_hz)
        if not np.isfinite(self.carrier_hz):raise ValueError('载频必须为有限值。')
        carrier_bin = int(round(self.carrier_hz/FREQUENCY_STEP_HZ))
        if (not np.isclose(carrier_bin*FREQUENCY_STEP_HZ,self.carrier_hz,rtol=0,atol=1e-4) or
                self.carrier_hz < 4e9 or self.carrier_hz > 20e9):
            raise ValueError('载频需要位于4..20GHz及共同DFT网格。')
        self.branch_band_w = _finite(branch_band_w,(64,255),'逐路光强度系数',np.complex128).copy()
        self.branch_dc_w = _finite(branch_dc_w,(64,),'逐路平均光功率',np.float64).copy()
        if np.any(self.branch_dc_w<0):raise ValueError('平均光功率不能为负。')
        self.pilots = _finite(pilot_qpsk,(31,2),'公共导频',np.complex128).copy()
        if not np.allclose(abs(self.pilots),1.,rtol=0,atol=1e-7):
            raise ValueError('公共QPSK导频必须为单位模。')
        self.extract = extraction_matrix()
        self.symbol_noise_row_norm = np.sum(abs(self.extract)**2,axis=-1)
        self.rf_frequency_hz = self.carrier_hz+BAND_OFFSETS*FREQUENCY_STEP_HZ
        self.phase = np.exp(-2j*np.pi*np.arange(77)[:,None]/OPTICAL_SAMPLE_RATE_HZ
                            *self.rf_frequency_hz[None,:])
        self.transmissions = 10**(-(FIXED_LOSS_DB+np.arange(25)*.5)/10)
        # 光功率衰减直接缩放RF电流幅度；这里不再开平方。
        self.current_factor = DEMODULATOR_GAIN*APD_GAIN*RESPONSIVITY_A_PER_W
        phased = self.branch_band_w[:,None,:]*self.phase[None,:,:]
        self.symbol_table = (phased.reshape(-1,255) @ self.extract.reshape(93,255).T
                             ).reshape(64,77,3,31)*self.current_factor
        self.excess_noise_factor = IONIZATION_RATIO*APD_GAIN+(1-IONIZATION_RATIO)*(2-1/APD_GAIN)
        self.shot_factor = 2*ELECTRON_CHARGE_C*APD_GAIN**2*self.excess_noise_factor
        self.noise_model = 'stationary Gaussian APD using mean optical power; antenna noise already in laser output'
        self.objective_definition = 'negative pilot2 expected NMSE; pilot1 mean-output MMSE gain; payload excluded'
        for a in (self.branch_band_w,self.branch_dc_w,self.pilots,self.extract,
                  self.symbol_noise_row_norm,self.phase,self.transmissions,self.symbol_table):
            a.setflags(write=False)

    def project(self, control):
        u = _finite(control,(128,),'控制',np.float64)
        levels = np.r_[np.full(64,76),np.full(64,24)]
        return np.floor(np.clip(u,0,1)*levels+.5)/levels

    def codes(self, control):
        u = self.project(control)
        return np.rint(u*np.r_[np.full(64,76),np.full(64,24)]).astype(np.int16)

    def _noise(self, optical_dc_w):
        # 噪声根据合光后的平均光功率变化；不能对所有控制固定同一个散粒噪声。
        psd = THERMAL_PSD_A2_HZ+self.shot_factor*(RESPONSIVITY_A_PER_W*np.asarray(optical_dc_w)+DARK_CURRENT_A)
        coefficient_variance = DEMODULATOR_GAIN**2*FREQUENCY_STEP_HZ*psd
        return psd,coefficient_variance

    def _pilot_values(self, symbols, optical_dc_w):
        """最后两维为[块,31]，允许批量候选；不访问第3块。"""
        _,var = self._noise(optical_dc_w)
        noise1 = np.asarray(var)[...,None]*self.symbol_noise_row_norm[0]
        noise2 = np.asarray(var)[...,None]*self.symbol_noise_row_norm[1]
        gain = symbols[...,0,:]/self.pilots[:,0]
        w = np.conj(gain)/np.maximum(abs(gain)**2+noise1,np.finfo(float).tiny)
        residual = abs(w*symbols[...,1,:]-self.pilots[:,1])**2
        loss = np.mean(residual+abs(w)**2*noise2,axis=-1)
        return -loss,gain,w,noise1

    def state(self, control):
        u=self.project(control);code=self.codes(u)
        transmission=self.transmissions[code[64:]]
        branches=self.symbol_table[np.arange(64),code[:64]]*transmission[:,None,None]
        symbols=branches.sum(axis=0)
        return dict(control=u,codes=code,transmission=transmission,branches=branches,
                    symbols=symbols,optical_dc_w=float(np.dot(transmission,self.branch_dc_w)))

    def _coefficients(self, state):
        c=state['codes']
        return self.current_factor*np.sum(self.branch_band_w*self.phase[c[:64]]
                                          *state['transmission'][:,None],axis=0)

    def evaluate(self, control):
        s=self.state(control)
        score,gain,w,noise=self._pilot_values(s['symbols'],s['optical_dc_w'])
        psd,coefficient_variance=self._noise(s['optical_dc_w'])
        coeff=self._coefficients(s)
        return dict(objective=float(score),pilot_nmse=float(-score),
            snr_db=float(-10*np.log10(max(float(-score),1e-30))),
            snr_db_definition='-10*log10(pilot2 expected NMSE), not physical thermal SNR',
            gain_a=gain.copy(),noise_per_tone_a2=noise.copy(),pilot_equalizer=w.copy(),
            pilot_symbols_a=s['symbols'][:2].T.copy(),
            signal_current_power_a2=float(np.sum(abs(coeff)**2)),
            noise_current_power_a2=float(255*coefficient_variance),
            optical_dc_w=s['optical_dc_w'],
            apd_dc_a=float(APD_GAIN*(RESPONSIVITY_A_PER_W*s['optical_dc_w']+DARK_CURRENT_A)),
            apd_noise_psd_a2_hz=float(psd),noise_coefficient_variance_a2=float(coefficient_variance),
            noise_psd_a2_hz=dict(stationary_apd=float(psd)),
            noise_model=self.noise_model,objective_definition=self.objective_definition,
            teacher_uses_payload=False,physical_operating_range_verified=False)

    def scan_coordinate(self, state, index, values):
        """只替换一条贡献，批量扫描；自动包含当前码，避免坐标步变差。"""
        if not isinstance(index,(int,np.integer)) or not 0<=index<128:
            raise ValueError('坐标索引应在0..127。')
        v=np.asarray(values,float)
        if v.ndim!=1 or not len(v) or not np.all(np.isfinite(v)):
            raise ValueError('候选需要非空有限一维数组。')
        n=int(index)%64;levels=76 if index<64 else 24
        codes=np.unique(np.r_[np.floor(np.clip(v,0,1)*levels+.5).astype(int),state['codes'][index]])
        old=state['branches'][n]
        if index<64:
            replacement=self.symbol_table[n,codes]*state['transmission'][n]
            dc=np.full(len(codes),state['optical_dc_w'])
        else:
            trans=self.transmissions[codes]
            replacement=self.symbol_table[n,state['codes'][n]][None,:,:]*trans[:,None,None]
            dc=state['optical_dc_w']+(trans-state['transmission'][n])*self.branch_dc_w[n]
        symbols=state['symbols'][None,:,:]-old[None,:,:]+replacement
        scores=self._pilot_values(symbols,dc)[0]
        best=int(np.argmax(scores));current=float(self._pilot_values(state['symbols'],state['optical_dc_w'])[0])
        u=state['control'].copy()
        # 仅把舍入级同分视为不移动，避免等价候选间无效来回切换。
        if float(scores[best])>current+1e-14*max(1.,abs(current)):
            u[index]=codes[best]/levels;score=float(scores[best])
        else:score=current
        return u,score,len(codes)

    def _noisy_coefficients(self, state, rng):
        coeff=self._coefficients(state)
        _,var=self._noise(state['optical_dc_w'])
        noise=np.sqrt(var/2)*(rng.standard_normal(255)+1j*rng.standard_normal(255))
        return coeff+noise,noise

    def symbols(self, control, rng=None):
        """供调用方独立评价的[31,3]符号；噪声由同一255系数投影，保留相关性。"""
        state=self.state(control)
        if rng is None:return state['symbols'].T.copy()
        coeff,_=self._noisy_coefficients(state,rng)
        return np.einsum('btk,k->tb',self.extract,coeff)

    def iq(self, control, rng=None):
        """返回512个物理I+jQ电流样点，单位A；噪声不重复加入天线分量。"""
        state=self.state(control)
        coeff=self._coefficients(state) if rng is None else self._noisy_coefficients(state,rng)[0]
        bins=np.zeros(BASEBAND_COUNT,np.complex128);bins[BAND_OFFSETS%BASEBAND_COUNT]=coeff
        return BASEBAND_COUNT*np.fft.ifft(bins)

    def measure(self, control, pilots, rng, return_iq=False):
        """公开反馈，只返回两导频的带噪符号；第三块不暴露给控制算法。

        接收端MMSE反馈采用已标定平均APD噪声水平；实际负MSE不再另加
        期望噪声项。该标定统计假设与只靠两导频估计方差的反馈有所不同。
        """
        supplied=_finite(pilots,(31,2),'测量导频',np.complex128)
        if not np.array_equal(supplied,self.pilots):
            raise ValueError('缓存波形对应固定公共导频，不能替换符号后复用光场。')
        measured=self.measure_detailed(control,rng)
        return (measured['score'],measured['symbols']) if return_iq else measured['score']

    def measure_detailed(self, control, rng):
        """同一次带噪记录的公开反馈、两块导频和可标定APD统计。

        16次探测共用固定的含天线噪声光场缓存，APD抽样独立。这是一组配对
        控制响应，不能称为16次独立天线噪声硬件测量。不返回第三块payload。
        """
        state=self.state(control);coeff,_=self._noisy_coefficients(state,rng)
        y=np.einsum('btk,k->tb',self.extract[:2],coeff)
        psd,var=self._noise(state['optical_dc_w'])
        gain=y[:,0]/self.pilots[:,0]
        w=np.conj(gain)/np.maximum(abs(gain)**2+var*self.symbol_noise_row_norm[0],np.finfo(float).tiny)
        score=-float(np.mean(abs(w*y[:,1]-self.pilots[:,1])**2))
        bins=np.zeros(BASEBAND_COUNT,np.complex128);bins[BAND_OFFSETS%BASEBAND_COUNT]=coeff
        iq=BASEBAND_COUNT*np.fft.ifft(bins)
        return dict(score=score,symbols=y,pilot_iq=iq[:248].copy(),
            noise_symbol_var=var*self.symbol_noise_row_norm[0],
            noise_symbol_var_by_pilot=(var*self.symbol_noise_row_norm[:2]).T,
            optical_dc_w=state['optical_dc_w'],
            apd_dc_a=float(APD_GAIN*(RESPONSIVITY_A_PER_W*state['optical_dc_w']+DARK_CURRENT_A)),
            apd_noise_psd_a2_hz=float(psd),noise_coefficient_variance_a2=float(var),
            noise_model=self.noise_model,
            feedback_definition='negative actual pilot2 MSE; noisy pilot1 MMSE; calibrated APD noise variance',
            paired_frozen_antenna_noise=True,independent_apd_noise_draw=True)


def optimize_teacher(engine, initial_controls, rng, starts=2, sweeps=1):
    """高预算离线教师：公共起点→全有效时延码/衰减码坐标搜索。

    两个起点、每起点一轮为固定默认。教师读取分路缓存，不是只有合路
    反馈的公平在线baseline；不保证全局最优。不把各路最小时延相减。
    """
    initial=np.asarray(initial_controls,float)
    if initial.ndim!=2 or initial.shape[1]!=128 or not len(initial):
        raise ValueError('需要至少一个[128]公共初始控制。')
    if not isinstance(starts,(int,np.integer)) or not 1<=starts<=len(initial):
        raise ValueError('起点数量无效。')
    if not isinstance(sweeps,(int,np.integer)) or sweeps<1:raise ValueError('扫描轮数必须为正整数。')
    initial=np.stack([engine.project(u) for u in initial])
    scores=np.asarray([engine.evaluate(u)['objective'] for u in initial])
    evaluations=len(initial);best_index=int(np.argmax(scores))
    best=initial[best_index].copy();best_score=float(scores[best_index])
    baseline=best.copy();baseline_score=best_score
    for start in np.argsort(scores,kind='stable')[-starts:]:
        state=engine.state(initial[start])
        for _ in range(sweeps):
            for n in rng.permutation(64):
                for index,levels in ((int(n),76),(64+int(n),24)):
                    u,score,calls=engine.scan_coordinate(state,index,np.arange(levels+1)/levels)
                    evaluations+=calls
                    state=engine.state(u)
                    if score>best_score:
                        best=u.copy();best_score=score
    final=engine.evaluate(best);evaluations+=1
    if final['objective']<baseline_score:
        # 最终以普通前向重新确认；任何浮点累计或扫描实现问题都不牺牲起点。
        best=baseline;final=engine.evaluate(best);evaluations+=1
    return best,dict(objective=final['objective'],snr_db=final['snr_db'],
        snr_db_definition=final['snr_db_definition'],
        initial_best_objective=baseline_score,objective_evaluations=evaluations,
        starts=int(starts),sweeps=int(sweeps),globally_optimal=False,
        privileged_branch_cache=True,public_feedback_only=False,teacher_uses_payload=False,
        teacher_pilot1_apd_noise_ignored=True,
        noise_model=engine.noise_model,objective_definition=engine.objective_definition)


# 兼容旧教师入口的调用形式，但生成器应显式导入本文件的已描述教师。
optimize = optimize_teacher
