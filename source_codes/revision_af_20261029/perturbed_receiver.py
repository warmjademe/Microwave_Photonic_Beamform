"""在冻结接收器外叠加静态误差和独立激光 RIN，保留原计算核心。"""
from contextlib import contextmanager
import copy
import numpy as np
from native_sim.control_engine import NativeControlEngine,APD_GAIN,RESPONSIVITY_A_PER_W,DEMODULATOR_GAIN
from study_full_baselines.common import rng_for,LEVELS,reception_metrics
from study_full_baselines.confirmation_receiver import frame,scoring_engines
from compact_dataset import pack_observation

SCENARIOS=['nominal','delay_x1p5','angular_x2','power_minus6',
           'rin_minus150','rin_minus140','rin_minus130','delay_error_5ps','delay_error_10ps',
           'loss_error_0p5db','loss_error_1db']


def configuration(original, name):
    if name not in SCENARIOS:raise ValueError(name)
    env=copy.deepcopy(original);delay=np.zeros(64);loss=np.zeros(64);rin=0.
    if name=='delay_x1p5':
        env['delays_s']=(np.asarray(env['delays_s'])*1.5).tolist()
        for key in ['actual_max_delay_ns','pdp_rms_delay_ns','realized_rms_delay_ns']:env[key]*=1.5
        env['factors']['max_delay_ns']*=1.5
    if name=='angular_x2':
        center=np.asarray([env['mean_az_deg'],env['mean_el_deg']])
        env['angles_deg']=(center+2*(np.asarray(env['angles_deg'])-center)).tolist()
        env['factors']['angular_std_deg']*=2
    if name=='power_minus6':env['power_dbm']-=6.
    if name.startswith('rin_minus'):rin=10**(-int(name.split('minus')[1])/10)
    if name.startswith('delay_error'):
        std={'delay_error_5ps':5e-12,'delay_error_10ps':10e-12}[name]
        delay=std*np.clip(rng_for(env['seed'],940).standard_normal(64),-3,3)
    if name.startswith('loss_error'):
        std={'loss_error_0p5db':.5,'loss_error_1db':1.}[name]
        loss=std*np.clip(rng_for(env['seed'],941).standard_normal(64),-3,3)
    assert not env['interference'] and env['doppler_hz']==0
    return env,dict(name=name,delay_error_s=delay.tolist(),loss_error_db=loss.tolist(),rin_linear_per_hz=rin)


class PerturbedEngine(NativeControlEngine):
    def __init__(self,base,config):
        delay=np.asarray(config['delay_error_s']);loss=np.asarray(config['loss_error_db'])
        trans=10**(-loss/10)
        phase=np.exp(-2j*np.pi*delay[:,None]*base.rf_frequency_hz[None,:])
        super().__init__(base.cfg,base.branch_band_w*phase*trans[:,None],base.branch_dc_w*trans,base.carrier_hz,base.pilots)
        self.rin=float(config['rin_linear_per_hz']);self._noise_context=None
        self.noise_model='stationary APD plus independent flat laser RIN; calibrated detector variance'

    def noise_for_transmission(self,transmission):
        t=np.asarray(transmission);power=t@self.branch_dc_w
        psd,var=NativeControlEngine._noise(self,power)
        if self.rin==0:return psd,var
        extra=self.rin*(APD_GAIN*RESPONSIVITY_A_PER_W)**2*np.sum((t*self.branch_dc_w)**2,axis=-1)
        return psd+extra,var+DEMODULATOR_GAIN**2*self.cfg.df_hz*extra

    @contextmanager
    def noise_context(self,transmission):
        if self._noise_context is not None:raise RuntimeError('Nested noise context')
        t=np.asarray(transmission);self._noise_context=(t@self.branch_dc_w,self.noise_for_transmission(t))
        try:yield
        finally:self._noise_context=None

    def _noise(self,optical_dc_w):
        if self._noise_context is None:
            if self.rin:raise RuntimeError('RIN variance requires per-branch transmission')
            return NativeControlEngine._noise(self,optical_dc_w)
        power,answer=self._noise_context
        np.testing.assert_allclose(optical_dc_w,power,rtol=1e-14,atol=0)
        return answer

    def measure_detailed(self,control,rng):
        t=self.transmissions[self.codes(control)[64:]]
        with self.noise_context(t):return super().measure_detailed(control,rng)


def public_input(environment,carrier,public,backend,config):
    base,_=frame(environment,carrier,public['pilot_qpsk'],0,backend)
    engine=PerturbedEngine(base,config)
    measured=[engine.measure_detailed(u,rng_for(environment['seed'],carrier,103,i)) for i,u in enumerate(public['probe_controls'])]
    arrays=dict(combined_iq_a=np.stack([m['symbols'] for m in measured]),quality=np.asarray([m['score'] for m in measured]),
        noise_symbol_var_a2=np.stack([m['noise_symbol_var'] for m in measured]),probe_apd_dc_a=np.asarray([m['apd_dc_a'] for m in measured]))
    return pack_observation(arrays,carrier),engine


def score(environment,carrier,public,backend,config,decisions):
    base,clean_base,payload=scoring_engines(environment,carrier,public,backend)
    engine,clean=PerturbedEngine(base,config),PerturbedEngine(clean_base,config)
    trans=engine.transmissions[decisions['control_code'][:,64:]]
    with engine.noise_context(trans):
        quality,codes=reception_metrics(engine,clean,decisions['control_code']/LEVELS,payload,environment['seed'],carrier)
    np.testing.assert_array_equal(codes,decisions['control_code'])
    return np.column_stack([quality,decisions['costs']])
