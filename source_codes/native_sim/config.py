"""本批统一时频网格；载频变化不改变阵列、时窗或控制分辨率。"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class NativeConfig:
    nx: int = 8
    ny: int = 8
    pitch_m: float = 299792458.0 / 8e9
    sample_rate_hz: float = 51.2e9
    sample_count: int = 131072
    iq_sample_rate_hz: float = 200e6
    iq_samples: int = 512
    bandwidth_hz: float = 100e6
    tones: int = 31
    pilot_symbols: int = 2
    probes: int = 16
    useful_samples: int = 64
    cp_samples: int = 60
    delay_levels: int = 76
    delay_max_ps: float = 76 / 51.2e9 * 1e12
    attenuation_levels: int = 24
    attenuation_max_db: float = 12.0
    temperature_k: float = 290.0
    switch_s: float = 1e-6

    def __post_init__(self):
        if (self.n, self.tones, self.probes, self.pilot_symbols) != (64,31,16,2):
            raise ValueError('本模型固定64路、31子载波、16探测、2导频块。')
        if self.sample_count/self.sample_rate_hz != self.iq_samples/self.iq_sample_rate_hz:
            raise ValueError('RF与IQ窗口长度必须相同。')
        if not np.isclose(self.delay_max_ps*1e-12, self.delay_levels/self.sample_rate_hz):
            raise ValueError('时延码必须对应原生整数包络采样。')

    @property
    def n(self): return self.nx*self.ny
    @property
    def positions(self):
        x,y=np.meshgrid(np.arange(self.nx)-(self.nx-1)/2,np.arange(self.ny)-(self.ny-1)/2)
        return np.stack([x.ravel(),y.ravel(),np.zeros(self.n)],axis=-1)*self.pitch_m
    @property
    def offsets_hz(self): return np.arange(-15,16)*self.iq_sample_rate_hz/self.useful_samples
    @property
    def cp_s(self): return self.cp_samples/self.iq_sample_rate_hz
    @property
    def measurement_s(self): return 2*(self.cp_samples+self.useful_samples)/self.iq_sample_rate_hz+self.switch_s
    @property
    def df_hz(self): return self.sample_rate_hz/self.sample_count
    @property
    def band_offsets(self): return np.arange(-127,128,dtype=int)
    @property
    def block_starts(self): return np.array([60,184,308],dtype=int)
    def to_dict(self): return asdict(self)
