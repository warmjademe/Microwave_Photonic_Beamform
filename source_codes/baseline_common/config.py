"""首批研究配置：单位显式记录；不将研究网格称为 OSD 实测分辨率。"""
from dataclasses import dataclass, asdict
import numpy as np

C0 = 299792458.0
KB = 1.380649e-23


@dataclass(frozen=True)
class Config:
    nx: int = 8
    ny: int = 8
    pitch_m: float = C0 / (2 * 4e9)
    bandwidth_hz: float = 100e6
    tones: int = 64
    pilot_symbols: int = 128
    probes: int = 16
    cp_s: float = 300e-9
    delay_max_ps: float = 1500.0
    delay_levels: int = 300
    attenuation_max_db: float = 12.0
    attenuation_levels: int = 24
    temperature_k: float = 290.0
    # 切换时间属于研究预算假设，不宣称已测得实体器件速度。
    switch_s: float = 1e-6

    def __post_init__(self):
        if self.n != 64 or self.tones != 64 or self.probes != 16:
            raise ValueError('v1 固定 64 阵元、64 频点和 16 组初始探测。')
        if min(self.bandwidth_hz, self.pitch_m, self.delay_max_ps,
               self.attenuation_max_db, self.delay_levels, self.attenuation_levels) <= 0:
            raise ValueError('控制范围与带宽必须为正。')
        if self.pilot_symbols < 2:
            raise ValueError('噪声残差估计至少需要两个参考符号。')

    @property
    def n(self):
        return self.nx * self.ny

    @property
    def positions(self):
        x, y = np.meshgrid(np.arange(self.nx)-(self.nx-1)/2,
                           np.arange(self.ny)-(self.ny-1)/2)
        return np.stack([x.ravel(), y.ravel(), np.zeros(self.n)], -1)*self.pitch_m

    @property
    def offsets_hz(self):
        return np.fft.fftshift(np.fft.fftfreq(self.tones, 1/self.bandwidth_hz))

    @property
    def measurement_s(self):
        return self.pilot_symbols*(self.tones/self.bandwidth_hz+self.cp_s)+self.switch_s

    def to_dict(self):
        return asdict(self)


def cn(rng, shape):
    """单位平均功率的圆对称复高斯噪声。"""
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)


def qpsk(rng, shape):
    """单位平均功率 QPSK；符号编号由实、虚部正负确定。"""
    return ((2*rng.integers(0, 2, shape)-1)+1j*(2*rng.integers(0, 2, shape)-1))/np.sqrt(2)


def rng_for(seed, *parts):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, parts)]))
