"""独立数值诊断：RK4 每个子时刻使用连续带限电输入的精确采样。

只替换输入时间网格，不改速率方程、逐路器件参数和初态算法。
输出仍为每个积分步末的复光场；不声称与 OSD 离散输出等价。
"""
import ctypes
from pathlib import Path
import subprocess
import numpy as np

from laser_clock import SOURCE
from native_sim.laser import PARAMETER_KEYS


def load_staged_clock(output):
    original = (SOURCE / 'native_sim/native_laser.cpp').read_text()
    replacements = [
        ('const double integration_ratio=double(count-1)/double(count);\n'
         '    const double selector_ratio=double(count-2)/double(count+1);\n'
         '    const double step_ns=1e9/fs*integration_ratio;',
         'const double step_ns=1e9/fs;'),
        ('const double* x=input+route*count;',
         'const double* x=input+route*(2*count+1);'),
        ('double stage=double(k)+double(j)*.5;\n'
         '                int64_t left=int64_t(std::floor(stage*selector_ratio));\n'
         '                if(left<0 || left+1>=count) return 2000+route;\n'
         '                double a=stage*integration_ratio-double(left);\n'
         '                current[j]=bias+peak*((1-a)*x[left]+a*x[left+1]);',
         'current[j]=bias+peak*x[2*k+j];'),
    ]
    changed = original
    for old, new in replacements:
        if changed.count(old) != 1:
            raise ValueError('冻结核结构改变，停止自动构建诊断副本。')
        changed = changed.replace(old, new)
    changed = ('// 独立诊断：统一时间、精确子时刻输入；非 OSD 等价核。\n'
               + '\n'.join(changed.splitlines()[2:]) + '\n')
    source = Path(output) / 'staged_clock.cpp'
    source.write_text(changed)
    library = Path(output) / 'staged_clock.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off',
                    '-fPIC', '-shared', str(source), '-o', str(library)],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.native_laser.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int64,
                                ctypes.c_double, pointer, pointer]
    lib.native_laser.restype = ctypes.c_int

    def simulate(stages, sample_rate_hz, profile):
        if np.iscomplexobj(stages):
            raise ValueError('电驱动必须为实数。')
        single = np.ndim(stages) == 1
        x = np.ascontiguousarray(np.atleast_2d(stages), np.float64)
        if (x.ndim != 2 or len(x) != len(profile) or x.shape[1] < 9
                or x.shape[1] % 2 != 1 or not np.all(np.isfinite(x))
                or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0):
            raise ValueError('每路需要 2N+1 个有限子时刻采样及正积分采样率。')
        count = (x.shape[1] - 1) // 2
        parameters = np.array([[r['physical'][k] for k in PARAMETER_KEYS] + [
            r['bias_current_a'], r['modulation_peak_current_a'],
            r['optical_frequency_hz']] for r in profile], np.float64)
        if not np.all(np.isfinite(parameters)):
            raise ValueError('器件参数必须有限。')
        fields = np.empty((len(x), count), np.complex128)
        bounds = np.empty((len(x), 2), np.float64)
        address = lambda a: a.ctypes.data_as(pointer)
        status = lib.native_laser(address(x), address(parameters), len(x), count,
                                  sample_rate_hz, address(fields), address(bounds))
        if status:
            raise FloatingPointError('精确子时刻核状态码：' + str(status))
        return (fields[0], bounds[0]) if single else (fields, bounds)
    return simulate


def stage_samples(positive_coefficients, carrier_hz, cfg, factor):
    """同一连续周期带限输入；闭合点复制 t=0，不用最后两点外推。"""
    if not isinstance(factor, int) or factor < 1:
        raise ValueError('factor 必须为正整数。')
    values = np.asarray(positive_coefficients, complex)
    if values.shape[-1] != len(cfg.band_offsets):
        raise ValueError('正频系数必须覆盖统一的 255 个带内格点。')
    n = cfg.sample_count * factor * 2
    bins = int(round(carrier_hz / cfg.df_hz)) + cfg.band_offsets
    if not np.isclose(round(carrier_hz / cfg.df_hz) * cfg.df_hz,
                      carrier_hz, rtol=0, atol=1e-3):
        raise ValueError('载频须位于共同频格。')
    if bins.min() <= 0 or bins.max() >= n // 2:
        raise ValueError('子时刻采样率不覆盖输入频带。')
    spectrum = np.zeros(values.shape[:-1] + (n // 2 + 1,), complex)
    spectrum[..., bins] = values
    samples = np.fft.irfft(spectrum, n=n, axis=-1) * n
    return np.concatenate([samples, samples[..., :1]], axis=-1)
