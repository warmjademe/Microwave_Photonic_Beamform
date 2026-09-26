"""自动编译自有C++核；输入/输出与冻结Python激光器逐点比较后启用。"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
PARAMETER_KEYS=['active_volume_cm3','quantum_efficiency','spontaneous_emission_factor',
    'gain_compression_cm3','transparency_density_cm_neg3','group_velocity_cm_s',
    'differential_gain_cm2','mode_confinement_factor','carrier_lifetime_s',
    'photon_lifetime_s','linewidth_enhancement_factor']
_LIB=None


def build():
    source=HERE/'native_laser.cpp'
    key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory=HERE/'_build';directory.mkdir(exist_ok=True)
    target=directory/(platform.system()+'-'+platform.machine()+'-'+key+'.so')
    if not target.exists():
        temporary=target.with_suffix('.'+str(os.getpid())+'.tmp.so')
        command=['c++','-std=c++17','-O3','-ffp-contract=off','-fPIC',
            '-dynamiclib' if platform.system()=='Darwin' else '-shared',str(source),'-o',str(temporary)]
        subprocess.run(command,check=True,capture_output=True)
        temporary.replace(target)
        (directory/(target.name+'.json')).write_text(json.dumps(dict(command=command,
            compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')
    return target


def simulate(drive, sample_rate_hz, profile):
    global _LIB
    if np.iscomplexobj(drive): raise ValueError('激光电输入必须为实数。')
    x=np.ascontiguousarray(drive,dtype=np.float64)
    if x.ndim!=2 or len(profile)!=len(x) or x.shape[1]<4 or not np.all(np.isfinite(x)):
        raise ValueError('输入必须是逐路有限电信号。')
    values=np.array([[row['physical'][k] for k in PARAMETER_KEYS]+[row['bias_current_a'],
        row['modulation_peak_current_a'],row['optical_frequency_hz']] for row in profile],dtype=np.float64)
    if not np.all(np.isfinite(values)) or not sample_rate_hz>0:
        raise ValueError('参数必须有限，采样率须为正。')
    if _LIB is None:
        _LIB=ctypes.CDLL(str(build()));pointer=ctypes.POINTER(ctypes.c_double)
        _LIB.native_laser.argtypes=[pointer,pointer,ctypes.c_int,ctypes.c_int64,ctypes.c_double,pointer,pointer]
        _LIB.native_laser.restype=ctypes.c_int
    fields=np.empty(x.shape,np.complex128);bounds=np.empty((len(x),2),np.float64)
    ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    status=_LIB.native_laser(ptr(x),ptr(values),len(x),x.shape[1],float(sample_rate_hz),ptr(fields),ptr(bounds))
    if status: raise FloatingPointError('原生离散核拒绝该输入/状态，状态码='+str(status))
    return fields,bounds


def load_profile():
    return json.loads((HERE/'laser_profile_019.json').read_text())['routes']
