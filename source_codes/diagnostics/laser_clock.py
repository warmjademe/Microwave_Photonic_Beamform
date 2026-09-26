"""单路、无多径、无噪声的时钟对照；不修改冻结激光器。

counterfactual 是排查数值时间网格的物理对照，不代表新模型已与 OSD 校准。
"""
import argparse
import ctypes
import json
import platform
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from compact_dataset import sha256
from generate_native_dataset import write_json, now
from native_sim.config import NativeConfig
from native_sim.laser import simulate, load_profile, PARAMETER_KEYS
from native_sim.control_engine import extraction_matrix
from native_sim.waveforms import transmit_coefficients
from round01 import reception


def load_counterfactual(out):
    original=(SOURCE/'native_sim/native_laser.cpp').read_text()
    changed=original.replace('double(count-1)/double(count)','1.0')
    changed=changed.replace('double(count-2)/double(count+1)','1.0')
    changed=changed.replace('int64_t left=int64_t(std::floor(stage*selector_ratio));',
        'int64_t left=std::min(count-2,int64_t(std::floor(stage*selector_ratio)));')
    if changed==original:raise ValueError('预定数值时间网格对照未生效。')
    path=out/'counterfactual_uniform_clock.cpp';path.write_text(changed)
    library=out/'counterfactual_uniform_clock.so'
    command=['c++','-std=c++17','-O3','-ffp-contract=off','-fPIC','-shared',str(path),'-o',str(library)]
    subprocess.run(command,check=True,capture_output=True)
    lib=ctypes.CDLL(str(library));ptr=ctypes.POINTER(ctypes.c_double)
    lib.native_laser.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int64,ctypes.c_double,ptr,ptr]
    lib.native_laser.restype=ctypes.c_int
    def run(x,fs,profile):
        single=np.ndim(x)==1
        x=np.ascontiguousarray(np.atleast_2d(x),np.float64)
        parameters=np.array([[r['physical'][k] for k in PARAMETER_KEYS]+[
            r['bias_current_a'],r['modulation_peak_current_a'],r['optical_frequency_hz']] for r in profile],np.float64)
        y=np.empty(x.shape,np.complex128);bounds=np.empty((len(x),2))
        address=lambda a:a.ctypes.data_as(ptr)
        status=lib.native_laser(address(x),address(parameters),len(x),x.shape[1],fs,address(y),address(bounds))
        if status:raise FloatingPointError(str(status))
        return (y[0],bounds[0]) if single else (y,bounds)
    return run


def cache(field,fc,cfg):
    n=len(field); f=np.fft.fftshift(np.fft.fft(field))/n
    nf=1<<(2*n-2).bit_length()
    corr=np.fft.ifft(abs(np.fft.fft(f,n=nf))**2)
    bins=int(round(fc/cfg.df_hz))+cfg.band_offsets
    return corr[bins],float(corr[0].real)


def run(root,out):
    if platform.node()!='qyb-HuaShuo':raise RuntimeError('仅在华硕运行。')
    out.mkdir(parents=True,exist_ok=False)
    cfg=NativeConfig();profile=load_profile()[:1]
    with np.load(root/'dataset_simulation/dataset_train/public.npz') as f:pilots=f['pilot_qpsk']
    rng=np.random.default_rng(20260925)
    payload=((2*rng.integers(0,2,31)-1)+1j*(2*rng.integers(0,2,31)-1))/np.sqrt(2)
    coeff=transmit_coefficients(pilots,payload,cfg);ref=np.column_stack([pilots,payload])
    protocol=dict(created_at_utc=now(),scope='single route; no multipath, no fading, no antenna/APD noise',
        carrier_ghz=list(range(4,21)),drive_scale=0.002,reference_blocks=3,
        variants=['frozen_native_clock','uniform_clock_counterfactual'],
        interventions='same ODE, initial condition, parameters, drive, extraction; only time-grid indexing changed',
        frozen_source_sha256=sha256(SOURCE/'native_sim/native_laser.cpp'),source_sha256=sha256(__file__),
        interpret_as='numerical diagnosis, not OSD certification',seed=20260925)
    write_json(out/'protocol.json',protocol)
    alternative=load_counterfactual(out);rows=[];arrays={}
    for ghz in range(4,21):
        fc=ghz*1e9
        spectrum=np.zeros(cfg.sample_count//2+1,complex)
        spectrum[int(round(fc/cfg.df_hz))+cfg.band_offsets]=.001*coeff
        drive=np.fft.irfft(spectrum,n=cfg.sample_count)*cfg.sample_count
        original,bounds=simulate(drive[None,:],cfg.sample_rate_hz,profile)
        corrected,corrected_bounds=alternative(drive,cfg.sample_rate_hz,profile)
        for name,field,bound in [('frozen_native_clock',original[0],bounds[0]),
                                 ('uniform_clock_counterfactual',corrected,corrected_bounds)]:
            band,dc=cache(field,fc,cfg)
            symbols=np.einsum('btk,k->tb',extraction_matrix(),band)
            metrics=reception(symbols,ref)
            rows.append(dict(carrier_ghz=ghz,variant=name,**metrics,optical_dc_w=dc,
                             current_min_a=float(bound[0]),current_max_a=float(bound[1])))
            arrays[name+'_'+str(ghz)]=symbols
            # 频域线性自相关与直接强度FFT在目标带的差别也单独记录。
            direct=np.fft.fft(abs(field)**2)/len(field)
            direct_band=direct[int(round(fc/cfg.df_hz))+cfg.band_offsets]
            rows[-1]['noncircular_vs_circular_band_relative_error']=float(
                np.linalg.norm(band-direct_band)/max(np.linalg.norm(direct_band),1e-100))
        write_json(out/'results.json',rows)
        print(json.dumps(rows[-2:]),flush=True)
    np.savez_compressed(out/'symbols.npz',reference=ref,**arrays)
    if sha256(SOURCE/'native_sim/native_laser.cpp')!=protocol['frozen_source_sha256']:
        raise ValueError('冻结核改变。')
    write_json(out/'complete.json',dict(status='complete',finished_at_utc=now(),
        original_unchanged=True,results_sha256=sha256(out/'results.json'),
        counterfactual_source_sha256=sha256(out/'counterfactual_uniform_clock.cpp')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.root,a.output)
    except BaseException:
        if a.output.exists():write_json(a.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise
