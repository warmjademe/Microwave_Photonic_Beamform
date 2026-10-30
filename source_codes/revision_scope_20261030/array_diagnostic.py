"""冻结阵列上的真时延/中心频率匹配相移对照；仅在远程实验机运行。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source_codes'))
from native_sim.config import NativeConfig
from baseline_common.controls import direction


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    if platform.system() != 'Linux' or str(ROOT) != '/home/qyb/RESEARCH/wangluqiang_2026_paper_1':
        raise RuntimeError('Numerical diagnostic must run in the authorised remote project.')
    if (output/'complete.json').exists():
        raise FileExistsError('Completed result already exists; do not overwrite.')
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name('protocol.json')
    protocol = json.loads(protocol_path.read_text())
    cfg = NativeConfig()
    assert cfg.n == 64 and cfg.bandwidth_hz == 1e8 and cfg.delay_levels == 76
    az = np.linspace(-35,35,41)
    el = np.linspace(-15,15,21)
    steering = np.array([(a,e) for a in az for e in el])
    geo = direction(steering[:,0],steering[:,1])@cfg.positions.T/299792458.0
    tau = geo-geo.min(axis=1,keepdims=True)
    codes = np.floor(tau*cfg.sample_rate_hz+.5).astype(int)
    assert codes.min() >= 0 and codes.max() <= cfg.delay_levels
    quantized = codes/cfg.sample_rate_hz
    offsets = np.unique(np.r_[cfg.offsets_hz, -cfg.bandwidth_hz/2, cfg.bandwidth_hz/2])
    carriers = np.array(protocol['carriers_ghz'])*1e9
    gains = np.empty((len(carriers),len(steering),len(offsets),2))
    for i,fc in enumerate(carriers):
        freq = fc+offsets
        propagation = np.exp(2j*np.pi*freq[None,:,None]*geo[:,None,:])
        true_delay = np.exp(-2j*np.pi*freq[None,:,None]*quantized[:,None,:])
        phase_only = np.exp(-2j*np.pi*fc*quantized[:,None,:])
        gains[i,:,:,0] = abs(np.mean(propagation*true_delay,axis=-1))**2
        gains[i,:,:,1] = abs(np.mean(propagation*phase_only,axis=-1))**2
    assert np.isfinite(gains).all() and (gains>0).all()
    center = np.flatnonzero(offsets==0)[0]
    np.testing.assert_allclose(gains[:,:,center,0],gains[:,:,center,1],rtol=0,atol=1e-14)
    broadside = np.flatnonzero((steering==0).all(axis=1))[0]
    np.testing.assert_allclose(gains[:,broadside],1,rtol=0,atol=1e-14)
    db = 10*np.log10(gains)
    delta = db[...,0]-db[...,1]
    indices = np.unravel_index(np.argmax(abs(delta)),delta.shape)
    maxima = dict(absolute_difference_db=float(abs(delta[indices])),signed_ttd_minus_phase_db=float(delta[indices]),
                  carrier_ghz=float(carriers[indices[0]]/1e9),steering_deg=steering[indices[1]].tolist(),offset_mhz=float(offsets[indices[2]]/1e6))
    summaries = []
    for i,fc in enumerate(carriers):
        d = delta[i]
        summaries.append(dict(carrier_ghz=float(fc/1e9),minimum_difference_db=float(d.min()),
                              median_difference_db=float(np.median(d)),maximum_difference_db=float(d.max()),
                              mean_difference_db=float(d.mean()),maximum_absolute_difference_db=float(abs(d).max())))
    aliases = []
    for fc in [4e9,12e9,20e9]:
        pitch_wave = cfg.pitch_m*fc/299792458.0
        values = [float(np.degrees(np.arcsin(m/pitch_wave))) for m in range(-4,5) if m and abs(m/pitch_wave)<=1]
        aliases.append(dict(carrier_ghz=fc/1e9,broadside_azimuth_replica_deg=values))
        for angle in values:
            response = abs(np.mean(np.exp(2j*np.pi*fc*(cfg.positions@direction(angle,0)/299792458.0))))**2
            np.testing.assert_allclose(response,1,rtol=0,atol=1e-12)
    summary = dict(at=datetime.now(timezone.utc).isoformat(),protocol=protocol,
                   steering_count=len(steering),carrier_count=len(carriers),offset_count=len(offsets),
                   paired_conditions=int(np.prod(delta.shape)),maximum_used_delay_ps=float(quantized.max()*1e12),
                   prescribed_control_range_max_phase_difference_deg=float(360*(cfg.bandwidth_hz/2)*cfg.delay_max_ps*1e-12),
                   steering_grid_max_phase_difference_deg=float(360*(cfg.bandwidth_hz/2)*quantized.max()),
                   global_extreme=maxima,per_carrier=summaries,aliases=aliases,
                   checks='all passed',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                      [Path(__file__),protocol_path,ROOT/'source_codes/native_sim/config.py',ROOT/'source_codes/baseline_common/controls.py']})
    np.savez_compressed(output/'array_response.npz',carriers_hz=carriers,steering_deg=steering,
                        offsets_hz=offsets,delay_codes=codes,normalised_power=gains)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42})
    fig,axes = plt.subplots(3,2,figsize=(7.2,6.8),sharex=True,sharey=True,layout='constrained')
    look = np.linspace(-90,90,3601)
    looks = cfg.positions@direction(look,np.zeros_like(look)).T/299792458.0
    for row,fc in enumerate([4e9,12e9,20e9]):
        for col,target in enumerate([0,35]):
            index = np.flatnonzero((steering==[target,0]).all(axis=1))[0]
            tq = quantized[index]
            def power(f,control_f):
                return abs(np.mean(np.exp(2j*np.pi*(f*looks-control_f*tq[:,None])),axis=0))**2
            ax = axes[row,col]
            ax.plot(look,10*np.log10(np.maximum(power(fc,fc),1e-9)),color='0.35',lw=.9,label='Centre: both controls')
            ax.plot(look,10*np.log10(np.maximum(power(fc+50e6,fc+50e6),1e-9)),color='#0072B2',lw=1.1,label='+50 MHz: true delay')
            ax.plot(look,10*np.log10(np.maximum(power(fc+50e6,fc),1e-9)),color='#D55E00',ls='--',lw=1,label='+50 MHz: matched phase')
            ax.axvline(target,color='0.65',lw=.6,ls=':')
            ax.set_title(f'{fc/1e9:g} GHz; steering {target} deg')
            ax.set_ylim(-40,1);ax.set_xlim(-90,90);ax.set_xticks([-90,-45,0,45,90]);ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Normalised power (dB)')
            if row==2:ax.set_xlabel('Azimuth (deg); elevation = 0')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=1,frameon=False)
    fig.savefig(output/'array_ttd_phase.pdf')
    fig.savefig(output/'array_ttd_phase.png',dpi=160)
    plt.close(fig)
    complete=dict(status='complete',checks='passed',paired_conditions=summary['paired_conditions'],
                  outputs={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                  source_sha256=summary['source_sha256'],no_receiver_or_training_run=True)
    (output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(dict(status='complete',conditions=summary['paired_conditions'],extreme=maxima,per_carrier=[s for s in summaries if s['carrier_ghz'] in [4,12,20]])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
