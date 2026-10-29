"""重新排版已核对的同输入星座案例；不改控制、不加点、不选择新案例。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from study_full_baselines.common import require_host,sha256,write_json,now
from paper_results_20260927.build_uniform64 import METHODS


def read(p):return json.loads(p.read_text())


def run(project,font):
    require_host();out=project/'dataset_simulation/revision_af_20261029/D/constellation'
    if out.exists():raise FileExistsError(out)
    source=project/'dataset_simulation/diagnostics/20260927_constellation_reconstruction'
    manifest=read(source/'manifest.json');data=source/'reconstruction_case.npz'
    assert sha256(data)==manifest['files'][data.name]
    with np.load(data,allow_pickle=False) as a:
        names=a['methods'].tolist();codes=a['control_code'].copy();got=a['received_qpsk'].copy()
        sent=a['sent_qpsk'].copy();raw=a['metrics'].copy()
    assert names==METHODS and got.shape==(14,8,31)
    protocol=read(project/'dataset_simulation/revision_rq2_af_20261029/full/protocol.json')
    bindings={str(data):sha256(data)};reference={}
    for folder,names0 in [(Path(protocol['source']),protocol['prior']['methods']),
                          (Path(protocol['uniform_source']),protocol['uniform']['methods'])]:
        path=folder/'records/environment_00363/carrier_12.npz'
        marker=read(path.with_suffix('.json'));assert sha256(path)==marker['sha256']
        bindings[str(path)]=sha256(path)
        with np.load(path,allow_pickle=False) as a:
            for i,n in enumerate(names0):
                if n in names:reference[n]=(a['control_code'][i].copy(),a['metrics'][i,:10].copy())
    groups=[];metrics=[]
    for i,n in enumerate(names):
        np.testing.assert_array_equal(codes[i],reference[n][0])
        np.testing.assert_allclose(raw[i,:10],reference[n][1],rtol=1e-12,atol=1e-28)
        errors=np.count_nonzero((got[i].real>=0)!=(sent.real>=0))+np.count_nonzero((got[i].imag>=0)!=(sent.imag>=0))
        nmse=float(np.mean(abs(got[i]-sent)**2)/np.mean(abs(sent)**2))
        assert errors==raw[i,0] and raw[i,1]==496
        np.testing.assert_allclose(nmse,raw[i,2],rtol=1e-12,atol=1e-28)
        metrics.append(dict(method=n,bit_errors=int(errors),bits=496,ber_percent=100*errors/496,rms_evm_percent=100*np.sqrt(nmse)))
        for group in groups:
            if np.array_equal(codes[i],codes[group[0]]):
                np.testing.assert_array_equal(got[i],got[group[0]]);group.append(i);break
        else:groups.append([i])
    assert [len(g) for g in groups]==[12,1,1]
    assert names[groups[1][0]]=='coordinate' and names[groups[2][0]]=='cnn_warm64'
    out.mkdir(parents=True)
    ideal=np.asarray([-1-1j,1-1j,-1+1j,1+1j])/np.sqrt(2)
    target=np.tile(sent,8);quadrant=(target.real>=0).astype(int)+2*(target.imag>=0).astype(int)
    limit=max(1.2,float(max(abs(got.real).max(),abs(got.imag).max()))*1.05)
    colors=['#0072B2','#E69F00','#009E73','#CC79A7']
    for lang in ['en','zh']:
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42 if lang=='en' else 3})
        fp=FontProperties(fname=str(font),size=12) if lang=='zh' else FontProperties(size=12)
        titles=['(a) Transmitted QPSK','(b) 12 baselines: identical output','(c) Coordinate search','(d) Proposed A+F'] if lang=='en' else ['(a) 发送的 QPSK 符号','(b) 12 个基线：相同输出','(c) 坐标搜索','(d) 本文 A+F']
        fig,axes=plt.subplots(2,2,figsize=(7.2,6.6),layout='constrained')
        for ax,title in zip(axes.flat,titles):
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',xlabel='I',ylabel='Q')
            ax.axhline(0,c='gray',lw=.7,ls='--');ax.axvline(0,c='gray',lw=.7,ls='--');ax.grid(alpha=.18)
            ax.scatter(ideal.real,ideal.imag,c='black',marker='x',s=52,linewidths=1.2,zorder=5)
            ax.set_title(title,fontproperties=fp)
        for i,color in enumerate(colors):axes.flat[0].scatter(ideal[i].real,ideal[i].imag,s=125,c=color,alpha=.6)
        for group,ax,panel in zip(groups,list(axes.flat)[1:],['b','c','d']):
            v=got[group[0]].ravel()
            for q,color in enumerate(colors):ax.scatter(v.real[quadrant==q],v.imag[quadrant==q],s=14,c=color,alpha=.65,edgecolors='none')
            for i in group:metrics[i]['panel']=panel
        for ext in ['pdf','png']:fig.savefig(out/f'rq1_constellation_clear_{lang}.{ext}',dpi=180)
        plt.close(fig)
    write_json(out/'method_panel_mapping.json',metrics)
    for path,h in bindings.items():assert sha256(path)==h
    write_json(out/'complete.json',dict(status='passed',at=now(),script_sha256=sha256(Path(__file__)),
        source_files_sha256=bindings,case_manifest_sha256=sha256(source/'manifest.json'),
        source_case=manifest['case'],selection_rule=manifest['selection_rule'],eligible_cases=manifest['eligible_cases'],
        group_members=[[names[i] for i in g] for g in groups],original_data_unchanged=True,
        all_metrics_recomputed=True,all_controls_match_frozen_RQ1=True,
        file_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--zh-font',type=Path,required=True)
    a=p.parse_args();run(a.project.resolve(),a.zh_font.resolve())
