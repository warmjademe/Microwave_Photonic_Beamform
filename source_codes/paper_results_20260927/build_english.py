"""English-label figures from frozen paper data; no training or control selection."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.paired_statistics import sufficient,metrics

NAMES=['initial_select64','codebook','coordinate','spsa','done','de','mlp_feedback64','dnn_feedback64',
       'cnn_feedback64','rescnn_feedback64','transformer_feedback64','complex_cnn_feedback64','jct_feedback64','cnn_warm64']
LABELS=['Fixed scan','Geometric codebook','Coordinate search','SPSA','DONE','DE','MLP + feedback','DNN + feedback',
        'CNN + feedback','ResCNN + feedback','Transformer + feedback','Complex CNN + feedback','JCT + feedback','Proposed']
COLORS=['#7f7f7f','#0072B2','#999933','#CC79A7','#E69F00','#009E73','#332288','#44AA99','#882255','#88CCEE','#AA4499','#661100','#117733','#D55E00']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(project,out):
    assert 'huashuo' in socket.gethostname().lower()
    out.mkdir(parents=True,exist_ok=False)
    root=project/'dataset_simulation';analysis=root/'baseline_results/20260927_uniform64_all13/analysis'
    completed=read(analysis/'complete.json')
    for name,digest in completed['file_sha256'].items():assert sha(analysis/name)==digest,name
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight')
        plt.close(fig)
    groups=read(analysis/'groups.json')
    fig,axes=plt.subplots(2,2,figsize=(10.6,7.5))
    configs=[('carrier_ghz','ber',100,'Carrier (GHz)','BER (%)'),('carrier_ghz','rms_evm_percent',1,'Carrier (GHz)','RMS EVM (%)'),
             ('power_bin','ber',100,'Nominal received power bin center (dBm)','BER (%)'),('power_bin','paired_output_snr_db',1,'Nominal received power bin center (dBm)','Paired SNR (dB)')]
    for ax,(group,metric,scale,xlabel,ylabel) in zip(axes.flat,configs):
        for i,(name,label,color) in enumerate(zip(NAMES,LABELS,COLORS)):
            rows=sorted([g for g in groups if g['method']==name and g['group']==group],key=lambda r:r['value'])
            x=[r['value'] if group=='carrier_ghz' else -102.5+5*r['value'] for r in rows]
            ax.plot(x,[r[metric]*scale for r in rows],label=label,color=color,marker=['x','s','v','D','P','^','o','<','>','h','*','+','d','o'][i],
                    linewidth=1.8 if name=='cnn_warm64' else .9,markersize=3,linestyle='--' if '_feedback64' in name else '-')
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.83));save(fig,'rq1_quality_profiles_all13_64')
    with np.load(analysis/'environment_metrics.npz') as a:values=metrics(sufficient(a['metrics']));order=a['methods'].tolist()
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7))
    for name,label,color in zip(NAMES,LABELS,COLORS):
        v=values[:,order.index(name)]
        for ax,k in zip(axes,[3,4]):ax.scatter(v[:,k],v[:,0]*100,s=5,alpha=.25,color=color,label=label,rasterized=True)
    for ax,xlabel in zip(axes,['Post-equalization RMS EVM (%)','Pre-equalization paired SNR (dB)']):ax.set(xlabel=xlabel,ylabel='Environment BER (%)');ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.73));save(fig,'rq1_metric_relations_all13_64')
    origin_path=root/'diagnostics/20260927_uniform64_all13_figures/selection_origins.json';origins=read(origin_path)
    keys=list(origins);fig,ax=plt.subplots(figsize=(10.6,4.5));left=np.zeros(len(keys))
    for key,label,color in [('initial','Shared initial controls','#999999'),('model','New model candidate','#D55E00'),('codebook','New codebook controls','#0072B2')]:
        v=np.array([origins[n][key]/origins[n]['total']*100 for n in keys]);ax.barh(range(len(keys)),v,left=left,label=label,color=color);left+=v
    np.testing.assert_allclose(left,100)
    ax.set(yticks=range(len(keys)),yticklabels=[LABELS[NAMES.index(n)] for n in keys],xlim=(0,100),xlabel='Source of selected control (%)');ax.invert_yaxis()
    ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1),frameon=False);fig.tight_layout();save(fig,'rq1_selection_origins_64')
    case=root/'diagnostics/20260927_constellation_reconstruction';cm=read(case/'manifest.json')
    assert sha(case/'reconstruction_case.npz')==cm['files']['reconstruction_case.npz']
    with np.load(case/'reconstruction_case.npz') as a:received=a['received_qpsk'];sent=a['sent_qpsk'];code=a['control_code'];raw=a['metrics'];names=a['methods'].tolist()
    assert names==NAMES
    clusters=[]
    for i in range(14):
        for g in clusters:
            if np.array_equal(code[i],code[g[0]]):
                np.testing.assert_array_equal(received[i],received[g[0]]);g.append(i);break
        else:clusters.append([i])
    assert [len(g) for g in clusters]==[12,1,1]
    ber=[];evm=[]
    for i in range(14):
        errors=np.count_nonzero((received[i].real>=0)!=(sent.real>=0))+np.count_nonzero((received[i].imag>=0)!=(sent.imag>=0))
        nmse=float(np.mean(abs(received[i]-sent)**2)/np.mean(abs(sent)**2));assert errors==raw[i,0];np.testing.assert_allclose(nmse,raw[i,2],rtol=1e-12)
        ber.append(errors/496*100);evm.append(100*nmse**.5)
    fig=plt.figure(figsize=(10.6,10.3),layout='constrained');grid=fig.add_gridspec(3,2,height_ratios=[1,1,.85]);axes=[fig.add_subplot(grid[r,c]) for r in range(2) for c in range(2)]
    ideal=np.array([-1-1j,1-1j,-1+1j,1+1j])/np.sqrt(2);colors=['#0072B2','#E69F00','#009E73','#CC79A7'];target=np.tile(sent,8)
    quadrant=(target.real>=0).astype(int)+2*(target.imag>=0).astype(int);limit=max(1.2,max(abs(received.real).max(),abs(received.imag).max())*1.05)
    for ax in axes:
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',xlabel='I',ylabel='Q');ax.axhline(0,color='gray',lw=.65,ls='--');ax.axvline(0,color='gray',lw=.65,ls='--');ax.grid(alpha=.15);ax.scatter(ideal.real,ideal.imag,marker='x',s=48,color='black',zorder=5)
    for i,color in enumerate(colors):axes[0].scatter(ideal[i].real,ideal[i].imag,s=150,color=color,alpha=.55)
    axes[0].set_title('(a) Transmitted ideal QPSK symbols\nCrosses mark intended symbol locations')
    mapping={}
    for g,ax,title,panel in zip(clusters,axes[1:],['(b) 12 baselines with identical controls','(c) Coordinate search','(d) Proposed'],['b','c','d']):
        i=g[0];v=received[i].ravel()
        for q,color in enumerate(colors):ax.scatter(v.real[quadrant==q],v.imag[quadrant==q],s=13,alpha=.62,color=color,edgecolors='none')
        ax.set_title(title+'\nBER %.4f%% / EVM %.4f%%'%(ber[i],evm[i]))
        for j in g:mapping[names[j]]=panel
    for col,indices in enumerate([range(7),range(7,14)]):
        ax=fig.add_subplot(grid[2,col]);ax.axis('off')
        rows=[[LABELS[i],'('+mapping[names[i]]+')','%.4f'%ber[i],'%.4f'%evm[i]] for i in indices]
        table=ax.table(cellText=rows,colLabels=['Method','Panel','BER (%)','EVM (%)'],colWidths=[.49,.11,.2,.2],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.6)
        for (r,c),cell in table.get_celld().items():cell.set_edgecolor('#d5d5d5');cell.set_linewidth(.4)
    fig.suptitle('Same-input recovery: identical outputs share a panel\nTest index 363 | 12 GHz | 64 probes per method | 248 received symbols per panel',fontsize=12)
    save(fig,'rq1_constellation_all13_12_64')
    scale_manifest=root/'diagnostics/20260927_paper_rq3_figures_02/manifest.json';scale=read(scale_manifest)
    for name,digest in scale['sources'].items():assert sha(project/name)==digest
    curves=scale['curves'];fig,axes=plt.subplots(1,2,figsize=(10.6,4.1))
    for curve,color,marker,label in zip(curves,['#0072B2','#D55E00'],['o','s'],['40 epochs','9200 updates']):
        rows=curve['rows'];assert [r['training_environments'] for r in rows]==[216,432,864,1728,3456]
        for ax,key,mult in zip(axes,['ber','rms_evm_percent'],[100,1]):ax.plot(range(5),[r[key]*mult for r in rows],color=color,marker=marker,label=label)
    for ax,ylabel in zip(axes,['Validation BER (%)','Validation RMS EVM (%)']):
        ax.set(xticks=range(5),xticklabels=['216','432','864','1728','3456'],xlabel='Training environments',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();save(fig,'rq3_training_quality')
    proof=dict(status='passed',language='English',new_experiments=False,controls_and_results_unchanged=True,
               final_analysis_sha256=sha(analysis/'complete.json'),case_arrays_sha256=sha(case/'reconstruction_case.npz'),
               scale_source_manifest_sha256=sha(scale_manifest),selection_origins_sha256=sha(origin_path),
               script_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in out.iterdir()})
    (out/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':'passed','figures':5}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.project,a.output)
