"""绘制统一 64 次的 13 基线结果，并核对固定案例与候选选择来源。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))

METHODS=['initial_select64','codebook','coordinate','spsa','done','de',
         'mlp_feedback64','dnn_feedback64','cnn_feedback64','rescnn_feedback64',
         'transformer_feedback64','complex_cnn_feedback64','jct_feedback64','cnn_warm64']
LABELS=['固定扫描','几何码本','坐标搜索','SPSA','DONE','差分进化',
        'MLP＋反馈','DNN＋反馈','CNN＋反馈','ResCNN＋反馈','Transformer＋反馈',
        '复数 CNN＋反馈','JCT＋反馈','本文方法']
COLORS=['#7f7f7f','#0072B2','#999933','#CC79A7','#E69F00','#009E73',
        '#332288','#44AA99','#882255','#88CCEE','#AA4499','#661100','#117733','#D55E00']
MARKERS=['x','s','v','D','P','^','o','<','>','h','*','+','d','o']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(project,output):
    if 'huashuo' not in socket.gethostname().lower():raise RuntimeError('仅在华硕绘图及回放')
    if output.exists():raise FileExistsError(output)
    root=project/'dataset_simulation/baseline_results/20260927_uniform64_all13'
    analysis=root/'analysis';complete=read(analysis/'complete.json')
    for name,digest in complete['file_sha256'].items():
        assert sha(analysis/name)==digest,name
    protocol=read(root/'protocol.json');prior=Path(protocol['source_output'])
    old_protocol=read(prior/'protocol.json');audit=read(analysis/'audit.json')
    assert sha(root/'protocol.json')==audit['protocol_sha256']
    summary={r['method']:r for r in read(analysis/'summary.json')}
    assert set(summary)==set(METHODS) and all(r['measurement_budget']==64 for r in summary.values())
    groups=read(analysis/'groups.json');comparisons=read(analysis/'comparisons.json')
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import cairosvg
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'svg.fonttype':'path','font.size':10})
    figures=[]
    def save(fig,name):
        for ext in ['svg','png']:fig.savefig(output/(name+'.'+ext),dpi=240,bbox_inches='tight')
        cairosvg.svg2pdf(url=str(output/(name+'.svg')),write_to=str(output/(name+'.pdf')))
        figures.append(name+'.pdf');plt.close(fig)

    configs=[('carrier_ghz','ber',100,'载频（GHz）','BER（%）'),
             ('carrier_ghz','rms_evm_percent',1,'载频（GHz）','RMS EVM（%）'),
             ('power_bin','ber',100,'名义接收功率档中心（dBm）','BER（%）'),
             ('power_bin','paired_output_snr_db',1,'名义接收功率档中心（dBm）','配对 SNR（dB）')]
    fig,axes=plt.subplots(2,2,figsize=(9.3,6.8))
    for ax,(group,metric,scale,xlabel,ylabel) in zip(axes.ravel(),configs):
        for name,label,color,marker in zip(METHODS,LABELS,COLORS,MARKERS):
            rows=sorted([r for r in groups if r['method']==name and r['group']==group],key=lambda r:r['value'])
            x=[r['value'] if group=='carrier_ghz' else -102.5+5*r['value'] for r in rows]
            ax.plot(x,[scale*r[metric] for r in rows],label=label,color=color,marker=marker,
                    linewidth=1.8 if name=='cnn_warm64' else .9,markersize=2.7,
                    linestyle='-' if name=='cnn_warm64' or '_feedback64' not in name else '--')
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.87));save(fig,'rq1_quality_profiles_all13_64')

    # 所有比较均为64次，点线直接使用已冻结统计程序的差值与区间。
    pairs={r['baseline']:r for r in comparisons if r['metric']=='ber'}
    fig,ax=plt.subplots(figsize=(9.3,4.4))
    for i,(name,label,color) in enumerate(zip(METHODS[:-1],LABELS[:-1],COLORS[:-1])):
        r=pairs[name];value=100*r['estimate'];lo,hi=np.array(r['ci95'])*100
        ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=color,capsize=3,markersize=4)
    ax.axvline(0,color='black',linewidth=.8,linestyle='--')
    ax.set(yticks=range(13),yticklabels=LABELS[:-1],xlabel='本文方法减去基线的 BER（百分点；负值表示改善）')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'rq1_paired_all13_64')

    from study_full_baselines.paired_statistics import sufficient,metrics
    with np.load(analysis/'environment_metrics.npz') as f:
        env_values=metrics(sufficient(f['metrics']));names=f['methods'].tolist()
    fig,axes=plt.subplots(1,2,figsize=(9.3,4.1))
    for name,label,color in zip(METHODS,LABELS,COLORS):
        v=env_values[:,names.index(name)]
        for ax,idx in zip(axes,[3,4]):
            ax.scatter(v[:,idx],100*v[:,0],s=5,alpha=.25,color=color,label=label,rasterized=True)
    for ax,xlabel in zip(axes,['均衡后 RMS EVM（%）','均衡前配对 SNR（dB）']):
        ax.set(xlabel=xlabel,ylabel='环境 BER（%）');ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.79));save(fig,'rq1_metric_relations_all13_64')

    # 审计模型候选、共同初始控制与码本分别贡献多少最终选择。
    from study_uniform64.run import load_original
    from study_uniform64.controllers import DIRECT
    selection={n:dict(initial=0,model=0,codebook=0,model_newly_measured=0,total=0)
               for n in [x+'_feedback64' for x in DIRECT]+['cnn_warm64']}
    for record in read(root/'records.json'):
        row=next(r for r in protocol['rows'] if r['index']==record['index'])
        for item in record['carriers']:
            assert sha(root/item['path'])==item['sha256']
            with np.load(root/item['path']) as f:new={k:f[k].copy() for k in f.files}
            _,old,_=load_original(row,item['carrier_ghz'],protocol)
            for name,count in selection.items():
                if name=='cnn_warm64':
                    trace=old[name+'__trace_control_code'];scores=old[name+'__trace_scores']
                    proposed=old['control_code'][old_protocol['methods'].index('complex_response_cnn')]
                else:
                    trace=new[name+'__trace_control_code'];scores=new[name+'__trace_scores']
                    proposed=new[name+'__proposal_control_code']
                fresh=not any(np.array_equal(proposed,u) for u in trace[:16])
                assert any(np.array_equal(proposed,u) for u in trace)
                index=int(scores.argmax());winner=trace[index]
                origin='initial' if index<16 else 'model' if np.array_equal(winner,proposed) else 'codebook'
                count[origin]+=1;count['model_newly_measured']+=int(fresh);count['total']+=1
    assert all(r['total']==864*17 for r in selection.values())
    (output/'selection_origins.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    fig,ax=plt.subplots(figsize=(9.3,3.8));order=list(selection);positions=np.arange(len(order));left=np.zeros(len(order))
    for source,label,color in [('initial','共同初始控制','#999999'),('model','模型新增候选','#D55E00'),('codebook','码本新增控制','#0072B2')]:
        values=np.array([selection[n][source]/selection[n]['total']*100 for n in order])
        ax.barh(positions,values,left=left,label=label,color=color,height=.7);left+=values
    ax.set(yticks=positions,yticklabels=[LABELS[METHODS.index(n)] for n in order],xlim=(0,100),
           xlabel='最终实测择优控制的来源（%）');ax.invert_yaxis();ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1),frameon=False)
    fig.tight_layout();save(fig,'rq1_selection_origins_64')

    # 测试案例固定为索引0，保持先前固定的4/12/20GHz，不依据改进幅度选图。
    from study_full_baselines.confirmation_receiver import scoring_engines
    from study_full_baselines.export_signal_examples import photonic_trace,checked_equal,quality
    from study_full_baselines.common import verify_sources
    from our_method_response_control.train import precision
    import torch
    assert torch.cuda.is_available();precision();verify_sources(protocol['source_sha256'])
    with np.load(Path(protocol['runtime_bundle'])/'public.npz') as f:public={k:f[k].copy() for k in f.files}
    row=protocol['rows'][0];cases=[]
    for fc in [4,12,20]:
        env,old,_=load_original(row,fc,protocol)
        p=root/'records/environment_00000'/('carrier_%02d.npz'%fc)
        assert sha(p)==audit['raw_file_sha256'][str(p.relative_to(root))]
        with np.load(p) as f:new={k:f[k].copy() for k in f.files}
        engine,clean,payload=scoring_engines(env,fc,public,protocol['backend']);traces=[]
        for name in METHODS:
            source=new if name in protocol['methods'] else old
            order=protocol['methods'] if source is new else old_protocol['methods'];index=order.index(name)
            trace=photonic_trace(engine,clean,source['control_code'][index],payload,env['seed'],fc)
            checked_equal(trace['metrics'],source['metrics'][index,:10],name);traces.append(trace)
        received=np.stack([r['received_qpsk'] for r in traces]);target=np.tile(payload,8)
        quadrant=(target.real>0).astype(int)+2*(target.imag>0).astype(int)
        limit=max(1.2,float(np.max(np.maximum(abs(received.real),abs(received.imag))))*1.05)
        np.savez_compressed(output/('constellation_all13_%02d.npz'%fc),methods=METHODS,
            received_qpsk=received,sent_qpsk=payload,metrics=np.stack([r['metrics'] for r in traces]))
        fig,axes=plt.subplots(4,4,figsize=(9.3,9.3),sharex=True,sharey=True)
        for ax,name,label,trace in zip(axes.ravel(),METHODS,LABELS,traces):
            got=trace['received_qpsk'].ravel()
            for q,color in enumerate(['#0072B2','#E69F00','#009E73','#CC79A7']):
                ax.scatter(got.real[quadrant==q],got.imag[quadrant==q],s=6,alpha=.45,color=color,edgecolors='none')
            ideal=np.array([-1-1j,1-1j,-1+1j,1+1j])/np.sqrt(2)
            ax.scatter(ideal.real,ideal.imag,marker='x',s=28,color='black',linewidths=.8)
            ax.axhline(0,color='grey',linewidth=.5,linestyle='--');ax.axvline(0,color='grey',linewidth=.5,linestyle='--')
            v=quality(trace['metrics']);ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',
                xlabel='I',ylabel='Q',title=label+'\nBER %.2f%% / EVM %.2f%%'%(100*v['ber'],v['rms_evm_percent']))
            ax.title.set_fontsize(10)
            ax.grid(alpha=.15)
        for ax in axes.ravel()[14:]:ax.axis('off')
        axes.ravel()[14].text(.05,.5,'共同条件：测试索引 0\n载频 %d GHz\n每方法 64 次探测\n每图 248 个业务符号'%fc,transform=axes.ravel()[14].transAxes,va='center',fontsize=9)
        axes.ravel()[15].text(.05,.5,'黑叉：理想 QPSK 点\n虚线：判决边界\n颜色：实际发送符号\n所有面板坐标范围相同',transform=axes.ravel()[15].transAxes,va='center',fontsize=9)
        fig.tight_layout();save(fig,'rq1_constellation_all13_%02d_64'%fc)
        cases.append(dict(carrier_ghz=fc,environment_id=row['environment_id'],methods={n:quality(t['metrics']) for n,t in zip(METHODS,traces)}))
    manifest=dict(analysis_complete_sha256=sha(analysis/'complete.json'),
        source_protocol_sha256=sha(root/'protocol.json'),builder_sha256=sha(__file__),
        methods=METHODS,measurements=64,environment_index=0,cases=cases,
        controls_reoptimized=False,figures={n:sha(output/n) for n in figures})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'status':'complete','figures':len(figures),'selection_origins':selection}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.project,a.output)
