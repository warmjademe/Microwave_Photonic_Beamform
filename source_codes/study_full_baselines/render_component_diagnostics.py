"""华硕生成探索性组件图，全部图明确标注未完成最终确认。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,sha256,write_json,now


def run(project,output):
    require_host()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # TTC在Matplotlib中可能登记为JP面，不能只依赖fontconfig显示的SC族名。
    font_path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font_path)
    font_name=font_manager.FontProperties(fname=font_path).get_name()
    from matplotlib.ft2font import FT2Font
    if not all(FT2Font(font_path).get_char_index(ord(c)) for c in '误码率信号训练环境协方差'):
        raise RuntimeError('中文字体未找到，不输出缺字图。')
    plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,
        'pdf.fonttype':42,'svg.fonttype':'path','font.size':10})
    base=project/'dataset_simulation';pilot=base/'diagnostics/20260926_pilot_components'
    meta=json.loads((pilot/'summary.json').read_text())
    if meta['status']!='complete' or meta['environments']!=24 or meta['cases']!=72:
        raise ValueError('固定探索性子组未完整完成。')
    output.mkdir(parents=True,exist_ok=True);files=[]
    def save(fig,name,bottom=.035):
        fig.tight_layout(rect=(0,bottom,1,.945))
        for ext in ['png','pdf','svg']:
            file=output/(name+'.'+ext);fig.savefig(file,dpi=180,bbox_inches='tight');files.append(file)
        plt.close(fig)
    names={'covariance':'训练协方差','cnn':'复响应CNN','per_tone_relative':'逐频相对线性',
           'joint_relative':'联合相对线性','joint_absolute':'联合普通线性'}
    controls={'base_2sweeps':'单起点2轮','single_10sweeps':'单起点10轮','multi_mmse':'5起点各2轮'}
    labels=[names[r['method'].split('__')[0]]+' / '+controls[r['method'].split('__')[1]] for r in meta['records']]
    colors=['#0072B2' if r['method'].startswith('cnn__') else '#D55E00' if r['method'].startswith('covariance__')
            else '#777777' for r in meta['records']]
    fig,axes=plt.subplots(1,3,figsize=(15.5,7),sharey=True)
    for ax,key,title,multiplier in zip(axes,['ber','rms_evm_percent','paired_output_snr_db'],
            ['误码率 BER（%）↓','均方根 EVM（%）↓','配对输出 SNR（dB）↑'],[100,1,1]):
        values=np.asarray([r[key]*multiplier for r in meta['records']])
        ax.barh(np.arange(len(labels)),values,color=colors,alpha=.9)
        ax.set_xlabel(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_xlim(0,max(values)*1.2)
        for i,v in enumerate(values):ax.text(v+max(values)*.015,i,'%.2f'%v,va='center',fontsize=9)
    axes[0].set_yticks(np.arange(len(labels)),labels);axes[0].invert_yaxis()
    fig.suptitle('固定旧测试小组：相同公开测量下的估计器与控制器比较',fontsize=15)
    fig.text(.5,.012,'24个环境 × 4/12/20 GHz；每个控制8次配对噪声。探索性结果，尚未完成新留出确认。',ha='center',fontsize=10)
    save(fig,'pilot_quality')
    with np.load(pilot/'all_metrics.npz') as f:a=f['metrics'].reshape(24,3,12,-1)
    order=[r['method'] for r in meta['records']]
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,name in zip(axes,['covariance','cnn','joint_relative']):
        i=order.index(name+'__base_2sweeps');j=order.index(name+'__multi_mmse')
        x=100*a[:,:,i,0].sum(1)/a[:,:,i,1].sum(1);y=100*a[:,:,j,0].sum(1)/a[:,:,j,1].sum(1)
        end=max(5,float(max(x.max(),y.max()))*1.07)
        ax.plot([0,end],[0,end],color='#999999',linestyle='--',linewidth=1)
        ax.scatter(x,y,s=32,color='#0072B2',alpha=.8)
        ax.set(xlim=(0,end),ylim=(0,end),xlabel='单起点2轮 BER（%）',ylabel='5起点各2轮 BER（%）',title=names[name])
        ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('每个点代表一个传播环境：对角线下方表示多起点更好',fontsize=13)
    fig.text(.5,.012,'固定24个旧测试环境，各环境合并3个载频；未按结果筛选案例。',ha='center')
    save(fig,'pilot_paired_environments',bottom=.17)
    paths=[base/'baseline_results/20260925_response_control',base/'baseline_results/20260925_full_baselines/real_response_cnn']
    fig,ax=plt.subplots(figsize=(8,4.8));training_sources={}
    for path,label,color in zip(paths,['复响应CNN：17,540参数','实数响应CNN：17,524参数'],['#0072B2','#D55E00']):
        h=json.loads((path/'history.json').read_text());complete=json.loads((path/'complete.json').read_text())
        if len(h)!=40 or complete['epochs']!=40:raise ValueError('容量匹配训练未完成40轮。')
        ax.plot([v['epoch'] for v in h],[v['loss'] for v in h],label=label,color=color,linewidth=2)
        training_sources[str(path/'history.json')]=sha256(path/'history.json')
    ax.set(xlabel='训练轮数',ylabel='相对复响应均方误差',xlim=(1,40));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('相同864训练环境、相同目标和预算的表示方式对照',fontsize=13)
    fig.text(.5,.012,'这是训练损失曲线；实数CNN的完整接收评分仍待完成。',ha='center')
    save(fig,'matched_cnn_training')
    write_json(output/'build.json',dict(status='complete',scope='diagnostic figures only, not final website delivery',
        pilot_summary_sha256=sha256(pilot/'summary.json'),training_sources=training_sources,
        files={f.name:sha256(f) for f in files},font_name=font_name,font_path=font_path,at=now()))
    print(json.dumps(dict(status='rendered',figures=3,files=len(files))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.project,a.output)
