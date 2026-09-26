"""训练内候选实验的中文图：质量、配对差值区间与实际选择来源。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,sha256,source_record,write_json,now


def run(project,output,overwrite=False):
    require_host()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font_path);font_name=font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,
        'pdf.fonttype':42,'svg.fonttype':'path','font.size':10})
    base=project/'dataset_simulation/diagnostics'
    diagnostic=base/'20260926_feedback_candidates_train';statistics=base/'20260926_feedback_candidates_training_statistics'
    meta=json.loads((statistics/'complete.json').read_text())
    for name,digest in meta['file_sha256'].items():
        if sha256(statistics/name)!=digest:raise ValueError('训练统计产物改变。')
    if sha256(diagnostic/'all_metrics.npz')!=meta['diagnostic_results_sha256']:
        raise ValueError('训练评分与统计来源不同。')
    summary=json.loads((diagnostic/'summary.json').read_text())
    stats=json.loads((statistics/'results.json').read_text())
    if output.exists() and not overwrite:raise FileExistsError('图像已存在；复核后可显式重绘。')
    if output.exists() and not (output/'build.json').exists():raise ValueError('不能覆盖未知目录。')
    output.mkdir(parents=True,exist_ok=True);files=[]
    labels={'codebook64':'几何码本','covariance_warm64':'协方差估计＋码本',
        'cnn_warm64':'复响应 CNN＋码本','covariance_candidates64':'协方差估计：多候选',
        'cnn_candidates64':'复响应 CNN：多候选','cnn_base16':'复响应 CNN 基础控制（16次）'}
    order=['codebook64','covariance_warm64','cnn_warm64','covariance_candidates64','cnn_candidates64']
    lookup={r['method']:r for r in summary['records']}
    colors=['#777777','#D55E00','#0072B2','#E69F00','#56B4E9']
    footer='训练内诊断：24个环境 × 4/12/20 GHz；固定模型。新留出确认尚未开始。'
    def save(fig,name,bottom=.09):
        fig.tight_layout(rect=(0,bottom,1,.94))
        for ext in ['png','pdf','svg']:
            path=output/(name+'.'+ext);fig.savefig(path,dpi=180,bbox_inches='tight');files.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2),sharey=True)
    for ax,key,mult,title in zip(axes,['ber','rms_evm_percent','paired_output_snr_db'],[100,1,1],
                               ['BER（%）越低越好','RMS EVM（%）越低越好','配对输出 SNR（dB）越高越好']):
        values=np.asarray([lookup[n][key]*mult for n in order])
        ax.barh(np.arange(5),values,color=colors);ax.set_xlabel(title);ax.set_xlim(0,float(values.max())*1.23)
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        for i,value in enumerate(values):ax.text(value+values.max()*.015,i,'%.2f'%value,va='center')
    axes[0].set_yticks(np.arange(5),[labels[n] for n in order]);axes[0].invert_yaxis()
    fig.suptitle('相同64次测量下的候选策略比较',fontsize=14)
    fig.text(.5,.025,footer,ha='center',fontsize=10);save(fig,'feedback_train_quality')
    pairs=[r for r in stats['comparisons'] if r['metric']=='ber']
    fig,ax=plt.subplots(figsize=(12.5,5.6))
    for i,r in enumerate(pairs):
        point=100*r['estimate'];lo,hi=np.asarray(r['ci95'])*100
        color='#777777' if r['budget_scope'].startswith('different') else '#0072B2'
        ax.plot([lo,hi],[i,i],color=color,linewidth=2);ax.scatter([point],[i],color=color,s=45,zorder=3)
        ax.text(6.2,i,'%d / %d / %d'%(r['wins'],r['ties'],r['losses']),va='center',fontsize=10)
    ax.axvline(0,color='#444444',linestyle='--',linewidth=1)
    ax.set_yticks(np.arange(len(pairs)),[labels[r['left']]+' − '+labels[r['right']] for r in pairs]);ax.invert_yaxis()
    ax.set_xlim(-5,8.5);ax.set_xlabel('BER 差值（百分点）；负值表示左侧方法更好')
    ax.grid(axis='x',alpha=.18);ax.text(6.2,-.72,'更好 / 持平 / 更差',fontsize=10)
    fig.suptitle('按环境配对：单候选组合的改善区间跨过零',fontsize=14)
    fig.text(.5,.035,'横线：10,000次环境块 bootstrap 的95%区间；灰色行同时增加了测量次数。',ha='center',fontsize=10)
    fig.text(.5,.006,footer,ha='center',fontsize=10);save(fig,'feedback_train_paired',bottom=.14)
    fig,ax=plt.subplots(figsize=(10,4.8));selection={r['method']:r for r in stats['selections']}
    left=np.zeros(5)
    for key,label,color in [('selected_initial_probe','原16套探测','#999999'),
        ('selected_response_candidate','估计响应提出的候选','#0072B2'),('selected_geometric','额外几何码本','#E69F00')]:
        values=np.asarray([selection[n][key] for n in order])
        ax.barh(np.arange(5),values,left=left,color=color,label=label)
        for i,v in enumerate(values):
            if v:ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white' if key=='selected_response_candidate' else '#111111')
        left+=values
    ax.set_yticks(np.arange(5),[labels[n] for n in order]);ax.invert_yaxis();ax.set_xlim(0,72)
    ax.set_xlabel('最终选中设置的案例数（每种方法72个案例）');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3)
    fig.suptitle('最终使用了哪一类器件设置',fontsize=14)
    fig.text(.5,.025,'候选被选中的次数用于解释选择过程；质量由后续数据帧的接收指标评价。',ha='center')
    save(fig,'feedback_train_selection',bottom=.14)
    write_json(output/'build.json',dict(status='rendered',scope='training diagnostic figures; website not published',
        source_sha256=source_record(['our_method_feedback_candidates/render_training.py']),
        diagnostic_summary_sha256=sha256(diagnostic/'summary.json'),statistics_sha256=sha256(statistics/'results.json'),
        font_name=font_name,font_path=font_path,files={p.name:sha256(p) for p in files},at=now()))
    print(json.dumps(dict(status='rendered',figures=3,files=len(files))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    run(a.project,a.output,a.overwrite)
