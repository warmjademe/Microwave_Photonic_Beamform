"""仅在完整64次反馈结果已统计后绘图，不对未完成子集排名。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, write_json, now, source_record, SOURCE


def run(project, output):
    require_host()
    if output.exists(): raise FileExistsError('不覆盖已完成图表。')
    root=project/'dataset_simulation/baseline_results/20260925_full_baselines/analysis_feedback_warm'
    complete=json.loads((root/'complete.json').read_text())
    if complete['status']!='complete' or complete['metric_tests']!=50:
        raise ValueError('完整反馈统计尚未完成。')
    for name,digest in complete['file_sha256'].items():
        if sha256(root/name)!=digest: raise ValueError('反馈统计文件改变。')
    records=json.loads((root/'overall.json').read_text())['records']
    comparisons=json.loads((root/'comparisons.json').read_text())['records']
    origins=json.loads((root/'selection_origins.json').read_text())['records']
    if len(origins)!=216*17*2: raise ValueError('控制来源没有覆盖全部环境。')
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    figures=[]
    def save(fig,stem):
        for ext in ['png','pdf','svg']:
            filename=stem+'.'+ext;fig.savefig(output/filename,dpi=200,bbox_inches='tight');figures.append(filename)
        plt.close(fig)
    names=['codebook','coordinate','spsa','done','de','covariance_warm64','cnn_warm64']
    labels=['几何码本','坐标搜索','SPSA','DONE','差分进化','传统估计＋反馈','复响应CNN＋反馈']
    fields=['ber','rms_evm_percent','paired_output_snr_db'];scales=[100,1,1]
    titles=['BER（%）越低越好','RMS EVM（%）越低越好','输出SNR（dB）越高越好']
    fig,axes=plt.subplots(1,3,figsize=(15,5.8))
    for ax,field,scale,title in zip(axes,fields,scales,titles):
        y=[next(r for r in records if r['method']==n)[field]*scale for n in names]
        ax.barh(range(len(names)),y,color=['#999999']*5+['#009E73','#0072B2'])
        ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.set_title(title,fontsize=11)
        ax.set_xlim(0,max(y)*1.22);ax.grid(axis='x',alpha=.18)
        for i,value in enumerate(y):ax.text(value+max(y)*.015,i,'%.3f'%value,va='center',fontsize=9)
    fig.suptitle('相同64次测量预算：完整216环境×17载频',fontsize=14)
    fig.text(.5,.025,'响应候选使用864环境训练、seed0；误码率为主要指标，全部方法和无改善结果保留。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.92));save(fig,'feedback64_matched_budget')
    chosen=['warm_cnn_vs_codebook','warm_cnn_vs_covariance','warm_covariance_vs_codebook']
    labels=['复响应CNN反馈−码本','复响应CNN反馈−传统估计反馈','传统估计反馈−码本']
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharey=True)
    for ax,field,scale,title in zip(axes,fields,scales,['BER差值（百分点）','EVM差值（百分点）','输出SNR差值（dB）']):
        for i,key in enumerate(chosen):
            r=next(r for r in comparisons if r['comparison']==key and r['metric']==field)
            lo,hi=np.asarray(r['ci95'])*scale
            ax.plot([lo,hi],[i,i],color='#0072B2',lw=2);ax.scatter(r['estimate']*scale,i,color='#0072B2')
        ax.set_yticks(range(3),labels);ax.axvline(0,color='#555',ls='--');ax.grid(axis='x',alpha=.18)
        ax.set_xlabel(title+'\n'+('负值有利于前一方法' if field!='paired_output_snr_db' else '正值有利于前一方法'))
    axes[0].invert_yaxis();fig.suptitle('相同反馈预算的差值与95%环境配对区间',fontsize=14)
    fig.text(.5,.025,'10,000次环境块bootstrap；完整50项检验统一BH–FDR；原始与校正p值见CSV。旧测试探索。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.92));save(fig,'feedback64_paired_effects')
    counts={name:Counter(r['selected_source'] for r in origins if r['method']==name)
            for name in ['covariance_warm64','cnn_warm64']}
    fig,ax=plt.subplots(figsize=(9,4.8));bottom=np.zeros(2)
    for key,label,color in [('initial','最初16套','#999999'),('geometric','追加几何设置','#E69F00'),('response','估计器提出的设置','#0072B2')]:
        values=np.asarray([counts[n][key]/(216*17)*100 for n in counts])
        ax.barh([0,1],values,left=bottom,label=label,color=color)
        for i,v in enumerate(values):
            if v>3:ax.text(bottom[i]+v/2,i,'%.1f%%'%v,ha='center',va='center',color='white' if key=='response' else 'black')
        bottom+=values
    ax.set_yticks([0,1],['传统估计＋反馈','复响应CNN＋反馈']);ax.set_xlim(0,100)
    ax.set_xlabel('最终采用设置的来源占比（%）');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,fontsize=9)
    ax.set_title('选中了哪一类控制设置？',fontsize=14)
    fig.text(.5,.025,'来源占比是描述性统计；被选择的频率不等于该组件的因果贡献。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.13,1,.95));save(fig,'feedback64_selection_origins')
    write_json(output/'origin_counts.json',{n:dict(v) for n,v in counts.items()})
    source=source_record(['study_full_baselines/render_completed_feedback.py'])
    for name in source:
        dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dst)
    write_json(output/'complete.json',dict(status='complete_exploratory_figures',at=now(),
        analysis_complete_sha256=sha256(root/'complete.json'),source_sha256=source,
        files={n:sha256(output/n) for n in figures+['origin_counts.json']},visual_review='pending',full_experiment_complete=False))
    print(json.dumps(dict(status='complete',figures=3,methods=7,scope='old216 exploratory only')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['project','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.project.resolve(),a.output.resolve())
