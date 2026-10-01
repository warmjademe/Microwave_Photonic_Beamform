"""从完整审计结果机械生成论文数值表；不选择模型、不修改评分。"""
import argparse
import hashlib
import json
from pathlib import Path
import socket

METHODS=['initial_select64','codebook','coordinate','spsa','done','de',
         'mlp_feedback64','dnn_feedback64','cnn_feedback64','rescnn_feedback64',
         'transformer_feedback64','complex_cnn_feedback64','jct_feedback64','cnn_warm64']
LABELS=['固定扫描选择','几何码本','坐标搜索','SPSA','DONE','差分进化',
        'MLP＋反馈','DNN＋反馈','CNN＋反馈','ResCNN＋反馈','Transformer＋反馈',
        '复数 CNN＋反馈','JCT＋反馈','本文方法']
FIELDS=['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db']
METRICS=['BER','SER','BLER','RMS EVM','配对 SNR']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(value,field,signed=False):
    scale=100 if field in FIELDS[:3] else 1
    return format(scale*value,'+.4f' if signed else '.4f')
def interval(row):return '$['+','.join(number(v,row['metric']) for v in row['ci95'])+']$'
def table(caption,label,columns,header,rows):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',
        r'\toprule',header+r'\\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def main(project,output):
    if 'huashuo' not in socket.gethostname().lower():raise RuntimeError('完整统计表在华硕生成')
    root=project/'dataset_simulation/baseline_results/20260927_uniform64_all13/analysis'
    complete=read(root/'complete.json')
    for name,digest in complete['file_sha256'].items():assert sha(root/name)==digest
    rows={r['method']:r for r in read(root/'summary.json')}
    comparisons={(r['baseline'],r['metric']):r for r in read(root/'comparisons.json')}
    budget=read(root/'budget_comparisons.json')
    old=read(project/'dataset_simulation/baseline_results/20260926_final864_selected/analysis/summary.json')
    old={r['method']:r for r in old}
    output.mkdir(parents=True,exist_ok=False)
    best={f:(max if f==FIELDS[-1] else min)(r[f] for r in rows.values()) for f in FIELDS}
    content=[]
    for name,label in zip(METHODS,LABELS):
        vals=[]
        for field in FIELDS:
            value=number(rows[name][field],field)
            if abs(rows[name][field]-best[field])<1e-12:value=r'\textbf{'+value+'}'
            vals.append(value)
        content.append(' & '.join([label,*vals])+r'\\')
    (output/'rq1_main_table.tex').write_text(table(
        '统一 64 次探测下全部 13 个基线与本文方法的总体接收质量。覆盖 864 个测试环境和 17 档载频。'
        '前四项越低越好，SNR 越高越好；SNR 在均衡前计算，其余指标在均衡后计算。粗体标记最优值。',
        'tab:mainresults','lrrrrr',r'方法 & BER（\%） & SER（\%） & BLER（\%） & RMS EVM（\%） & SNR（dB）',content))
    content=[]
    for name,label in zip(METHODS[:-1],LABELS[:-1]):
        r=comparisons[name,'ber']
        content.append(' & '.join([label,'$'+number(r['estimate'],'ber',True)+'$',interval(r),
            '%d/%d/%d'%(r['wins'],r['ties'],r['losses'])])+r'\\')
    (output/'rq1_paired_table.tex').write_text(table(
        '统一 64 次探测下本文方法相对 13 个基线的配对 BER 比较。差值单位为百分点，负值表示本文方法误码率更低。'
        '区间以传播环境为单位进行分层配对重采样；胜/平/负根据每个环境的全部 17 档载频判定。',
        'tab:rq1-paired-baselines','lrrr',r'对比基线 & BER 差值 & 95\% 区间 & 胜/平/负',content))
    strongest=min(METHODS[:-1],key=lambda n:rows[n]['ber']);label=LABELS[METHODS.index(strongest)]
    content=[]
    for field,metric in zip(FIELDS,METRICS):
        r=comparisons[strongest,field]
        content.append(' & '.join([metric,'$'+number(r['estimate'],field,True)+'$',interval(r)])+r'\\')
    (output/'rq1_effects_table.tex').write_text(table(
        '本文方法减去'+label+'的指标差值与环境配对 95\\% 区间，两者均使用 64 次探测。'
        '该基线在 13 个基线中具有最低总体 BER。百分比指标的差值单位为百分点，SNR 差值单位为 dB。',
        'tab:rq1-metric-effects','lrr',r'指标 & 差值 & 95\% 区间',content))
    budget_rows=[r for r in budget if r['metric']=='ber']
    content=[]
    for r in budget_rows:
        before,after=r['method16'],r['method64']
        label=LABELS[METHODS.index(after)].replace('＋反馈','')
        if before=='cnn__joint_full_noise':label='本文 A+B 到 A 加反馈'
        elif before=='complex_response_cnn':label='本文 A 到 A 加反馈'
        content.append(' & '.join([label,number(old[before]['ber'],'ber'),number(rows[after]['ber'],'ber'),
            '$'+number(r['estimate'],'ber',True)+'$',interval(r)])+r'\\')
    (output/'rq3_budget_table.tex').write_text(table(
        '相同测试环境下由 16 次流程到 64 次流程的 BER 变化。冻结模型权重保持不变；'
        '预算增加伴随候选反馈或扫描范围扩展。A+B 到 A 加反馈还改变了联合校正开关。'
        '差值为 64 次减去 16 次，单位为百分点。',
        'tab:budget-comparison','lrrrr',r'控制方法 & 16 次 BER（\%） & 64 次 BER（\%） & 差值 & 95\% 区间',content))
    facts=dict(strongest_baseline=strongest,strongest_label=LABELS[METHODS.index(strongest)],
        primary_rank_by_ber=1+sum(r['ber']<rows['cnn_warm64']['ber'] for r in rows.values()),
        primary_best_metrics=[f for f in FIELDS if abs(rows['cnn_warm64'][f]-best[f])<1e-12],
        strongest_comparisons={f:comparisons[strongest,f] for f in FIELDS},
        relative_ber_reduction_percent=100*(rows[strongest]['ber']-rows['cnn_warm64']['ber'])/rows[strongest]['ber'],
        baseline_network_ber_range=[min(rows[n]['ber'] for n in METHODS[6:13]),max(rows[n]['ber'] for n in METHODS[6:13])],
        analysis_complete_sha256=sha(root/'complete.json'))
    (output/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2))
    (output/'manifest.json').write_text(json.dumps(dict(source=sha(root/'complete.json'),builder=sha(Path(__file__)),
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(facts,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.project,a.output)
