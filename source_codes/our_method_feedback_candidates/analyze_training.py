"""训练内候选诊断的环境配对区间与选择轨迹；不作为独立泛化证据。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.paired_statistics import sufficient,metrics,bootstrap,contrast,bh_adjust,FIELDS


def run(diagnostic,output):
    require_host()
    if output.exists():raise FileExistsError('训练统计不覆盖。')
    protocol=json.loads((diagnostic/'protocol.json').read_text())
    complete=json.loads((diagnostic/'complete.json').read_text())
    if protocol['test_used'] or complete['status']!='complete':raise ValueError('不是已完成训练内诊断。')
    if sha256(diagnostic/'all_metrics.npz')!=complete['results_sha256']:raise ValueError('指标数组改变。')
    verify_sources(protocol['source_sha256']);cases=protocol['cases']
    ids=[]
    for case in cases:
        if case['environment_id'] not in ids:ids.append(case['environment_id'])
    if len(ids)!=24 or len(cases)!=72:raise ValueError('原诊断环境数量不同。')
    if any(cases[3*i+j]['environment_id']!=eid or cases[3*i+j]['carrier_ghz']!=[4,12,20][j]
           for i,eid in enumerate(ids) for j in range(3)):
        raise ValueError('环境/载频顺序不同。')
    with np.load(diagnostic/'all_metrics.npz') as f:
        a=f['metrics'].reshape(24,3,8,10);traces=f['trace_control_code'];scores=f['trace_scores']
    names=protocol['methods'];s=sufficient(a);draws=bootstrap(s,10000,20260926)
    point=metrics(s.sum(0));summary=json.loads((diagnostic/'summary.json').read_text())
    for mi,name in enumerate(names):
        old=next(r for r in summary['records'] if r['method']==name)
        for field in ['ber','ser','block_error_rate','rms_evm_percent','mean_nmse','paired_output_snr_db']:
            if not np.isclose(point[mi,FIELDS.index(field)],old[field],rtol=1e-12,atol=0):
                raise ValueError('训练诊断汇总不能从环境块重算。')
    pairs=[('cnn_warm64','codebook64'),('covariance_warm64','codebook64'),
           ('cnn_warm64','covariance_warm64'),('cnn_candidates64','cnn_warm64'),
           ('covariance_candidates64','covariance_warm64'),('cnn_warm64','cnn_base16')]
    compared=[]
    for left,right in pairs:
        c=np.zeros(len(names));c[names.index(left)]=1;c[names.index(right)]=-1
        for field in ['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db']:
            compared.append(dict(left=left,right=right,
                budget_scope='different feedback budgets' if right.endswith('16') else 'same64feedback',
                **contrast(s,draws,c,field)))
    q=bh_adjust([r['p_value'] for r in compared])
    for r,p in zip(compared,q):r['p_bh_fdr_training_family']=float(p)
    selections=[];chosen=np.argmax(scores,axis=2)
    for mi,name in enumerate(names[:5]):
        unique=[];response_selected=0
        for ci in range(72):
            marker=json.loads((diagnostic/'records'/('%03d.json'%ci)).read_text())
            path=diagnostic/'records'/('%03d.npz'%ci)
            if sha256(path)!=marker['sha256']:raise ValueError('反馈轨迹记录改变。')
            info=marker['candidate_details'][mi]
            unique.append(info.get('unique_response_candidates',0))
            # 新响应候选（若存在）顺序位于初始16次之后，之后才进行几何补齐。
            if 16<=chosen[ci,mi]<16+unique[-1]:response_selected+=1
        selections.append(dict(method=name,cases=72,selected_initial_probe=int(np.sum(chosen[:,mi]<16)),
            selected_response_candidate=response_selected,
            selected_geometric=int(np.sum(chosen[:,mi]>=16))-response_selected,
            mean_unique_response_candidates=float(np.mean(unique)),
            attribution='Selection categories are descriptive, not a causal estimate of model benefit.'))
    output.mkdir(parents=True)
    write_json(output/'results.json',dict(status='complete',scope='training-only design evidence; not independent test',
        environments=24,carriers_per_environment=3,bootstrap_repetitions=10000,
        ci_scope='Variation across selected training environments with fixed model; no generalization or retraining claim',
        comparisons=compared,selections=selections,at=now()))
    atomic_npz(output/'paired_statistics.npz',sufficient=s,bootstrap_metrics=draws,
        environment_ids=np.asarray(ids),methods=np.asarray(names),fields=np.asarray(FIELDS))
    write_json(output/'complete.json',dict(status='complete',diagnostic_results_sha256=complete['results_sha256'],
        source_sha256=source_record(['our_method_feedback_candidates/analyze_training.py',
                                    'study_full_baselines/paired_statistics.py']),
        file_sha256={n:sha256(output/n) for n in ['results.json','paired_statistics.npz']},at=now()))
    print(json.dumps(dict(ber_comparisons=[r for r in compared if r['metric']=='ber'],selections=selections)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.diagnostic,a.output)
