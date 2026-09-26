"""全部864环境完成后，独立核查原始控制和指标，生成最终排名及消融。"""
import json
from pathlib import Path
import numpy as np
from study_final864.run import validate,read,comparisons,BASELINES,PRIMARY,ABLATIONS
from study_full_baselines.common import require_host,sha256,write_json,now,atomic_npz
from study_full_baselines.confirmation_batch import committed_environment,fingerprint
from study_full_baselines.analyze_confirmation import check_values,grouped
from study_full_baselines.paired_statistics import FIELDS,sufficient,metrics,bootstrap,contrast,bh_adjust,self_check
from study_full_baselines.audit_results import statistics
from study_full_baselines.analyze_reception import clean_json


def run(frozen,folder,output):
    require_host();self_check();identity=validate(frozen);complete=read(folder/'complete.json')
    if (read(folder/'protocol.json')!=identity or complete['environments']!=864
            or complete['identity_sha256']!=fingerprint(identity)
            or complete['records_sha256']!=sha256(folder/'records.json')
            or complete['cases']!=308448 or not complete['final_confirmation']):
        raise ValueError('最终记录不完整或身份不符')
    if output.exists():
        if (output/'complete.json').exists():
            old=read(output/'complete.json')
            if old['evaluation_complete_sha256']==sha256(folder/'complete.json'):
                for name,digest in old['file_sha256'].items():
                    if sha256(output/name)!=digest:raise ValueError('已有分析改变')
                return
        raise FileExistsError('保留未完成的分析现场')
    with np.load(Path(identity['runtime_bundle'])/'public.npz') as f:
        public={k:f[k].copy() for k in f.files}
    values=[];records=[];files={};traces=0
    for row in identity['rows']:
        record=committed_environment(folder,row,identity)
        if record is None:raise ValueError('环境未完成')
        records.append(record);current=[]
        for item in record['carriers']:
            with np.load(folder/item['path'],allow_pickle=False) as f:
                arrays={k:f[k].copy() for k in f.files}
            traces+=check_values(arrays,item['carrier_ghz'],identity,public)
            current.append(arrays['metrics']);files[item['path']]=item['sha256']
        values.append(current)
    if records!=read(folder/'records.json'):raise ValueError('环境索引不同')
    values=np.asarray(values);s=sufficient(values);point=metrics(s.sum(0));names=identity['methods']
    recomputed=0
    for i,name in enumerate(names):
        direct=statistics(values[:,:,i],name)
        for j,field in enumerate(FIELDS):
            if field in direct:
                np.testing.assert_allclose(point[i,j],direct[field],rtol=1e-12,atol=0,equal_nan=True)
                recomputed+=1
    draws=bootstrap(s,10000,20260926,strata=[r['joint_stratum_id'] for r in identity['rows']])
    contrasts=[]
    for item in identity['statistics']['comparisons']:
        for field in identity['statistics']['metrics']:
            contrasts.append(dict(**item,**contrast(s,draws,item['coefficients'],field)))
    for family in sorted({r['family'] for r in contrasts}):
        for field in identity['statistics']['metrics']:
            group=[r for r in contrasts if r['family']==family and r['metric']==field]
            for row,q in zip(group,bh_adjust([r['p_value'] for r in group])):row['sign_test_bh_q']=float(q)
    def role(name):
        return 'baseline' if name in BASELINES else 'primary' if name in PRIMARY else 'ablation' if name in ABLATIONS else 'reference'
    summary=[dict(method=n,role=role(n),measurement_budget=identity['measurement_budgets'].get(n),
        reference=n in ['teacher','mrc'],**{f:float(point[i,j]) for j,f in enumerate(FIELDS)}) for i,n in enumerate(names)]
    conclusions=[]
    for own in PRIMARY:
        probes=identity['measurement_budgets'][own]
        peers=[r for r in summary if r['role'] in ['baseline','primary'] and r['measurement_budget']==probes]
        ordered=sorted(peers,key=lambda r:r['ber'])
        conclusions.append(dict(method=own,measurement_budget=probes,
            rank_by_ber=next(i+1 for i,r in enumerate(ordered) if r['method']==own),
            tied_best=next(r for r in peers if r['method']==own)['ber']==ordered[0]['ber'],
            compared_methods=[r['method'] for r in ordered]))
    output.mkdir(parents=True)
    write_json(output/'summary.json',clean_json(summary));write_json(output/'groups.json',clean_json(grouped(values,identity)))
    write_json(output/'comparisons.json',clean_json(contrasts));write_json(output/'conclusions.json',conclusions)
    atomic_npz(output/'environment_metrics.npz',metrics=values,methods=np.asarray(names),
               environment_ids=np.asarray([r['environment_id'] for r in identity['rows']]))
    write_json(output/'audit.json',dict(status='passed',at=now(),scope='fresh_confirmation',
        final_confirmation=True,environments=864,carriers=17,methods=21,cases=308448,
        control_codes_checked=308448*128,feedback_trace_rows_checked=traces,
        independently_recomputed_aggregate_numbers=recomputed,raw_file_sha256=files,
        source_sha256=identity['source_sha256'],protocol_sha256=sha256(folder/'protocol.json'),
        records_sha256=sha256(folder/'records.json'),runtime_manifest_sha256=identity['runtime_manifest_sha256']))
    lines=['# 最终864环境结果','',
        '固定3456训练、216验证、864测试；每环境17载频。主指标BER，同预算比较。',
        'EVM由平均NMSE开方；配对SNR由总信号功率/总噪声功率取dB。教师与MRC单列。','']
    for item in conclusions:
        lines.append('%s：%d次测量，BER排名%d/%d。'%(item['method'],item['measurement_budget'],item['rank_by_ber'],len(item['compared_methods'])))
    for title,roles in [('正式基线与主方法',['baseline','primary']),('必要消融',['ablation']),('额外信息参考',['reference'])]:
        lines+=['','## '+title,'','| 方法 | 测量次数 | BER % | EVM % | 配对SNR dB |','|---|---:|---:|---:|---:|']
        for row in sorted([r for r in summary if r['role'] in roles],key=lambda r:(r['measurement_budget'] or 0,r['ber'])):
            snr='不适用' if not np.isfinite(row['paired_output_snr_db']) else '%.4f'%row['paired_output_snr_db']
            lines.append('| %s | %s | %.4f | %.4f | %s |'%(row['method'],row['measurement_budget'],100*row['ber'],row['rms_evm_percent'],snr))
    lines+=['','## 组件效果','',
        '差值为有组件版本减去对应对照；BER负值表示改善。反馈消融同时改变测量次数。',
        '区间为层内配对bootstrap 95%区间，BH q对应环境胜负符号检验；一个训练seed不刻画重新训练的不确定性。','',
        '| 比较 | BER差/百分点 | 95%区间 | 胜/平/负 | BH q |','|---|---:|---|---|---:|']
    for r in contrasts:
        if r['family'].startswith('components') and r['metric']=='ber':
            lines.append('| %s | %.4f | [%.4f, %.4f] | %d/%d/%d | %.6g |'%(r['id'],100*r['estimate'],100*r['ci95'][0],100*r['ci95'][1],r['wins'],r['ties'],r['losses'],r['sign_test_bh_q']))
    (output/'结果说明.md').write_text('\n'.join(lines)+'\n')
    validate(frozen)
    files={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'complete.json',dict(status='complete_final_analysis',at=now(),comparisons=len(contrasts),
        evaluation_complete_sha256=sha256(folder/'complete.json'),file_sha256=files,
        final_confirmation=True,overall_research_complete=False))
