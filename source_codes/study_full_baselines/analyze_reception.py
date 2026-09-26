"""六条流水线完整审计后生成可追溯统计表；拒绝用未完成部分填最终排名。"""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.audit_reception_v2 import collect,PHASES
from study_full_baselines.paired_statistics import (FIELDS, sufficient, metrics,
    bootstrap,contrast,bh_adjust,self_check)


def canonical(name):
    if name=='covariance__base_2sweeps':return 'covariance_response'
    if name=='cnn__base_2sweeps':return 'complex_response_cnn'
    return name.removesuffix('__base_2sweeps')


def catalog_from_arrays(phases,rows):
    """相同控制器跨流水线只占一行，合并前逐档位和指标核对。"""
    names=[];columns=[];codes=[];catalog=[];lookup={}
    ids=[r['environment_id'] for r in rows]
    for phase in PHASES:
        current=phases[phase]
        if current is None or current['ids']!=ids:
            raise ValueError('统计需要六阶段全部环境且顺序一致。')
        for mi,name in enumerate(current['methods']):
            key=canonical(name);a=current['values'][:,:,mi,:10];u=current['controls'][:,:,mi]
            if key in lookup:
                previous=lookup[key]
                if (not np.array_equal(u,codes[previous]) or
                        not np.allclose(a,columns[previous],rtol=1e-12,atol=0,equal_nan=True)):
                    raise ValueError('重复方法的控制或接收指标不同：'+key)
                catalog[previous]['sources'].append(dict(phase=phase,method=name))
                continue
            lookup[key]=len(names);names.append(key);columns.append(a);codes.append(u)
            call_index=current['metric_order'].index('feedback_calls')
            calls=current['values'][:,:,mi,call_index]
            match=re.search(r'_n(\d+)',key)
            catalog.append(dict(id=key,sources=[dict(phase=phase,method=name)],
                feedback_calls=float(calls.mean()) if np.isfinite(calls).all() else None,
                privileged=key in ['teacher','mrc'],photonic_hardware=key!='mrc',
                training_environments=(None if key in ONLINE_CLASSIC+['teacher','mrc','public16','ridge_response']
                    else int(match.group(1)) if match else 864),
                training_count_scope='None means no fitted training model for this method',
                timing='Decoder-only score timing is not used as end-to-end inference time'))
    return names,np.stack(columns,axis=2),catalog


def clean_json(value):
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_csv(path,records):
    keys=list(dict.fromkeys(k for record in records for k in record))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                             for k,v in clean_json(record).items()})


def planned_comparisons(plan,names,s,draws):
    rows=[];identifiers=[p['id'] for p in plan['comparisons']]
    if len(set(identifiers))!=len(identifiers):raise ValueError('重复的对比ID。')
    for spec in plan['comparisons']:
        unknown=set(spec['terms'])-set(names)
        if unknown:raise ValueError('比较清单中的方法缺失：'+str(unknown))
        coefficients=np.asarray([spec['terms'].get(name,0) for name in names],float)
        for field in plan['metrics']:
            if field=='paired_output_snr_db' and 'mrc' in spec['terms']:
                rows.append(dict(comparison=spec['id'],family=spec['family'],metric=field,
                    status='not_applicable',reason='MRC is a different digital receiver with no photonic output power',
                    terms=spec['terms']))
                continue
            record=contrast(s,draws,coefficients,field)
            rows.append(dict(comparison=spec['id'],family=spec['family'],terms=spec['terms'],
                interpretation=spec['meaning'],status='computed',**record))
    for family in sorted({r['family'] for r in rows}):
        members=[r for r in rows if r['family']==family and r['status']=='computed']
        adjusted=bh_adjust([r['p_value'] for r in members])
        for r,q in zip(members,adjusted):
            r['p_bh_fdr']=float(q);r['family_tests']=len(members)
    return rows


def compute(values,names,catalog,rows,plan):
    s=sufficient(values)
    if plan['stratified']:
        # 本入口的216旧环境每个层只有一个成员，不能伪造层内方差。
        raise ValueError('本轮旧216使用普通环境块bootstrap。新确认另行冻结。')
    draws=bootstrap(s,plan['bootstrap_repetitions'],plan['bootstrap_seed'])
    point=metrics(s.sum(axis=0));overall=[]
    for mi,name in enumerate(names):
        record=dict(method=name,independent_environments=len(rows),carriers_per_environment=17,
                    **{k:float(point[mi,fi]) for fi,k in enumerate(FIELDS)})
        for fi,field in enumerate(FIELDS):
            column=draws[:,mi,fi]
            record[field+'_ci95']=np.quantile(column,[.025,.975]).tolist() if np.isfinite(column).all() else None
        record.update(catalog[mi]);overall.append(record)
    comparisons=planned_comparisons(plan,names,s,draws)
    # 单环境记录用于显示分布与核对，不将频点拆成独立样本。
    env_metrics=metrics(s)
    return overall,comparisons,dict(sufficient=s,bootstrap_metrics=draws,
        environment_metrics=env_metrics,methods=np.asarray(names),fields=np.asarray(FIELDS),
        environment_ids=np.asarray([r['environment_id'] for r in rows]))


def run(data,study,output,plan_path):
    require_host()
    if output.exists():raise FileExistsError('统计产物另建版本，不覆盖。')
    plan=json.loads(plan_path.read_text())
    if plan['schema']!='old216-exploratory-comparisons-v1':raise ValueError('比较清单不是本轮探索协议。')
    audit,phases,rows=collect(data,study,False)
    if audit['status']!='passed':raise ValueError('完整接收审计尚未通过。')
    names,values,catalog=catalog_from_arrays(phases,rows)
    sources=source_record(['study_full_baselines/analyze_reception.py',
        'study_full_baselines/audit_reception_v2.py','study_full_baselines/audit_results.py',
        'study_full_baselines/paired_statistics.py','study_full_baselines/ANALYSIS_PROTOCOL.md'])
    overall,comparisons,arrays=compute(values,names,catalog,rows,plan)
    output.mkdir(parents=True)
    write_json(output/'audit.json',audit)
    write_json(output/'plan.json',plan)
    atomic_npz(output/'statistics.npz',**arrays)
    write_json(output/'overall.json',clean_json(dict(status='complete',scope=plan['scope'],
        records=overall,uncertainty='10000 paired environment bootstrap; one fixed training seed')))
    write_json(output/'comparisons.json',clean_json(dict(status='complete',scope=plan['scope'],records=comparisons)))
    write_csv(output/'overall.csv',overall);write_csv(output/'comparisons.csv',comparisons)
    verify_sources(sources)
    write_json(output/'complete.json',dict(status='complete',at=now(),independent_environments=len(rows),
        methods=len(names),planned_comparisons=len(plan['comparisons']),scope=plan['scope'],
        data_manifest_sha256=sha256(data/'manifest.json'),plan_sha256=sha256(plan_path),
        source_sha256=sources,file_sha256={name:sha256(output/name) for name in
            ['audit.json','plan.json','statistics.npz','overall.json','comparisons.json','overall.csv','comparisons.csv']}))
    print(json.dumps(dict(status='complete',methods=len(names),comparisons=len(comparisons),scope=plan['scope'])),flush=True)


def preflight(data,study,plan_path,output):
    """用已提交环境核对真实接收聚合；只做完整性检查，不报告局部优胜排名。"""
    require_host()
    if output.exists():raise FileExistsError('前置验证不覆盖。')
    report,phases,rows=collect(data,study,True)
    plan=json.loads(plan_path.read_text());checks=self_check()
    checked=[]
    from study_full_baselines.audit_results import statistics
    for phase,current in phases.items():
        if current is None:continue
        s=sufficient(current['values']);calculated=metrics(s.sum(0))
        for mi,name in enumerate(current['methods']):
            direct=statistics(current['values'][:,:,mi],name)
            for fi,field in enumerate(FIELDS):
                if field not in direct:continue
                if not np.isclose(calculated[mi,fi],direct[field],rtol=1e-12,atol=0):
                    raise ValueError('充分统计聚合与原始计数重算不同。')
        checked.append(dict(phase=phase,completed_environments=len(current['indices']),
                            methods=len(current['methods'])))
    if not checked:raise ValueError('没有可核查的实际评分。')
    # 即使尚未运行全部评测，也从冻结的方法清单核对计划中的每个名称。
    from our_method_quality_rank.common import METHODS as QUALITY_METHODS
    known=set(ONLINE_CLASSIC+['teacher','mrc','mlp']+DEEP_METHODS)
    known.update('quality_'+m for m in QUALITY_METHODS)
    known.update(['ridge_response','covariance_response','complex_response_cnn','global_prior',
        'frequency_prior','public16','per_tone_relative','joint_relative','joint_absolute',
        'real_response_cnn','real_response_cnn__multi_mmse'])
    known.update(a+'__'+b for a in ['cnn','covariance'] for b in ['single_10sweeps','multi_mmse'])
    for n in [216,432,1728,3456]:
        known.add('covariance_n%04d'%n)
        known.update('response_n%04d_%s'%(n,s) for s in ['fixed_epochs','equal_updates'])
    for spec in plan['comparisons']:
        if set(spec['terms'])-known:raise ValueError('比较清单含不存在的方法：'+spec['id'])
        if not np.isclose(sum(spec['terms'].values()),0):raise ValueError('对比系数和非零。')
    write_json(output,dict(status='passed',at=now(),statistical_checks=checks,
        completed_records_aggregate_checks=checked,planned_comparisons=len(plan['comparisons']),
        partial_scores_not_used_for_ranking=True,plan_sha256=sha256(plan_path),
        source_sha256=source_record(['study_full_baselines/analyze_reception.py',
                                    'study_full_baselines/paired_statistics.py'])))
    print(json.dumps(dict(status='passed',checked=checked,comparisons=len(plan['comparisons']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','study','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path(__file__).with_name('EXPLORATORY_COMPARISONS.json'))
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.preflight:preflight(a.data,a.study,a.plan,a.output)
    else:run(a.data,a.study,a.output,a.plan)
