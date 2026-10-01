"""合并补测与旧 64 次结果，重新计算 13 项同预算比较的多重校正。"""
import argparse
import json
import os
from pathlib import Path
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import (require_host,sha256,verify_sources,
    write_json,atomic_npz,now)
from study_full_baselines.confirmation_batch import committed_environment,fingerprint
from study_full_baselines.paired_statistics import (FIELDS,sufficient,metrics,bootstrap,
    contrast,bh_adjust,self_check)
from study_full_baselines.analyze_confirmation import grouped
from study_uniform64.controllers import DIRECT,NEW_METHODS,REUSED_METHODS,BASELINES64,audit
from study_uniform64.run import read,load_original


def run(folder,output):
    require_host();self_check()
    if output.exists():raise FileExistsError(output)
    identity=read(folder/'protocol.json');done=read(folder/'complete.json')
    verify_sources(identity['source_sha256'])
    if (done['status']!='complete' or done['environments']!=864 or done['cases']!=864*17*8
            or done['identity_sha256']!=fingerprint(identity)
            or done['records_sha256']!=sha256(folder/'records.json')):
        raise ValueError('补测尚未完整提交')
    with np.load(Path(identity['runtime_bundle'])/'public.npz') as f:
        public={k:f[k].copy() for k in f.files}
    names=NEW_METHODS+REUSED_METHODS;values=[];old_values=[];records=[];trace_rows=0;files={}
    old16=['ttd_das',*DIRECT,'cnn__joint_full_noise','complex_response_cnn']
    for row in identity['rows']:
        record=committed_environment(folder,row,identity)
        if record is None:raise ValueError('缺少环境')
        records.append(record);current=[];old_current=[]
        for item in record['carriers']:
            with np.load(folder/item['path'],allow_pickle=False) as f:
                arrays={k:f[k].copy() for k in f.files}
            trace_rows+=audit(arrays,NEW_METHODS,public,item['carrier_ghz'])
            _,old,old_sha=load_original(row,item['carrier_ghz'],identity)
            if item['checks']['original_carrier_sha256']!=old_sha:raise ValueError('原结果绑定改变')
            ix=[identity['prior_methods'].index(n) for n in REUSED_METHODS]
            if not np.all(old['metrics'][ix,10]==64):raise ValueError('复用的方法未实际测量64次')
            current.append(np.concatenate([arrays['metrics'],old['metrics'][ix]]))
            old_current.append(old['metrics'][[identity['prior_methods'].index(n) for n in old16]])
            files[item['path']]=item['sha256']
        values.append(current);old_values.append(old_current)
    if records!=read(folder/'records.json'):raise ValueError('记录顺序不同')
    values=np.asarray(values);old_values=np.asarray(old_values)
    all_names=names+['old16__'+n for n in old16]
    s=sufficient(np.concatenate([values,old_values],axis=2));point=metrics(s.sum(0))
    draws=bootstrap(s,10000,20260926,strata=[r['joint_stratum_id'] for r in identity['rows']])
    contrasts=[]
    for other in BASELINES64:
        coefficients=[(1 if n=='cnn_warm64' else -1 if n==other else 0) for n in all_names]
        for field in identity['statistics']['metrics']:
            contrasts.append(dict(id='primary64_minus_'+other,baseline=other,
                family='same_budget_64_all13',**contrast(s,draws,coefficients,field)))
    for field in identity['statistics']['metrics']:
        rows=[r for r in contrasts if r['metric']==field]
        for row,q in zip(rows,bh_adjust([r['p_value'] for r in rows])):row['sign_test_bh_q']=float(q)
    budget_contrasts=[]
    for name in old16:
        new='initial_select64' if name=='ttd_das' else name+'_feedback64' if name in DIRECT else 'cnn_warm64'
        coefficients=[1 if n==new else -1 if n=='old16__'+name else 0 for n in all_names]
        for field in identity['statistics']['metrics']:
            budget_contrasts.append(dict(id='budget64_minus_16_'+name,method16=name,method64=new,
                family='budget_change_10',**contrast(s,draws,coefficients,field)))
    for field in identity['statistics']['metrics']:
        rows=[r for r in budget_contrasts if r['metric']==field]
        for row,q in zip(rows,bh_adjust([r['p_value'] for r in rows])):row['sign_test_bh_q']=float(q)
    summary=[dict(method=name,measurement_budget=64,role='primary' if name=='cnn_warm64' else 'baseline',
                  **{f:float(point[i,j]) for j,f in enumerate(FIELDS)}) for i,name in enumerate(names)]
    # 原六个方法保持数值完全一致，扩大的比较族只改变校正后的显著性值。
    prior={r['method']:r for r in read(Path(identity['source_output'])/'analysis/summary.json')}
    for row in summary:
        if row['method'] in REUSED_METHODS:
            for field in FIELDS:
                np.testing.assert_allclose(row[field],prior[row['method']][field],rtol=1e-12,atol=0)
    output.mkdir(parents=True)
    merged_identity=dict(identity,methods=names)
    write_json(output/'summary.json',summary);write_json(output/'comparisons.json',contrasts)
    write_json(output/'budget_comparisons.json',budget_contrasts)
    write_json(output/'groups.json',grouped(values,merged_identity))
    atomic_npz(output/'environment_metrics.npz',metrics=values,methods=np.asarray(names),
        environment_ids=np.asarray([r['environment_id'] for r in identity['rows']]))
    write_json(output/'audit.json',dict(status='passed',environments=864,carriers=17,
        baseline_count=13,methods=14,added_cases=864*17*8,merged_cases=864*17*14,
        added_trace_rows_checked=trace_rows,reused_results_equal=True,raw_file_sha256=files,
        protocol_sha256=sha256(folder/'protocol.json'),source_sha256=identity['source_sha256'],at=now()))
    write_json(output/'complete.json',dict(status='complete',at=now(),methods=14,comparisons=len(contrasts),
        file_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()},
        evaluation_complete_sha256=sha256(folder/'complete.json')))
    print(json.dumps({'status':'complete','ranking_by_ber':[(r['method'],100*r['ber']) for r in sorted(summary,key=lambda r:r['ber'])]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.folder,a.output)
