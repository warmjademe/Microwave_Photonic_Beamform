"""核查全量旧216探索记录，完成同预算排名及组件配对统计。"""
import argparse
import json
import os
from pathlib import Path
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,sha256,verify_sources,write_json,now,atomic_npz
from study_full_baselines.confirmation_batch import committed_environment,fingerprint
from study_full_baselines.analyze_confirmation import check_values,grouped
from study_full_baselines.runtime_bundle import verify
from study_full_baselines.paired_statistics import FIELDS,sufficient,metrics,bootstrap,contrast,bh_adjust,self_check
from study_full_baselines.audit_results import statistics
from study_full_baselines.analyze_reception import clean_json


def comparisons(identity):
    names=identity['methods']; budgets=identity['measurement_budgets']; result=[]
    def add(label,family,terms):
        c=[terms.get(n,0) for n in names]
        if set(terms)-set(names) or sum(c)!=0:
            raise ValueError('比较清单错误。')
        result.append(dict(id=label,family=family,terms=terms,coefficients=c))
    for n in identity['ordinary_methods']:
        if budgets[n]==16 and n!='cnn__spatial':
            add('cnn_spatial_minus_'+n,'same16',{'cnn__spatial':1,n:-1})
        elif budgets[n]==64 and n!='cnn_warm64':
            add('cnn_warm_minus_'+n,'same64',{'cnn_warm64':1,n:-1})
    for label,terms in [
        ('A_only',{'complex_response_cnn':1,'covariance_response':-1}),
        ('B_only',{'covariance__spatial':1,'covariance_response':-1}),
        ('remove_B',{'cnn__spatial':1,'complex_response_cnn':-1}),
        ('remove_A',{'cnn__spatial':1,'covariance__spatial':-1}),
        ('interaction',{'cnn__spatial':1,'complex_response_cnn':-1,'covariance__spatial':-1,'covariance_response':1}),
        ('spatial_vs_isotropic',{'cnn__spatial':1,'cnn__isotropic':-1})]:
        add(label,'components',terms)
    return result


def run(folder,output):
    require_host(); self_check()
    identity=json.loads((folder/'protocol.json').read_text())
    complete=json.loads((folder/'complete.json').read_text())
    if (identity['scope']!='old216_exploratory' or identity['final_confirmation']
            or len(identity['rows'])!=216 or identity['carriers']!=list(range(4,21))
            or complete['identity_sha256']!=fingerprint(identity)
            or complete['records_sha256']!=sha256(folder/'records.json')):
        raise ValueError('需要完整的旧216探索记录。')
    if output.exists():
        if (output/'complete.json').exists():
            prior=json.loads((output/'complete.json').read_text())
            if prior['evaluation_complete_sha256']==sha256(folder/'complete.json'):
                for name,digest in prior['file_sha256'].items():
                    if sha256(output/name)!=digest: raise ValueError('已有分析产物改变。')
                return
        raise FileExistsError('保留已有未完成分析，请检查后另建输出目录。')
    verify_sources(identity['source_sha256']); bundle=Path(identity['runtime_bundle']); verify(bundle)
    if sha256(bundle/'manifest.json')!=identity['runtime_manifest_sha256']:
        raise ValueError('模型来源改变。')
    with np.load(bundle/'public.npz') as f: public={k:f[k].copy() for k in f.files}
    values=[]; records=[]; files={}; traces=0
    for row in identity['rows']:
        record=committed_environment(folder,row,identity)
        if record is None: raise ValueError('缺少完整环境。')
        records.append(record); current=[]
        for item in record['carriers']:
            path=folder/item['path']
            with np.load(path,allow_pickle=False) as f: arrays={k:f[k].copy() for k in f.files}
            traces+=check_values(arrays,item['carrier_ghz'],identity,public)
            current.append(arrays['metrics']); files[item['path']]=item['sha256']
        values.append(current)
    if records!=json.loads((folder/'records.json').read_text()): raise ValueError('环境索引不一致。')
    values=np.asarray(values); s=sufficient(values); point=metrics(s.sum(0)); names=identity['methods']
    expected_cases=len(records)*17*len(names)
    if (complete['cases']!=expected_cases or complete['methods']!=len(names)
            or complete['environments']!=216 or complete['carriers']!=17):
        raise ValueError('完成计数不符。')
    recomputed=0
    for i,name in enumerate(names):
        direct=statistics(values[:,:,i],name)
        for j,field in enumerate(FIELDS):
            if field in direct:
                np.testing.assert_allclose(point[i,j],direct[field],rtol=1e-12,atol=0,equal_nan=True)
                recomputed+=1
    plan=comparisons(identity); draws=bootstrap(s,10000,20260926); results=[]
    for comparison in plan:
        for field in ['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db']:
            got=contrast(s,draws,comparison['coefficients'],field)
            results.append(dict(**comparison,**got))
    for family in sorted({r['family'] for r in results}):
        for field in ['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db']:
            group=[r for r in results if r['family']==family and r['metric']==field]
            for row,q in zip(group,bh_adjust([r['p_value'] for r in group])): row['sign_test_bh_q']=float(q)
    summary=[dict(method=n,measurement_budget=identity['measurement_budgets'].get(n),
        reference=n in ['teacher','mrc'],**{f:float(point[i,j]) for j,f in enumerate(FIELDS)})
        for i,n in enumerate(names)]
    output.mkdir(parents=True)
    write_json(output/'summary.json',clean_json(summary))
    write_json(output/'groups.json',clean_json(grouped(values,identity)))
    write_json(output/'comparisons.json',clean_json(results))
    atomic_npz(output/'environment_metrics.npz',metrics=values,methods=np.asarray(names),
        environment_ids=np.asarray([r['environment_id'] for r in identity['rows']]))
    audit=dict(status='passed',at=now(),scope=identity['scope'],final_confirmation=False,
        environments=216,carriers=17,methods=len(names),cases=expected_cases,
        control_codes_checked=expected_cases*128,feedback_trace_rows_checked=traces,
        independently_recomputed_aggregate_numbers=recomputed,raw_file_sha256=files,
        source_sha256=identity['source_sha256'],runtime_manifest_sha256=identity['runtime_manifest_sha256'],
        protocol_sha256=sha256(folder/'protocol.json'),records_sha256=sha256(folder/'records.json'))
    write_json(output/'audit.json',audit)
    text=['# 3,456训练环境全基线探索结果','',
        '旧216环境×17载频；单次训练seed0、40轮；本报告不是新864环境的独立确认。',
        '全部控制通过实际单条推理/搜索生成。不同测量预算分别排名；并行运行时间不作为正式在线时延。','']
    for budget in [0,16,64,None]:
        label='额外信息参考' if budget is None else str(budget)+'次测量'
        text+=['## '+label,'','| 方法 | BER（%） | EVM（%） | 配对SNR（dB） |','|---|---:|---:|---:|']
        for r in sorted([r for r in summary if r['measurement_budget']==budget],key=lambda r:r['ber']):
            snr='不适用' if not np.isfinite(r['paired_output_snr_db']) else '%.4f'%r['paired_output_snr_db']
            text+=['| %s | %.4f | %.4f | %s |'%(r['method'],100*r['ber'],r['rms_evm_percent'],snr)]
        text+=['']
    text+=['## 组件效果','',
        '差值为前者减后者，BER负值表示前者改善；remove_B表示完整方法减去去掉B的版本，remove_A同理。',
        '区间来自10,000次环境配对bootstrap，保留环境内17频率。符号检验的BH值检验环境胜负概率，不是平均差的p值。','',
        '| 比较 | BER差（百分点） | 95%区间 | 改善/平/差环境 | 符号检验BH值 |','|---|---:|---|---|---:|']
    for r in results:
        if r['family']=='components' and r['metric']=='ber':
            text+=['| %s | %.4f | [%.4f, %.4f] | %d/%d/%d | %.6g |'%(r['id'],100*r['estimate'],
                100*r['ci95'][0],100*r['ci95'][1],r['wins'],r['ties'],r['losses'],r['sign_test_bh_q'])]
    text+=['','下一步依据全部基线与分组误差迭代；没有收益的组件保留负结果并修改或删除。独立确认、正式计时和网站发布尚未完成。','']
    (output/'结果说明.md').write_text('\n'.join(text))
    verify_sources(identity['source_sha256'])
    file_hash={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'complete.json',dict(status='complete_exploratory_analysis',at=now(),
        evaluation_complete_sha256=sha256(folder/'complete.json'),file_sha256=file_hash,
        comparisons=len(results),final_confirmation=False,overall_research_complete=False))
    print(json.dumps(dict(status='complete_exploratory_analysis',methods=len(names),cases=expected_cases)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.evaluation.resolve(),a.output.resolve())
