"""验证集 H 的预定比较：环境为统计单位，候选种子分别呈现。"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host,sha256,write_json,now
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap,contrast,bh_adjust


def read(path):
    return json.loads(Path(path).read_text())


def run(project):
    require_host()
    root=project/'dataset_simulation/revision_af_20261029/H/full'
    output=root.parent/'analysis';output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=read(root/'complete.json');p=read(root/'protocol.json')
        assert done['status']=='complete' and not done['preflight'] and not done['joint_correction']
        assert done['environments']==216 and done['carriers']==17
        sources={str(root/n):h for n,h in done['file_sha256'].items()}
        for name,h in sources.items():assert sha256(name)==h
        with np.load(root/'environment_metrics.npz',allow_pickle=False) as f:
            values=f['values'].copy();names=f['methods'].tolist();ids=f['environment_ids'].tolist()
        assert names==p['methods'] and values.shape==(216,17,24,13)
        assert ids==[r['environment_id'] for r in p['rows']]
        rows=read(root/'records.json')
        assert [r['environment_id'] for r in rows]==ids
        count=0;origins={n:Counter() for n in names if n.startswith('candidates_')}
        model_measurements={n:[] for n in origins}
        for i,row in enumerate(rows):
            assert [e['carrier_ghz'] for e in row['carriers']]==list(range(4,21))
            for j,e in enumerate(row['carriers']):
                f=root/e['path'];assert sha256(f)==e['sha256']
                assert e['protocol_sha256']==done['protocol_sha256']
                assert not e['checks']['joint_correction'] and e['checks']['feedback_trace_audit']
                with np.load(f,allow_pickle=False) as a:
                    np.testing.assert_array_equal(values[i,j],a['metrics'])
                    for name in origins:
                        _,k,_,seed=name.split('_'); k,seed=int(k),int(seed)
                        selected=a['control_code'][names.index(name)]
                        trace=a[name+'__trace_control_code'];scores=a[name+'__trace_scores']
                        selected_index=int(scores.argmax())
                        np.testing.assert_array_equal(selected,trace[selected_index])
                        source='initial' if selected_index<16 else ('model' if any(
                            np.array_equal(selected,u) for u in a['candidate_codes'][seed,:k]) else 'geometric')
                        origins[name][source]+=1
                    trace=a['candidates_1_seed_0__trace_control_code'];scores=a['candidates_1_seed_0__trace_scores']
                    for b in [16,24,32,48]:
                        np.testing.assert_array_equal(a['control_code'][names.index(f'prefix_{b}')],trace[scores[:b].argmax()])
                for detail in e['checks']['candidate_measurements']:
                    model_measurements[detail['method']].append(detail['actually_measured_models'])
                count+=1
        assert count==3672
        s=sufficient(values);point=metrics(s.sum(0));draws=bootstrap(s,10000,20261029,strata=None)
        rows_summary=[dict(method=n,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
            ci95={f:np.quantile(draws[:,i,j],[.025,.975]).tolist() for j,f in enumerate(FIELDS[:5])}) for i,n in enumerate(names)]
        specs=[]
        specs += [('loading',f'A_loading_{x:g}','A_loading_0.01') for x in [.001,.003,.03,.1]]
        specs += [('noise_weighting',n,'ridge_isotropic') for n in ['ridge_noise_weighted','covariance']]
        specs += [('candidate_count',f'candidates_{k}_seed_{seed}',f'candidates_1_seed_{seed}') for seed in [0,1,2] for k in [4,16,48]]
        specs += [('budget',f'prefix_{b}','candidates_1_seed_0') for b in [16,24,32,48]]
        comparisons=[]
        for family,left,right in specs:
            coefficients=np.zeros(len(names));coefficients[names.index(left)]=1;coefficients[names.index(right)]=-1
            for field in FIELDS[:5]:
                comparisons.append(dict(family=family,left=left,right=right,**contrast(s,draws,coefficients,field)))
        for family in ['loading','noise_weighting','candidate_count','budget']:
            for field in FIELDS[:5]:
                group=[r for r in comparisons if r['family']==family and r['metric']==field]
                for r,q in zip(group,bh_adjust([r['p_value'] for r in group])):r['q_value']=float(q)
        counts=[dict(method=n,selected=dict(origins[n]),mean_actual_model_measurements=float(np.mean(model_measurements[n])),
            min_actual_model_measurements=min(model_measurements[n]),max_actual_model_measurements=max(model_measurements[n])) for n in origins]
        candidate_seed_variation=[]
        for k in [1,4,16,48]:
            m=np.asarray([point[names.index(f'candidates_{k}_seed_{seed}')] for seed in [0,1,2]])
            candidate_seed_variation.append(dict(candidates=k,metrics={f:dict(values=m[:,j].tolist(),mean=float(m[:,j].mean()),sample_sd=float(m[:,j].std(ddof=1))) for j,f in enumerate(FIELDS)}))
        write_json(output/'summary.json',rows_summary);write_json(output/'comparisons.json',comparisons)
        write_json(output/'candidate_sources.json',counts);write_json(output/'candidate_seed_variation.json',candidate_seed_variation)
        write_json(output/'protocol.json',dict(source_files_sha256=sources,code_sha256=sha256(Path(__file__)),
            statistical_unit='validation environment; all 17 carriers kept together',environments=216,
            bootstrap='ordinary paired environment bootstrap; one observation per design stratum',
            repetitions=10000,seed=20261029,multiplicity='BH within each predefined comparison family and metric',
            interpretation='frozen-model sensitivity, not validation-selected new model or independent test'))
        lines=['# A/F 验证集敏感性结果','',
            '全部 216 个验证环境、17 个载频；冻结 seed 0 网络。以下 BER 为百分数。',
            '正则化扫描只改变推理初值，没有为各值重新训练。候选数量比较固定总预算 64。',
            'prefix_16 选择初始实测最佳控制，与 A_loading_0.01 直接输出不同。','',
            '| 配置 | BER (%) | RMS EVM (%) | 输出 SNR (dB) |','|---|---:|---:|---:|']
        for r in rows_summary:lines.append(f"| {r['method']} | {100*r['ber']:.6f} | {r['rms_evm_percent']:.4f} | {r['paired_output_snr_db']:.4f} |")
        lines+=['','完整配对差值、置信区间和族内校正见 comparisons.json。',
            '候选种子间变异与环境不确定性分别报告；K=1 的三组结果相同是固定首候选定义产生的重复配置，不计为独立样本。','']
        (output/'REPORT_zh.md').write_text('\n'.join(lines))
        for name,h in sources.items():assert sha256(name)==h
        outputs=['summary.json','comparisons.json','candidate_sources.json','candidate_seed_variation.json','protocol.json','REPORT_zh.md']
        write_json(output/'complete.json',dict(status='complete',raw_records_verified=count,at=now(),
            file_sha256={n:sha256(output/n) for n in outputs}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
