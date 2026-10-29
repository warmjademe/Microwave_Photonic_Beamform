"""冻结敏感性实验的全部条件核查和环境配对统计。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host,sha256,write_json,now
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap,contrast,bh_adjust


def read(p):return json.loads(Path(p).read_text())


def run(project):
    require_host();root=project/'dataset_simulation/revision_af_20261029/R/full';out=root.parent/'analysis';out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=read(root/'complete.json');p=read(root/'protocol.json')
        assert done['status']=='complete' and not done['preflight'] and done['environments']==864 and done['carriers']==17 and done['scenarios']==11
        for name,h in done['file_sha256'].items():assert sha256(root/name)==h
        with np.load(root/'environment_metrics.npz',allow_pickle=False) as f:
            values=f['values'].copy();scenarios=f['scenarios'].tolist();methods=f['methods'].tolist();ids=f['environment_ids'].tolist()
        assert values.shape==(864,11,17,3,13) and scenarios==p['scenarios'] and methods==p['methods']
        records=read(root/'records.json');assert [r['environment_id'] for r in records]==ids
        checked=0;condition_hashes={}
        for i,row in enumerate(records):
            assert [r['scenario'] for r in row['scenarios']]==scenarios
            for c,condition in enumerate(row['scenarios']):
                assert [r['carrier_ghz'] for r in condition['carriers']]==list(range(4,21))
                for k,e in enumerate(condition['carriers']):
                    path=root/e['path'];assert sha256(path)==e['sha256'] and e['protocol_sha256']==done['protocol_sha256']
                    assert e['checks']['trace_audit'] and not e['checks']['joint_correction']
                    descriptor=path.parent/'condition.json'
                    if str(descriptor) not in condition_hashes:condition_hashes[str(descriptor)]=sha256(descriptor)
                    assert condition_hashes[str(descriptor)]==e['condition_sha256']
                    with np.load(path,allow_pickle=False) as f:np.testing.assert_array_equal(values[i,c,k],f['metrics'])
                    checked+=1
        assert checked==864*11*17
        names=[scenario+':'+method for scenario in scenarios for method in methods]
        packed=values.transpose(0,2,1,3,4).reshape(864,17,33,13)
        s=sufficient(packed);point=metrics(s.sum(0));draws=bootstrap(s,10000,20261030,strata=[r['joint_stratum_id'] for r in p['rows']])
        summaries=[];comparisons=[]
        for i,name in enumerate(names):
            scenario,method=name.split(':')
            summaries.append(dict(scenario=scenario,method=method,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
                ci95={f:np.quantile(draws[:,i,j],[.025,.975]).tolist() for j,f in enumerate(FIELDS[:5])}))
        specs=[]
        for scenario in scenarios:
            specs += [('AF_vs_scan',scenario+':cnn_warm64',scenario+':initial_select64'),
                      ('AF_vs_conventional',scenario+':cnn_warm64',scenario+':covariance_warm64')]
        for scenario in scenarios[1:]:
            specs += [('perturbation_vs_nominal',scenario+':'+method,'nominal:'+method) for method in methods]
        for family,left,right in specs:
            coeff=np.zeros(len(names));coeff[names.index(left)]=1;coeff[names.index(right)]=-1
            for field in FIELDS[:5]:comparisons.append(dict(family=family,left=left,right=right,**contrast(s,draws,coeff,field)))
        for family in ['AF_vs_scan','AF_vs_conventional','perturbation_vs_nominal']:
            for field in FIELDS[:5]:
                group=[r for r in comparisons if r['family']==family and r['metric']==field]
                for row,q in zip(group,bh_adjust([r['p_value'] for r in group])):row['q_value']=float(q)
        power=[]
        bins=np.asarray([r['factors']['power_bin'] for r in p['rows']])
        for group in range(6):
            ix=bins==group;assert ix.sum()==144;stats=metrics(s[ix].sum(0))
            for i,name in enumerate(names):power.append(dict(nominal_power_bin=group,method_condition=name,environments=144,**{f:float(stats[i,j]) for j,f in enumerate(FIELDS)}))
        write_json(out/'summary.json',summaries);write_json(out/'comparisons.json',comparisons);write_json(out/'power_groups.json',power)
        write_json(out/'protocol.json',dict(source_file_sha256=done['file_sha256'],source_root=str(root),code_sha256=sha256(Path(__file__)),
            statistical_unit='paired original environment, all 17 carriers and 11 conditions kept together',bootstrap=10000,seed=20261030,
            multiplicity='BH separately by comparison type and metric',independent_real_world_data=False))
        lines=['# 冻结 A/F 敏感性结果','',
            '864 个既有测试环境，17 个载频，11 个预定条件；所有条件保留。RIN 是独立、带内平坦、平稳高斯扩展。',
            '条件数不作为独立环境数量；本结果不是实测硬件或新现场数据。','',
            '| 条件 | A+F BER (%) | 传统估计+F BER (%) | 固定扫描 BER (%) |','|---|---:|---:|---:|']
        for scenario in scenarios:
            a=[next(r for r in summaries if r['scenario']==scenario and r['method']==m) for m in methods]
            lines.append('| '+scenario+' | '+' | '.join(f"{100*r['ber']:.6f}" for r in a)+' |')
        lines+=['','各指标、配对区间和全部预定对照见 summary.json 和 comparisons.json。低功率组结果单独见 power_groups.json。','']
        (out/'REPORT_zh.md').write_text('\n'.join(lines))
        for name,h in done['file_sha256'].items():assert sha256(root/name)==h
        outputs=['summary.json','comparisons.json','power_groups.json','protocol.json','REPORT_zh.md']
        write_json(out/'complete.json',dict(status='complete',raw_records_verified=checked,at=now(),file_sha256={n:sha256(out/n) for n in outputs}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);run(p.parse_args().project.resolve())
