"""同响应监督的实数/复数估计器：全记录校验和环境配对统计。"""
import argparse
import os
from pathlib import Path
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host,sha256,write_json,now,atomic_npz
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap,contrast,bh_adjust
from revision_af_20261029.evaluate_real_response import read,fingerprint


def verified_values(folder,rq2=False):
    proof=read(folder/('verification.json' if rq2 else 'complete.json'))
    hashes=proof['files_sha256' if rq2 else 'file_sha256']
    for name,h in hashes.items():assert sha256(folder/name)==h
    p=read(folder/'protocol.json')
    with np.load(folder/'environment_metrics.npz',allow_pickle=False) as f:
        values=f['values'].copy();names=f['methods'].tolist();ids=f['environment_ids'].tolist()
    assert len(ids)==864 and p['carriers']==list(range(4,21))
    assert ids==[r['environment_id'] for r in p['rows']]
    for i,row in enumerate(p['rows']):
        for j,fc in enumerate(p['carriers']):
            raw=folder/'records'/f"environment_{row['index']:05d}"/f'carrier_{fc:02d}.npz'
            marker=read(raw.with_suffix('.json'));assert sha256(raw)==marker['sha256']
            assert marker['protocol_sha256']==fingerprint(p)
            with np.load(raw,allow_pickle=False) as f:np.testing.assert_array_equal(f['metrics'],values[i,j])
    return values,names,ids,p,hashes


def run(project):
    require_host();root=project/'dataset_simulation';base=root/'revision_rq2_af_20261029/full'
    fresh=root/'revision_af_20261029/C_response/full';out=fresh.parent/'analysis'
    if (out/'complete.json').exists():raise RuntimeError('Analysis already complete')
    old,names,ids,p,oldhash=verified_values(base,True)
    new,newnames,newids,np_,newhash=verified_values(fresh)
    assert ids==newids and p['rows']==np_['rows']
    values=np.concatenate([old,new],axis=2);names=names+newnames
    assert names==['covariance_response','complex_response_cnn','covariance_warm64','cnn_warm64','initial_select64','real_response_direct16','real_response_feedback64']
    out.mkdir(parents=True,exist_ok=True);s=sufficient(values);point=metrics(s.sum(0))
    draws=bootstrap(s,10000,20261030,strata=[r['joint_stratum_id'] for r in p['rows']])
    specs=[('real_minus_complex_direct',[0,-1,0,0,0,1,0]),
           ('real_minus_complex_feedback',[0,0,0,-1,0,0,1]),
           ('feedback_added_to_real',[0,0,0,0,0,-1,1]),
           ('real_feedback_minus_scan',[0,0,0,0,-1,0,1])]
    comparisons=[]
    for label,c in specs:
        for field in FIELDS[:5]:comparisons.append(dict(id=label,**contrast(s,draws,c,field)))
    for field in FIELDS[:5]:
        selected=[r for r in comparisons if r['metric']==field]
        for row,q in zip(selected,bh_adjust([r['p_value'] for r in selected])):row['sign_test_bh_q']=float(q)
    summary=[dict(method=n,probes=int(values[0,0,i,10]),**{f:float(point[i,j]) for j,f in enumerate(FIELDS)}) for i,n in enumerate(names)]
    write_json(out/'summary.json',summary);write_json(out/'comparisons.json',comparisons)
    atomic_npz(out/'environment_metrics.npz',values=values,methods=np.asarray(names),environment_ids=np.asarray(ids))
    write_json(out/'protocol.json',dict(code_sha256=sha256(Path(__file__)),source_protocol_sha256=sha256(fresh/'protocol.json'),
        original_sha256=oldhash,new_sha256=newhash,training_audit=np_['training_audit'],
        bootstrap=dict(unit='propagation environment',strata=216,repetitions=10000,seed=20261030),
        comparisons=specs,multiplicity='four contrasts separately within each metric',
        interpretation='fixed-weight estimator comparison on previously analyzed test environments'))
    lines=['# 同响应监督的实数 CNN 补测','',
        '全部 864 个环境 × 17 载频。两个估计器共享响应监督、训练成员和离散求解器；实数/复数卷积与激活不同。',
        '逐条校验 14,688 个新增载频记录和 14,688 个原 RQ2 记录。','',
        '| 方法 | 测量数 | BER (%) | RMS EVM (%) | 配对 SNR (dB) |','|---|---:|---:|---:|---:|']
    for row in summary:lines.append(f"| {row['method']} | {row['probes']} | {100*row['ber']:.6f} | {row['rms_evm_percent']:.6f} | {row['paired_output_snr_db']:.6f} |")
    lines+=['','| BER 对照 | 差值 (百分点) | 95% 区间 | 胜/平/负 | 符号检验 BH q |','|---|---:|---|---|---:|']
    for row in comparisons:
        if row['metric']=='ber':
            lo,hi=np.asarray(row['ci95'])*100
            lines.append(f"| {row['id']} | {100*row['estimate']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {row['wins']}/{row['ties']}/{row['losses']} | {row['sign_test_bh_q']:.6g} |")
    lines+=['','均值区间与环境胜负检验分别解释；反馈增益同时增加测量次数。当前比较不将差异单独归因于某个卷积或激活算子。','']
    (out/'REPORT_zh.md').write_text('\n'.join(lines))
    write_json(out/'complete.json',dict(status='complete',at=now(),raw_records_verified=29376,
        file_sha256={n:sha256(out/n) for n in ['summary.json','comparisons.json','environment_metrics.npz','protocol.json','REPORT_zh.md']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
