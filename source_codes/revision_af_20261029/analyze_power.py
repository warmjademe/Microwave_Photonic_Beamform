"""从冻结 A/F 全量接收记录重算功率分层与环境等权 SNR。"""
import argparse
import json
import os
from pathlib import Path
import sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host,sha256,write_json,now,atomic_npz
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap,contrast,bh_adjust


def read(p):return json.loads(p.read_text())


def run(project):
    require_host();base=project/'dataset_simulation/revision_rq2_af_20261029/full'
    out=project/'dataset_simulation/revision_af_20261029/D/power'
    if out.exists():raise FileExistsError(out)
    proof=read(base/'verification.json');assert proof['status']=='passed' and proof['conditions']==14688
    bindings={}
    for name in ['protocol.json','environment_metrics.npz','records.json']:
        assert sha256(base/name)==proof['files_sha256'][name];bindings[str(base/name)]=sha256(base/name)
    p=read(base/'protocol.json');records=read(base/'records.json');checked=0
    for record in records:
        assert [r['carrier_ghz'] for r in record['carriers']]==list(range(4,21))
        for entry in record['carriers']:
            assert sha256(base/entry['path'])==entry['sha256'];checked+=1
    assert checked==14688
    with np.load(base/'environment_metrics.npz',allow_pickle=False) as f:
        names=f['methods'].tolist();values=f['values'];ids=f['environment_ids']
    assert values.shape==(864,17,5,13)
    groups=np.array([r['factors']['power_bin'] for r in p['rows']]);strata=np.asarray([r['joint_stratum_id'] for r in p['rows']])
    s=sufficient(values);env=metrics(s);all_total=s.sum(0)
    output=[];comparisons=[]
    for group in [None,0,1,2,3,4,5]:
        mask=np.ones(864,bool) if group is None else groups==group
        assert mask.sum()==(864 if group is None else 144)
        total=s[mask].sum(0);score=metrics(total)
        draws=bootstrap(s[mask],10000,20261030,strata=strata[mask])
        for i,name in enumerate(names):
            v=env[mask,i,4]
            output.append(dict(method=name,power_bin=group,environments=int(mask.sum()),
                **{f:float(score[i,j]) for j,f in enumerate(FIELDS)},
                snr_equal_environment_mean_db=float(v.mean()),snr_environment_median_db=float(np.median(v)),
                snr_environment_p05_p95_db=np.quantile(v,[.05,.95]).tolist(),
                fraction_total_signal_energy=float(total[i,7]/all_total[i,7]),
                fraction_total_noise_energy=float(total[i,8]/all_total[i,8])))
        for label,coeff in [('AF_vs_scan',[0,0,0,1,-1]),('AF_vs_conventional_F',[0,0,-1,1,0])]:
            for field in FIELDS[:5]:comparisons.append(dict(power_bin=group,id=label,**contrast(s[mask],draws,coeff,field)))
    # 每项指标中，六个功率档和总体的两个预设对比构成14项比较族。
    for field in FIELDS[:5]:
        group=[r for r in comparisons if r['metric']==field]
        for r,q in zip(group,bh_adjust([r['p_value'] for r in group])):r['q_value']=float(q)
    old=project/'dataset_simulation/revision_20261029/e7_original'
    oldproof=read(old/'complete.json');oldfile=old/'power_groups_and_snr.json'
    assert sha256(oldfile)==oldproof['file_sha256'][oldfile.name]
    archived=read(oldfile)
    for row in output:
        prior=next(x for x in archived if x['method']==row['method'] and x['power_bin']==row['power_bin'])
        for k in [*FIELDS,'snr_equal_environment_mean_db','fraction_total_signal_energy','fraction_total_noise_energy']:
            np.testing.assert_allclose(row[k],prior[k],rtol=1e-10,atol=1e-12)
    out.mkdir(parents=True)
    write_json(out/'power_groups.json',output);write_json(out/'comparisons.json',comparisons)
    atomic_npz(out/'environment_quality.npz',methods=np.asarray(names),environment_ids=ids,power_bins=groups,metrics=env)
    write_json(out/'protocol.json',dict(at=now(),code_sha256=sha256(Path(__file__)),source_files_sha256=bindings,
        archived_comparison_sha256=sha256(oldfile),raw_records_rechecked=checked,
        bootstrap_draws=10000,bootstrap_seed=20261030,unit='environment retaining all 17 carriers',
        strata='216 original strata, or the 36 strata within each power bin',
        multiplicity='BH over 14 contrasts per metric',new_controls=False,independent_interference=False))
    lines=['# A/F 功率分层与 SNR 汇总','',
        '完整 864×17 原始记录重新核验。各功率层 144 环境；没有重新选择控制。',
        '功率求和 SNR 与环境 dB 等权平均是不同指标，二者都报告，不互相替代。','',
        '| 功率层 (dBm) | A+F BER (%) | 固定扫描 BER (%) | A+F SNR (dB) | 固定扫描 SNR (dB) | A+F 信号能量占比 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for group in range(6):
        a=next(x for x in output if x['method']=='cnn_warm64' and x['power_bin']==group)
        b=next(x for x in output if x['method']=='initial_select64' and x['power_bin']==group)
        lines.append(f"| {-105+5*group} to {-100+5*group} | {100*a['ber']:.4f} | {100*b['ber']:.4f} | {a['paired_output_snr_db']:.4f} | {b['paired_output_snr_db']:.4f} | {100*a['fraction_total_signal_energy']:.4f} |")
    a=next(x for x in output if x['method']=='cnn_warm64' and x['power_bin'] is None)
    lines+=['',f"A+F 全体功率求和 SNR：{a['paired_output_snr_db']:.4f} dB；环境等权 dB 均值：{a['snr_equal_environment_mean_db']:.4f} dB；环境中位数：{a['snr_environment_median_db']:.4f} dB。",'']
    (out/'REPORT_zh.md').write_text('\n'.join(lines))
    for path,h in bindings.items():assert sha256(path)==h
    write_json(out/'complete.json',dict(status='complete',at=now(),raw_records_verified=checked,
        agrees_with_archived_power_analysis=True,file_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
