"""E7既有数据重分析：消融区间、按功率分层和环境等权SNR，不改变控制/评分。"""
import argparse
import json
import os
from pathlib import Path
import sys
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_full_baselines.common import require_host,sha256,write_json,now,atomic_npz
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap


def read(p):return json.loads(Path(p).read_text())


def load(folder):
    analysis=folder/'analysis'; proof=read(analysis/'complete.json')
    if sha256(folder/'complete.json')!=proof['evaluation_complete_sha256']:raise ValueError('Completion binding differs.')
    path=analysis/'environment_metrics.npz'
    if sha256(path)!=proof['file_sha256'][path.name]:raise ValueError('Archived metrics changed.')
    with np.load(path) as f:a={k:f[k].copy() for k in f.files}
    return a,dict(path=str(path),sha256=sha256(path),complete_sha256=sha256(analysis/'complete.json'))


def run(project,out,include_e1):
    require_host(); root=project/'dataset_simulation/baseline_results'
    original=root/'20260926_final864_selected';new=root/'20260927_uniform64_all13'
    a,old_binding=load(original);b,new_binding=load(new)
    np.testing.assert_array_equal(a['environment_ids'],b['environment_ids'])
    names=a['methods'].tolist();chunks=[a['metrics']];bindings=[old_binding,new_binding]
    added=[i for i,n in enumerate(b['methods']) if n not in names]
    chunks.append(b['metrics'][:,:,added]);names += b['methods'][added].tolist()
    values=np.concatenate(chunks,axis=2)
    if include_e1:
        folder=project/'dataset_simulation/revision_20261029/e1_full'
        if read(folder/'complete.json')['environments']!=864:raise ValueError('E1 is incomplete.')
        path=folder/'environment_metrics.npz'
        with np.load(path) as c:
            np.testing.assert_array_equal(a['environment_ids'],c['environment_ids'])
            ix=[i for i,n in enumerate(c['methods']) if n not in names]
            values=np.concatenate([values,c['values'][:,:,ix]],axis=2);names+=c['methods'][ix].tolist()
        bindings.append(dict(path=str(path),sha256=sha256(path),complete_sha256=sha256(folder/'complete.json')))
    identity=read(original/'protocol.json');rows=identity['rows']
    if len(rows)!=864 or values.shape[:2]!=(864,17):raise ValueError('Incomplete cohort.')
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope='fixed-model reanalysis; not new training or hardware evidence',
        sources=bindings,code_sha256=sha256(Path(__file__)),bootstrap_draws=10000,bootstrap_seed=20261029,
        bootstrap_unit='environment retaining 17 carriers',stratification='216 original joint strata',
        environment_equal_snr='arithmetic mean of per-environment dB power ratios; distinct from energy aggregation',
        include_e1=include_e1,at=now())
    write_json(out/'protocol.json',protocol)
    subset=['covariance_response','complex_response_cnn','covariance__joint_full_noise','cnn__joint_full_noise']
    ix=[names.index(n) for n in subset];s=sufficient(values[:,:,ix]);point=metrics(s.sum(0))
    draws=bootstrap(s,10000,20261029,strata=[r['joint_stratum_id'] for r in rows]);ci=np.quantile(draws,[.025,.975],axis=0)
    summaries=[dict(method=n,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
        ci95={f:ci[:,i,j].tolist() for j,f in enumerate(FIELDS[:5])}) for i,n in enumerate(subset)]
    write_json(out/'ablation16_intervals.json',summaries)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.1),layout='constrained')
    labels=['Conventional','A','Conventional+B','A+B'];colors=['#777777','#0072B2','#E69F00','#009E73']
    for ax,j,title,scale in [(axes[0],0,'BER (%)',100),(axes[1],3,'RMS EVM (%)',1)]:
        y=point[:,j]*scale;lo=ci[0,:,j]*scale;hi=ci[1,:,j]*scale
        ax.bar(np.arange(4),y,color=colors,width=.65)
        ax.errorbar(np.arange(4),y,yerr=np.array([y-lo,hi-y]),fmt='none',ecolor='black',capsize=4)
        ax.set_xticks(range(4),labels,rotation=18,ha='right');ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(out/'ablation16_ci95.pdf');fig.savefig(out/'ablation16_ci95.png',dpi=160);plt.close(fig)
    sufficient_all=sufficient(values);env_metrics=metrics(sufficient_all)
    group_results=[]
    for power_bin in [None,0,1,2,3,4,5]:
        mask=np.ones(864,bool) if power_bin is None else np.asarray([r['factors']['power_bin']==power_bin for r in rows])
        n=int(mask.sum())
        if n!=(864 if power_bin is None else 144):raise ValueError('Power stratum count differs.')
        totals=sufficient_all[mask].sum(0);combined=metrics(totals)
        for i,name in enumerate(names):
            if not np.isfinite(env_metrics[mask,i,4]).all():continue # digital MRC has no comparable optical SNR
            per=env_metrics[mask,i,4]
            group_results.append(dict(method=name,power_bin=power_bin,environments=n,
                power_interval_dbm=None if power_bin is None else [-105+5*power_bin,-100+5*power_bin],
                **{f:float(combined[i,j]) for j,f in enumerate(FIELDS)},
                snr_equal_environment_mean_db=float(per.mean()),snr_environment_median_db=float(np.median(per)),
                snr_environment_p05_p95_db=np.quantile(per,[.05,.95]).tolist(),
                fraction_total_signal_energy=float(totals[i,7]/sufficient_all[:,i,7].sum()),
                fraction_total_noise_energy=float(totals[i,8]/sufficient_all[:,i,8].sum())))
    write_json(out/'power_groups_and_snr.json',group_results)
    atomic_npz(out/'environment_quality.npz',metrics=env_metrics,methods=np.asarray(names),
               environment_ids=a['environment_ids'],power_bins=np.asarray([r['factors']['power_bin'] for r in rows]))
    for source in bindings:
        if sha256(source['path'])!=source['sha256']:raise ValueError('Source changed during analysis.')
    write_json(out/'complete.json',dict(status='complete_reanalysis',at=now(),environments=864,carriers=17,
        file_sha256={f.name:sha256(f) for f in out.iterdir() if f.is_file()},
        scope='ablation intervals and power/SNR stratification only; candidate/regularization sweeps still pending'))
    print(json.dumps(dict(status='complete',ablation=[(r['method'],r['ber']*100) for r in summaries]),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--include-e1',action='store_true');a=p.parse_args()
    run(a.project.resolve(),a.output.resolve(),a.include_e1)
