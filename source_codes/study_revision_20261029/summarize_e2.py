"""等待全部E2种子评价完成，统一报告三个固定种子；不选择最好种子。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host,sha256,write_json,now,atomic_npz
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap


def read(p):return json.loads(Path(p).read_text())


def run(project,wait):
    require_host();root=project/'dataset_simulation/revision_20261029'
    out=root/'e2_analysis';out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[root/'e1_full',root/'e2_seed_1_test',root/'e2_seed_2_test']
        while not all((p/'complete.json').exists() for p in pending):
            missing=[p.name for p in pending if not (p/'complete.json').exists()]
            for p in pending:
                if (p/'failure.json').exists():raise RuntimeError('Evaluation failed: '+str(p))
            if not wait:raise RuntimeError('Evaluation incomplete: '+str(missing))
            seq=read(root/'e2_sequence_launch.json');os.kill(seq['pid'],0)
            write_json(out/'progress.json',dict(status='waiting_for_evaluation',pending=missing,at=now()))
            time.sleep(15)
        original=project/'dataset_simulation/baseline_results/20260926_final864_selected'
        protocol=read(original/'protocol.json');done=read(original/'analysis/complete.json')
        original_npz=original/'analysis/environment_metrics.npz'
        if sha256(original_npz)!=done['file_sha256']['environment_metrics.npz']:raise ValueError('Original metrics changed.')
        sources=[original_npz];names=['A_direct16','AB_direct16','A_feedback64','AB_feedback64']
        with np.load(original_npz) as f:
            ids=f['environment_ids'].copy();old_names=f['methods'].tolist()
            seed0=f['metrics'][:,:,[old_names.index(n) for n in ['complex_response_cnn','cnn__joint_full_noise','cnn_warm64']]].copy()
        e1=root/'e1_full/environment_metrics.npz';sources.append(e1)
        with np.load(e1) as f:
            np.testing.assert_array_equal(ids,f['environment_ids'])
            ix=f['methods'].tolist().index('cnn_joint_feedback64')
            seed0=np.concatenate([seed0,f['values'][:,:,ix:ix+1]],axis=2)
        values=[seed0]
        for seed in (1,2):
            folder=root/('e2_seed_%d_test'%seed);proof=read(folder/'complete.json')
            if proof['seed']!=seed or proof['environments']!=864 or proof['preflight']:raise ValueError('Incomplete seed evaluation.')
            path=folder/'environment_metrics.npz';sources.append(path)
            with np.load(path) as f:
                np.testing.assert_array_equal(ids,f['environment_ids'])
                if f['methods'].tolist()!=names:raise ValueError('Method order differs.')
                values.append(f['values'].copy())
        values=np.asarray(values)
        if values.shape!=(3,864,17,4,13):raise ValueError('Wrong multi-seed evaluation shape.')
        source_sha={str(p):sha256(p) for p in sources}
        write_json(out/'protocol.json',dict(seeds=[0,1,2],methods=names,source_sha256=source_sha,
            code_sha256=sha256(Path(__file__)),scope='all three final checkpoints; no best-seed selection',
            uncertainty='per-seed environment bootstrap; cross-seed sample SD separately, not a population confidence interval',
            bootstrap_seed=20261029,bootstrap_draws=10000,at=now()))
        points=[];per_seed=[]
        for seed in (0,1,2):
            s=sufficient(values[seed]);point=metrics(s.sum(0));points.append(point)
            draws=bootstrap(s,10000,20261029,strata=[r['joint_stratum_id'] for r in protocol['rows']])
            for i,name in enumerate(names):
                per_seed.append(dict(seed=seed,method=name,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
                    ci95={f:np.quantile(draws[:,i,j],[.025,.975]).tolist() for j,f in enumerate(FIELDS[:5])}))
        points=np.asarray(points)
        summary=[dict(method=name,seed_count=3,
            metrics={f:dict(mean=float(points[:,i,j].mean()),sample_sd=float(points[:,i,j].std(ddof=1)),
                           min=float(points[:,i,j].min()),max=float(points[:,i,j].max())) for j,f in enumerate(FIELDS)})
            for i,name in enumerate(names)]
        effects=[]
        for first,second in [(1,0),(3,2)]:
            effects.append(dict(first=names[first],second=names[second],
                ber_difference_percentage_points=(100*(points[:,first,0]-points[:,second,0])).tolist(),
                seeds_with_lower_ber=int(np.sum(points[:,first,0]<points[:,second,0])),total_seeds=3))
        write_json(out/'per_seed.json',per_seed);write_json(out/'mean_and_seed_sd.json',summary)
        write_json(out/'B_effect_by_seed.json',effects)
        atomic_npz(out/'all_seed_environment_metrics.npz',values=values,methods=np.asarray(names),environment_ids=ids)
        for path,digest in source_sha.items():
            if sha256(path)!=digest:raise ValueError('Source changed during summary.')
        write_json(out/'complete.json',dict(status='complete',seeds=[0,1,2],environments_per_seed=864,
            file_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name not in ['run.lock','progress.json']},at=now()))
        write_json(out/'progress.json',dict(status='complete',at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    try:run(a.project.resolve(),a.wait)
    except BaseException:
        output=a.project/'dataset_simulation/revision_20261029/e2_analysis';output.mkdir(parents=True,exist_ok=True)
        write_json(output/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
