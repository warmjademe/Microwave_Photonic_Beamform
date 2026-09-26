"""全基线完成后，在原训练成员内检验跨频率校正假设。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE,require_host,sha256,source_record,verify_sources,write_json,now,atomic_npz,LEVELS
from study_full_baselines.runtime_bundle import verify,fingerprint
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_receiver import score_decisions
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap,contrast
from study_full_baselines.analyze_reception import clean_json
from our_method_response_control.train import precision,verify_targets
from our_method_quality_rank.train import load_split
from our_method_response_control.physics import ridge_estimate,covariance_estimate,design,measurement_noise,decode
from our_method_response_control.model import conditions
from study_fair_followup.controllers import binding
from our_method_measurement_refinement.method import refine
from our_method_joint_refinement.method import refine_joint,full_noise

METHODS=[a+'__'+v for a in ['covariance','cnn'] for v in ['original','spatial','joint_block_noise','joint_full_noise']]
STATE={}


def read(path): return json.loads(Path(path).read_text())


def setup(project):
    root=project/'dataset_simulation'; study=root/'baseline_results/20260925_full_baselines'
    bundle=study/'fair_3456/runtime_bundle'; package=verify(bundle)
    bound=binding(project,package)
    with np.load(bundle/'public.npz') as f: public={k:f[k].copy() for k in f.files}
    models={a:OnlineController(bundle,n,public) for a,n in
            [('covariance','covariance_response'),('cnn','complex_response_cnn')]}
    if bound['weights_sha256'] not in models['cnn'].artifacts.values(): raise ValueError('CNN权重来源不同。')
    spatial={a:np.load(Path(bound['path'])/(a+'_residual_covariance.npy')) for a in models}
    return bundle,package,bound,public,models,spatial


def numerical_check(project,output):
    precision(); _,package,_,public,models,spatial=setup(project)
    data=project/'dataset_simulation/outputs/scaling_train_3456_20260925'
    row=next(r for r in read(data/'manifest.json')['environments'] if r['split']=='train' and r['index']==0)
    with np.load(data/row['path']/'data.npz') as f: raw=f['X'].copy()
    identity=np.broadcast_to(np.eye(31),(17,31,31)).copy()
    smooth=np.broadcast_to(.8**abs(np.arange(31)[:,None]-np.arange(31)[None,:]),(17,31,31)).copy()
    records=[]
    with torch.inference_mode():
        for fc in [4,12,20]:
            x=raw[fc-4]; noise=full_noise(x,public['pilot_qpsk']); blocks=measurement_noise(design(fc),x,public['pilot_qpsk'])
            diagonal=np.stack([noise.reshape(31,16,31,16)[k,:,k,:] for k in range(31)])
            # 公开X可能为float32，各频率反推同一APD方差时允许其量化误差。
            noise_error=float(np.linalg.norm(diagonal-blocks)/np.linalg.norm(blocks))
            if noise_error>2e-6: raise ValueError('完整噪声的对角块与既有定义不同。')
            np.testing.assert_allclose(noise,noise.conj().T,rtol=1e-12,atol=1e-30)
            eigen=np.linalg.eigvalsh(noise)
            if eigen.min() < -1e-10*eigen.max(): raise ValueError('完整噪声非半正定。')
            for a,model in models.items():
                h=model.estimate(x)
                old,_=refine(x,public['pilot_qpsk'],h,spatial[a],'spatial')
                reverted,_=refine_joint(x,public['pilot_qpsk'],h,spatial[a],identity,False,True)
                error=float(np.linalg.norm(old-reverted)/max(np.linalg.norm(old),1e-30))
                if error>1e-8: raise ValueError('对角频率退化检查失败。')
                _,detail=refine_joint(x,public['pilot_qpsk'],h,spatial[a],smooth,True,True)
                if detail['solve_relative_residual']>1e-6: raise ValueError('联合求解残差过大。')
                records.append(dict(carrier=fc,estimator=a,noise_diagonal_relative_error=noise_error,
                    original_refinement_relative_error=error,**detail))
    write_json(output/'numerical_preflight.json',dict(status='passed',at=now(),records=records,
        train_environments=1,cases=6,test_used=False,full_noise_roundoff_tolerance=2e-6))


def fit(project,output):
    precision(); _,package,bound,public,models,spatial=setup(project)
    data=project/'dataset_simulation/outputs/scaling_train_3456_20260925'
    targets=project/'dataset_simulation/baseline_results/20260925_full_baselines/scale_3456/targets'
    verify_targets(data,targets)
    x,_,_,rows=load_split(data,'train',labels=False)
    if len(x)!=58752 or [r['environment_id'] for r in rows]!=package['cohort']['train_ids']:
        raise ValueError('拟合成员不同。')
    initial=ridge_estimate(x,public['pilot_qpsk']).astype(np.complex64)
    cond=conditions(x,initial,public['pilot_qpsk'])
    target=np.load(targets/'train_response.npy',mmap_mode='r')
    freq=np.zeros((2,17,31,31),complex); spatial_sum=np.zeros((2,17,64,64),complex);counts=np.zeros(17,int)
    with torch.inference_mode():
        for begin in range(0,len(x),64):
            end=min(begin+64,len(x))
            cnn=models['cnn'].model(torch.from_numpy(initial[begin:end]).cuda(),torch.from_numpy(cond[begin:end]).cuda()).cpu().numpy()
            traditional=np.stack([covariance_estimate(t,public['pilot_qpsk'],models['covariance'].weights) for t in x[begin:end]])
            for ai,prediction in enumerate([traditional,cnn]):
                scale=np.maximum(np.mean(abs(prediction.astype(complex))**2,axis=(1,2)),1e-24)
                error=(np.asarray(target[begin:end],complex)-prediction)/np.sqrt(scale)[:,None,None]
                for fc in np.unique(x[begin:end,1984].astype(int)):
                    mask=x[begin:end,1984]==fc;e=error[mask]
                    freq[ai,fc-4]+=np.sum(e.transpose(0,2,1)@e.conj(),axis=0)
                    spatial_sum[ai,fc-4]+=np.sum(e@e.conj().transpose(0,2,1),axis=0)
                    if ai==0: counts[fc-4]+=int(mask.sum())
            if begin%512==0:
                write_json(output/'progress.json',dict(status='fitting_frequency_statistics',completed=end,total=len(x),pid=os.getpid(),at=now()))
    if np.any(counts!=3456): raise ValueError('拟合频率计数不同。')
    files={}
    for ai,a in enumerate(['covariance','cnn']):
        recomputed=spatial_sum[ai]/(counts[:,None,None]*31)
        np.testing.assert_allclose(recomputed,spatial[a],rtol=1e-10,atol=1e-14)
        covariance=freq[ai]/(counts[:,None,None]*64)
        covariance=(covariance+covariance.conj().transpose(0,2,1))/2
        covariance/=np.trace(covariance,axis1=1,axis2=2).real[:,None,None]/31
        if np.linalg.eigvalsh(covariance).min() < -1e-10: raise ValueError('频率统计非半正定。')
        p=output/(a+'_frequency_covariance.npy');np.save(p,covariance);files[p.name]=sha256(p)
    write_json(output/'fit_complete.json',dict(status='complete',at=now(),train_environments=3456,
        train_samples=58752,network_updates=0,test_used=False,old_spatial_statistics_reproduced=True,
        file_sha256=files,data_manifest_sha256=sha256(data/'manifest.json'),
        weights_sha256=bound['weights_sha256'],target_complete_sha256=sha256(targets/'complete.json')))


def initialize(project,output):
    require_host();precision();project=Path(project);output=Path(output)
    _,_,_,public,models,spatial=setup(project)
    frequency={a:np.load(output/(a+'_frequency_covariance.npy')) for a in models}
    STATE.update(project=project,output=output,public=public,models=models,spatial=spatial,frequency=frequency)


def one(row):
    s=STATE;data=s['project']/'dataset_simulation/outputs/scaling_train_3456_20260925'
    with np.load(data/row['path']/'data.npz') as f:raw=f['X'].copy()
    public=s['public'];decisions=[];estimates=[]
    with torch.inference_mode():
        for fc in [4,12,20]:
            x=raw[fc-4];start=public['probe_controls'][int(x[1985:2001].argmax())]
            controls=[];costs=[];responses=[]
            for a,model in s['models'].items():
                tick=time.perf_counter();h=model.estimate(x);estimate_time=time.perf_counter()-tick
                for variant in ['original','spatial','joint_block_noise','joint_full_noise']:
                    tick=time.perf_counter()
                    if variant=='original':new=h
                    elif variant=='spatial':new=refine(x,public['pilot_qpsk'],h,s['spatial'][a],'spatial')[0]
                    else:new=refine_joint(x,public['pilot_qpsk'],h,s['spatial'][a],s['frequency'][a],variant=='joint_full_noise')[0]
                    u,_=decode(new,fc,start,sweeps=2)
                    controls.append(np.rint(u*LEVELS).astype(np.int16));responses.append(new)
                    costs.append([16,time.perf_counter()-tick+estimate_time,0.])
            decisions.append(dict(methods=METHODS,control_code=np.asarray(controls),costs=np.asarray(costs),
                                  feedback={},teacher_objective_evaluations=None))
            estimates.append(responses)
    # 决策完成后才读取训练真值；只用于误差诊断与同一接收评分。
    environment=read(data/row['path']/'environment.json')
    from our_method_response_control.physics import training_target
    quality=[];error=[];codes=[]
    for ci,fc in enumerate([4,12,20]):
        scored=score_decisions(environment,fc,public,decisions[ci],'cuda_fft_cpu_rk4_v1',include_mrc=False)
        quality.append(scored['metrics']);codes.append(scored['control_code'])
        truth=training_target(environment,fc)
        error.append(np.sum(abs(np.asarray(estimates[ci])-truth[None])**2,axis=(1,2))/max(np.sum(abs(truth)**2),1e-30))
    dest=s['output']/'records'/('environment_%05d.npz'%row['index'])
    atomic_npz(dest,metrics=np.asarray(quality),control_code=np.asarray(codes),response_relative_error=np.asarray(error))
    mark=dict(index=row['index'],environment_id=row['environment_id'],sha256=sha256(dest),path=dest.name,at=now())
    write_json(dest.with_suffix('.json'),mark);return mark


def screen(project,output,workers):
    manifest=read(project/'dataset_simulation/outputs/scaling_train_3456_20260925/manifest.json')
    selected=set(np.linspace(0,863,48,dtype=int).tolist())
    rows=[r for r in manifest['environments'] if r['split']=='train' and r['index'] in selected]
    if len(rows)!=48: raise ValueError('预设48训练成员缺失。')
    write_json(output/'screen_plan.json',dict(rows=rows,carriers=[4,12,20],methods=METHODS,test_used=False))
    (output/'records').mkdir();records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(str(project),str(output))) as pool:
        for job in as_completed([pool.submit(one,r) for r in rows]):
            records.append(job.result());write_json(output/'progress.json',dict(status='screening_training_only',
                completed=len(records),total=48,pid=os.getpid(),at=now()))
    records.sort(key=lambda r:r['index']);write_json(output/'records.json',records)
    arrays=[];errors=[]
    for r in records:
        path=output/'records'/r['path']
        if sha256(path)!=r['sha256']: raise ValueError('筛查记录改变。')
        with np.load(path) as f:arrays.append(f['metrics']);errors.append(f['response_relative_error'])
    a=np.asarray(arrays);s=sufficient(a);point=metrics(s.sum(0));draws=bootstrap(s,10000,20260926)
    summary=[dict(method=n,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
        response_relative_error=float(np.asarray(errors)[:,:,i].mean()),
        mean_controller_seconds=float(a[:,:,i,11].mean())) for i,n in enumerate(METHODS)]
    contrasts=[]
    for ai,bi in [(4,0),(5,4),(6,5),(7,5),(7,6),(2,0),(3,0),(6,2),(7,3),(6,4),(7,4)]:
        c=np.zeros(8);c[ai]=1;c[bi]=-1
        for field in ['ber','rms_evm_percent','paired_output_snr_db']:
            contrasts.append(dict(a=METHODS[ai],b=METHODS[bi],**contrast(s,draws,c,field)))
    eligible=[METHODS[i] for i in [6,7] if point[i,0]<point[5,0] and point[i,3]<=point[5,3]
              and point[i,0]<point[i-4,0] and point[i,0]<point[4,0]]
    report=dict(scope='training-only hypothesis screen; not generalization evidence',test_used=False,
        methods=summary,comparisons=contrasts,eligible_for_larger_exploratory_evaluation=eligible,
        rule='lower BER and nonworse EVM versus original CNN spatial; both component removals worsen BER',
        selection_final=False,automatic_full_test_started=False)
    write_json(output/'screen_summary.json',clean_json(report))


def run(args):
    require_host();precision();project=args.project.resolve();output=args.output.resolve()
    evaluation=args.after_evaluation.resolve();complete=read(evaluation/'analysis/complete.json')
    if (complete['status']!='complete_exploratory_analysis'
            or read(evaluation/'protocol.json')['common_training_environments']!=3456):
        raise ValueError('须先完成全基线和原始结果审核。')
    for name,digest in complete['file_sha256'].items():
        if sha256(evaluation/'analysis'/name)!=digest: raise ValueError('全基线分析产物改变。')
    sources=dict(read(evaluation/'protocol.json')['source_sha256'])
    sources.update(source_record(['our_method_joint_refinement/'+n for n in
                          ['experiment.py','method.py','PROTOCOL.md','when_ready.py']]))
    verify_sources(sources)
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(at=now(),source_sha256=sources,after_evaluation_complete_sha256=sha256(evaluation/'complete.json'),
        after_analysis_complete_sha256=sha256(evaluation/'analysis/complete.json'),
        test_used=False,train_environments=3456,screen_environments=48,carriers=[4,12,20],
        methods=METHODS,network_updates=0,new_final_test_started=False)
    write_json(output/'protocol.json',protocol)
    for n in sources:
        p=output/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/n,p)
    write_json(output/'progress.json',dict(status='numerical_preflight',at=now(),pid=os.getpid()))
    numerical_check(project,output)
    fit(project,output)
    screen(project,output,args.workers)
    verify_sources(sources)
    verify(project/'dataset_simulation/baseline_results/20260925_full_baselines/fair_3456/runtime_bundle')
    write_json(output/'complete.json',dict(status='complete_training_only_screen',at=now(),
        protocol_sha256=sha256(output/'protocol.json'),summary_sha256=sha256(output/'screen_summary.json'),
        overall_research_complete=False,new_final_test_started=False))
    write_json(output/'progress.json',dict(status='complete_training_only_screen',at=now(),pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['project','output','after-evaluation']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=4);args=p.parse_args()
    try:run(args)
    except BaseException:
        if args.output.exists():write_json(args.output/('failure_%d.json'%time.time()),dict(at=now(),traceback=traceback.format_exc()))
        raise
