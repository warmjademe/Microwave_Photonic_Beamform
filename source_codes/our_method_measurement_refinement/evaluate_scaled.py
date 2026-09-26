"""同3,456训练成员、同16测量：六种估计/校正组合的完整旧测试评分。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.scale_large import fingerprint
from study_full_baselines.evaluate_learned import summarize,METRICS
from our_method_response_control.physics import covariance_estimate,decode
from our_method_measurement_refinement.method import refine

METHODS=[dict(name=a+'__'+v) for a in ['covariance','cnn'] for v in ['original','isotropic','spatial']]
_STATE=None


def bundle(project,fit):
    b=project/'dataset_simulation';train=b/'outputs/scaling_train_3456_20260925'
    test=b/'outputs/quality_rank_hybrid_20260925';scale=b/'baseline_results/20260925_full_baselines/scale_3456'
    checkpoint=scale/'response_n3456_fixed_epochs';cp=json.loads((checkpoint/'complete.json').read_text())
    fp=json.loads((fit/'protocol.json').read_text());fc=json.loads((fit/'complete.json').read_text())
    train_manifest=check_data(train);test_manifest=check_data(test)
    ids=[r['environment_id'] for r in train_manifest['environments'] if r['split']=='train']
    if (len(ids)!=3456 or ids!=fp['train_environment_ids'] or fp['train_environments']!=3456
        or fp['test_used'] or fc['status']!='complete' or fc['fingerprint']!=fingerprint(fp)
        or fp['data_manifest_sha256']!=sha256(train/'manifest.json')
        or fp['weights_sha256']!=cp['weights_sha256']):raise ValueError('共同训练或权重来源不一致。')
    if sha256(train/'public.npz')!=sha256(test/'public.npz'):raise ValueError('公开测量配置不同。')
    if set(ids)&{r['environment_id'] for r in test_manifest['environments'] if r['split']=='test'}:
        raise ValueError('训练测试成员重合。')
    verify_sources(fp['source_sha256'])
    prior_meta=json.loads((scale/'targets/complete.json').read_text())
    paths=dict(prediction=checkpoint/'predicted_response.npy',prior=scale/'targets/training_covariance.npy',
        covariance_error=fit/'covariance_residual_covariance.npy',cnn_error=fit/'cnn_residual_covariance.npy')
    expected=dict(prediction=cp['prediction_sha256'],prior=prior_meta['covariance_sha256'],
        covariance_error=fc['file_sha256']['covariance_residual_covariance.npy'],cnn_error=fc['file_sha256']['cnn_residual_covariance.npy'])
    for k,p in paths.items():
        if sha256(p)!=expected[k]:raise ValueError('模型产物哈希不一致：'+k)
    return dict(paths={k:str(v) for k,v in paths.items()},sha256=expected,
        fit_complete_sha256=sha256(fit/'complete.json'),weights_sha256=cp['weights_sha256'],
        data_manifest_sha256=sha256(test/'manifest.json'),train_environments=3456),test,test_manifest


def init(data,bound):
    global _STATE
    _STATE=dict(data=Path(data),public=public_data(data),bound=bound,
        arrays={k:np.load(p,mmap_mode='r') for k,p in bound['paths'].items()})


def calculate(row,carriers):
    s=_STATE;data=s['data'];public=s['public'];arr=s['arrays']
    with np.load(data/row['path']/'data.npz') as f:raw=f['X']
    decisions=[];all_seconds=[];residuals=[]
    for carrier in carriers:
        x=raw[carrier-4];start=public['probe_controls'][int(x[1985:2001].argmax())]
        tick=time.perf_counter();trad=covariance_estimate(x,public['pilot_qpsk'],arr['prior'])
        estimate_seconds=time.perf_counter()-tick
        cnn=arr['prediction'][row['index']*17+carrier-4]
        controls=[];seconds=[];details=[]
        for name,h in [('covariance',trad),('cnn',cnn)]:
            for mode in ['original','isotropic','spatial']:
                tick=time.perf_counter()
                if mode=='original':new=h;detail=dict(extra_feedback=0)
                else:new,detail=refine(x,public['pilot_qpsk'],h,arr[name+'_error'],mode)
                u,_=decode(new,carrier,start,sweeps=2)
                seconds.append(time.perf_counter()-tick+(estimate_seconds if name=='covariance' else 0))
                controls.append(u);details.append(detail)
        decisions.append(controls);all_seconds.append(seconds);residuals.append(details)
    # 决策全部完成后才打开真实传播参数，仅传给评分器。
    env=json.loads((data/row['path']/'environment.json').read_text());scores=[];codes=[]
    for ci,carrier in enumerate(carriers):
        engine,payload=frame_engine(env,carrier,public['pilot_qpsk'],5)
        clean=clean_frame_engine(env,carrier,public['pilot_qpsk'],payload)
        metrics,code=reception_metrics(engine,clean,decisions[ci],payload,row['seed'],carrier)
        scores.append(np.column_stack([metrics,np.full(6,16),all_seconds[ci]]));codes.append(code)
    return dict(metrics=np.asarray(scores),control_code=np.asarray(codes)),residuals


def one(row,output,fp):
    dest=Path(output)/'records'/('environment_%05d.npz'%row['index']);marker=dest.with_suffix('.json')
    if marker.exists():
        m=json.loads(marker.read_text())
        if m['fingerprint']!=fp or m['sha256']!=sha256(dest):raise ValueError('已提交记录改变。')
        return m
    if dest.exists():raise ValueError('未提交记录须核查。')
    t=time.perf_counter();arrays,details=calculate(row,list(range(4,21)))
    atomic_npz(dest,**arrays,environment_id=np.asarray(row['environment_id']))
    record=dict(index=row['index'],environment_id=row['environment_id'],path=dest.name,
        fingerprint=fp,sha256=sha256(dest),seconds=time.perf_counter()-t,at=now(),refinement_details=details)
    write_json(marker,record);return record


def run(project,fit,output,workers,preflight):
    require_host();bound,data,manifest=bundle(project,fit)
    rows=[r for r in manifest['environments'] if r['split']=='test']
    if len(rows)!=216:raise ValueError('此入口只使用旧216探索环境。')
    sources=source_record(['our_method_measurement_refinement/'+n for n in ['evaluate_scaled.py','method.py','SCALED_PROTOCOL.md']]+
        ['study_full_baselines/common.py','study_full_baselines/evaluate_learned.py','our_method_response_control/physics.py'])
    protocol=dict(scope='old216 exploratory; common3456 training; no new sealed test',
        bundle=bound,methods=METHODS,metric_order=METRICS,frame=5,apd_draws=8,
        initial_measurements=16,extra_feedback=0,source_sha256=sources,
        timing='cached CNN predictions; full online timing remains separately required')
    fp=fingerprint(protocol)
    if preflight:
        if output.exists():raise FileExistsError(output)
        init(str(data),bound);arrays,details=calculate(rows[0],[4,12,20])
        old=project/'dataset_simulation/baseline_results/20260925_full_baselines/evaluation_scaling_3456/records/environment_00000.npz'
        om=json.loads(old.with_suffix('.json').read_text())
        if sha256(old)!=om['sha256']:raise ValueError('旧规模参考哈希错误。')
        with np.load(old) as f:
            reference=f['metrics'][[0,8,16]][:,[0,2],:10]
            ref_codes=f['control_code'][[0,8,16]][:,[0,2]]
        if not np.array_equal(arrays['control_code'][:,[0,3]],ref_codes):raise ValueError('共同原控制不一致。')
        if not np.allclose(arrays['metrics'][:,[0,3],:10],reference,rtol=1e-12,atol=1e-30):
            raise ValueError('共同原接收指标不一致。')
        if np.any(arrays['control_code']<0) or np.any(arrays['control_code']>LEVELS):raise ValueError('控制非法。')
        write_json(output,dict(status='passed',at=now(),cases=3,methods=6,train_environments=3456,
            original_controls_equal=True,original_metrics_equal=True,extra_feedback=0,source_sha256=sources,
            fit_complete_sha256=bound['fit_complete_sha256'],details=details));return
    output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('评测协议改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in sources:
            dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dst)
    records=[];started=time.perf_counter()
    def progress(status):
        state=dict(status=status,completed=len(records),total=216,pid=os.getpid(),at=now(),seconds=time.perf_counter()-started)
        write_json(output/'progress.json',state);print(json.dumps(state),flush=True)
    progress('running')
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(data),bound)) as pool:
        for f in as_completed([pool.submit(one,r,str(output),fp) for r in rows]):records.append(f.result());progress('running')
    write_json(output/'records.json',sorted(records,key=lambda r:r['index']))
    summarize(output,rows,records,METHODS);verify_sources(sources);bundle(project,fit)
    write_json(output/'complete.json',dict(status='complete',at=now(),fingerprint=fp,
        records_sha256=sha256(output/'records.json'),summary_sha256=sha256(output/'summary.json')))
    progress('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','fit','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    try:run(a.project.resolve(),a.fit.resolve(),a.output.resolve(),a.workers,a.preflight)
    except BaseException:
        if not a.preflight:
            a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/('failure_%d.json'%time.time()),dict(at=now(),traceback=traceback.format_exc()));
        raise
