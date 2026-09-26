"""对公开测量校正做训练内筛查，保留原控制器和评价器。"""
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
from our_method_feedback_candidates.diagnose import verify_cache,fingerprint
from our_method_response_control.physics import decode,design,pilot_gain,measurement_noise
from our_method_measurement_refinement.method import refine,project_response

METHODS=['covariance_original','cnn_original','cnn_isotropic_refinement','cnn_spatial_refinement']


def calculate(job):
    data,public,row,fc,path,digest,cov,output,index,fp=job
    if sha256(path)!=digest:raise ValueError('训练缓存哈希改变。')
    with np.load(path) as f:
        x=f['public_x'];h=f['response_estimate'][1]
        original=f['control_code'][[0,4]];reference=f['metrics'][[0,4]]
    with np.load(data/row['path']/'data.npz') as f:
        if not np.array_equal(x,f['X'][fc-4]):raise ValueError('缓存输入改变。')
    initial=public['probe_controls'][int(x[1985:2001].argmax())]
    controls=list(original/LEVELS);info=[];secs=[0.,0.]
    for mode in ['isotropic','spatial']:
        tick=time.perf_counter();new,detail=refine(x,public['pilot_qpsk'],h,cov,mode)
        u,_=decode(new,fc,initial,sweeps=2)
        controls.append(u);info.append(detail);secs.append(time.perf_counter()-tick)
    env=json.loads((data/row['path']/'environment.json').read_text())
    engine,payload=frame_engine(env,fc,public['pilot_qpsk'],5)
    clean=clean_frame_engine(env,fc,public['pilot_qpsk'],payload)
    metrics,codes=reception_metrics(engine,clean,controls,payload,row['seed'],fc)
    if not np.array_equal(codes[:2],original) or not np.allclose(metrics[:2],reference,rtol=1e-12,atol=1e-30):
        raise ValueError('旧控制或接收指标未复现。')
    arrays=dict(metrics=metrics,control_code=codes,correction_and_control_seconds=np.asarray(secs),
        weighted_residuals=np.asarray([[v['weighted_residual_before'],v['weighted_residual_after']] for v in info]))
    if output is not None:
        dest=output/'records'/('%03d.npz'%index);atomic_npz(dest,**arrays)
        write_json(dest.with_suffix('.json'),dict(fingerprint=fp,sha256=sha256(dest),environment_id=row['environment_id'],
            carrier_ghz=fc,at=now(),details=info))
    return arrays


def run(project,fit,output,preflight,workers):
    require_host();b=project/'dataset_simulation';data=b/'outputs/quality_rank_hybrid_20260925'
    cp,cases=verify_cache(data,b/'diagnostics/20260925_two_stage_train');public=public_data(data)
    fcpl=json.loads((fit/'complete.json').read_text());fitp=json.loads((fit/'protocol.json').read_text())
    if fcpl['status']!='complete' or fitp['test_used'] or fitp['network_updates']!=0:
        raise ValueError('残差拟合状态或范围不符。')
    if (fcpl['weights_sha256']!=cp['weight_sha256'] or sha256(fit/'residual_covariance.npy')!=fcpl['covariance_sha256']
        or fitp['data_manifest_sha256']!=sha256(data/'manifest.json')):raise ValueError('残差拟合来源不符。')
    verify_sources(fitp['source_sha256']);cov=np.load(fit/'residual_covariance.npy')
    sources=source_record(['our_method_measurement_refinement/'+n for n in ['method.py','fit.py','diagnose.py','PROTOCOL.md']]+
        ['study_full_baselines/common.py','our_method_response_control/physics.py','our_method_feedback_candidates/diagnose.py'])
    protocol=dict(scope='training-only measurement consistency screening',test_used=False,
        methods=METHODS,training_environments=24,cases=72,carriers_ghz=[4,12,20],frame=5,draws=8,
        fit_complete_sha256=sha256(fit/'complete.json'),cache_protocol_sha256=sha256(b/'diagnostics/20260925_two_stage_train/protocol.json'),
        source_sha256=sources,initial_measurements=16,extra_feedback=0,control='unchanged MMSE two sweeps',
        training_environment_ids=cp['training_environment_ids'],metric_order=QUALITY_METRICS)
    fp=fingerprint(protocol)
    jobs=[(data,public,row,carrier,path,digest,cov,None if preflight else output,i,fp)
        for i,(row,carrier,path,digest) in enumerate(cases)]
    if preflight:
        if output.exists():raise FileExistsError(output)
        evidence=[]
        for job in jobs[:3]:
            _,_,row,carrier,path,_,_,_,_,_=job
            with np.load(path) as f:h=f['response_estimate'][1];x=f['public_x']
            m=design(carrier);a=m['sensing'];r=measurement_noise(m,x,public['pilot_qpsk'])
            c=cov[carrier-4]*max(float(np.mean(abs(h)**2)),1e-24)
            exact=np.einsum('kpn,nk->kp',a,h)
            if not np.array_equal(project_response(h,exact,a,c,r),h):raise ValueError('零残差不变性失败。')
            y=pilot_gain(x,public['pilot_qpsk']).T
            phase=np.exp(.73j)
            z=project_response(h,y,a,c,r)
            if not np.allclose(project_response(h*phase,y*phase,a,c,r),z*phase,rtol=1e-10,atol=1e-15):
                raise ValueError('相位等变检查失败。')
            arrays=calculate(job)
            if np.any(arrays['control_code']<0) or np.any(arrays['control_code']>LEVELS):raise ValueError('控制非法。')
            evidence.append(dict(carrier_ghz=carrier,zero_residual_identity=True,common_phase_equivariance=True,
                original_metrics_reproduced=True,legal_codes=True,weighted_residuals=arrays['weighted_residuals'].tolist()))
        write_json(output,dict(status='passed',at=now(),checks=evidence,source_sha256=sources))
        print(json.dumps(dict(status='passed',checks=evidence)),flush=True);return
    output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('协议改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in sources:
            dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dst)
    pending=[];completed=0;start=time.perf_counter()
    for j in jobs:
        f=output/'records'/('%03d.npz'%j[-2]);marker=f.with_suffix('.json')
        if marker.exists():
            meta=json.loads(marker.read_text())
            if meta['fingerprint']!=fp or meta['sha256']!=sha256(f):raise ValueError('已提交结果改变。')
            completed+=1
        elif f.exists():raise ValueError('未提交记录须核查。')
        else:pending.append(j)
    def progress(status):
        s=dict(status=status,completed=completed,total=72,at=now(),pid=os.getpid(),seconds=time.perf_counter()-start)
        write_json(output/'progress.json',s);print(json.dumps(s),flush=True)
    progress('running')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(calculate,j) for j in pending]):f.result();completed+=1;progress('running')
    items=[]
    for i in range(72):
        path=output/'records'/('%03d.npz'%i)
        if sha256(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise ValueError('案例哈希不符。')
        with np.load(path) as f:items.append({k:f[k] for k in f.files})
    arrays={k:np.stack([v[k] for v in items]) for k in items[0]};m=arrays['metrics'];records=[]
    for i,method in enumerate(METHODS):
        z=m[:,i];records.append(dict(method=method,ber=float(z[:,0].sum()/z[:,1].sum()),
            ser=float(z[:,3].sum()/z[:,4].sum()),rms_evm_percent=float(100*np.sqrt(z[:,2].mean())),
            paired_output_snr_db=float(10*np.log10(z[:,7].sum()/z[:,8].sum()))))
    pairs=[];cube=m.reshape(24,3,4,10)
    for ai,bi in [(2,1),(3,1),(3,2)]:
        diff=cube[:,:,ai,0].sum(1)/cube[:,:,ai,1].sum(1)-cube[:,:,bi,0].sum(1)/cube[:,:,bi,1].sum(1)
        rng=np.random.default_rng(20260926);boot=diff[rng.integers(0,24,size=(10000,24))].mean(1)*100
        pairs.append(dict(a=METHODS[ai],b=METHODS[bi],ber_difference_pp=float(diff.mean()*100),
            ci95_pp=np.quantile(boot,[.025,.975]).tolist(),wins=int((diff<0).sum()),ties=int((diff==0).sum()),losses=int((diff>0).sum())))
    verify_sources(sources);verify_cache(data,b/'diagnostics/20260925_two_stage_train')
    atomic_npz(output/'all_metrics.npz',**arrays)
    write_json(output/'summary.json',dict(status='complete',scope=protocol['scope'],records=records,paired=pairs))
    write_json(output/'complete.json',dict(status='complete',at=now(),fingerprint=fp,
        metrics_sha256=sha256(output/'all_metrics.npz'),summary_sha256=sha256(output/'summary.json')))
    progress('complete');print(json.dumps(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['project','fit','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    try:run(a.project.resolve(),a.fit.resolve(),a.output.resolve(),a.preflight,a.workers)
    except BaseException:
        if not a.preflight:
            a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/('failure_%d.json'%time.time()),dict(at=now(),traceback=traceback.format_exc()))
        raise
