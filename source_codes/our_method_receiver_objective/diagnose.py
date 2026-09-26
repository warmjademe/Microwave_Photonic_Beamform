"""使用既有训练样本定位估计误差和控制目标；额外信息参考明确隔离。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from our_method_feedback_candidates.diagnose import verify_cache, fingerprint
from our_method_response_control.physics import training_target, decode as old_decode, design
from our_method_receiver_objective.control import decode, score_signal

METHODS = ['covariance_mmse', 'covariance_ber', 'cnn_mmse', 'cnn_ber',
           'true_response_mmse_reference', 'true_response_ber_reference', 'teacher_reference']
SOURCES = ['our_method_receiver_objective/'+n for n in ['control.py','diagnose.py','PROTOCOL.md']]
SOURCES += ['our_method_feedback_candidates/diagnose.py', 'our_method_response_control/physics.py',
    'our_method_two_stage/control.py', 'study_full_baselines/common.py',
    'native_sim/control_engine.py', 'native_sim/waveforms.py', 'our_method_quality_rank/generate.py']


def calculate(data, public, row, fc, cache_path, teacher_path):
    with np.load(cache_path) as f:
        x = f['public_x']; estimates = f['response_estimate']
        old_codes = f['control_code'][[0,4]]; old_metrics = f['metrics'][[0,4]]
    with np.load(data/row['path']/'data.npz') as f:
        if not np.array_equal(x, f['X'][fc-4]):
            raise ValueError('缓存公开输入与源数据不同。')
    start = public['probe_controls'][int(x[1985:2001].argmax())]
    # 普通控制器先执行，其函数接口没有环境真值和标签。
    controls = []; infos = []; seconds = []
    for h in estimates:
        for objective in ['mmse','ber']:
            tick = time.perf_counter(); u, info = decode(h,fc,start,objective)
            seconds.append(time.perf_counter()-tick); controls.append(u); infos.append(info)
    if not np.array_equal(np.rint(np.asarray(controls)[[0,2]]*LEVELS), old_codes):
        raise ValueError('新模块MMSE模式未逐码复现原控制。')
    # 仅以下两个诊断参考读取真实传播参数；不可作为部署方法。
    env = json.loads((data/row['path']/'environment.json').read_text())
    oracle_h = training_target(env,fc)
    for objective in ['mmse','ber']:
        tick = time.perf_counter(); u, info = decode(oracle_h,fc,start,objective)
        seconds.append(time.perf_counter()-tick); controls.append(u); infos.append(info)
    with np.load(teacher_path) as f:
        if str(f['environment_id']) != row['environment_id']:
            raise ValueError('教师标签环境身份不同。')
        controls.append(f['control_code'][fc-4]/LEVELS)
        seconds.append(float(f['optimization_seconds'][fc-4]))
    # 全部控制冻结后再打开评分帧；未知数据从未传给控制器。
    engine,payload = frame_engine(env,fc,public['pilot_qpsk'],5)
    clean = clean_frame_engine(env,fc,public['pilot_qpsk'],payload)
    metrics,codes = reception_metrics(engine,clean,controls,payload,row['seed'],fc)
    if not np.allclose(metrics[[0,2]],old_metrics,rtol=1e-12,atol=1e-30):
        raise ValueError('固定接收器未复现旧MMSE指标。')
    if np.any(codes<0) or np.any(codes>LEVELS): raise ValueError('非法控制码。')
    energy = max(float(np.sum(abs(oracle_h)**2)), np.finfo(float).tiny)
    return dict(metrics=metrics,control_code=codes,controller_seconds=np.asarray(seconds),
        response_relative_squared_error=np.sum(abs(estimates-oracle_h)**2,axis=(1,2))/energy),infos


def one_case(job):
    data,public,row,fc,cache_path,teacher_path,output,index,fp,hashes = job
    for p,digest in hashes.items():
        if sha256(p)!=digest:raise ValueError('案例来源已改变。')
    arrays,infos = calculate(data,public,row,fc,cache_path,teacher_path)
    path=output/'records'/('%03d.npz'%index)
    atomic_npz(path,**arrays)
    marker=dict(fingerprint=fp,sha256=sha256(path),case_index=index,
        environment_id=row['environment_id'],carrier_ghz=fc,at=now(),control_details=infos)
    write_json(path.with_suffix('.json'),marker)
    return index


def summaries(a):
    result=[]
    for i,name in enumerate(METHODS):
        m=a['metrics'][:,i]
        result.append(dict(method=name,privileged=i>=4,
            ber=float(m[:,0].sum()/m[:,1].sum()),ser=float(m[:,3].sum()/m[:,4].sum()),
            rms_evm_percent=float(100*np.sqrt(m[:,2].mean())),
            block_error_rate=float(m[:,5].sum()/m[:,6].sum()),
            paired_output_snr_db=float(10*np.log10(m[:,7].sum()/m[:,8].sum())),
            mean_stage_b_seconds=float(a['controller_seconds'][:,i].mean())))
    contrasts=[(3,2),(1,0),(2,0),(3,1),(4,2),(5,4),(6,4)]
    paired=[]; metric=a['metrics'].reshape(24,3,7,10)
    for ai,bi in contrasts:
        ea=metric[:,:,ai,0].sum(1)/metric[:,:,ai,1].sum(1)
        eb=metric[:,:,bi,0].sum(1)/metric[:,:,bi,1].sum(1)
        diff=ea-eb
        rng=np.random.default_rng(20260926)
        boot=diff[rng.integers(0,24,size=(10000,24))].mean(1)
        paired.append(dict(a=METHODS[ai],b=METHODS[bi],ber_difference_pp=float(100*diff.mean()),
            ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),
            wins=int((diff<0).sum()),ties=int((diff==0).sum()),losses=int((diff>0).sum()),
            independent_environments=24,scope='training-only fixed stratified subset; diagnostic uncertainty'))
    return result,paired


def run(project,output,workers,preflight):
    require_host()
    b=project/'dataset_simulation'; data=b/'outputs/quality_rank_hybrid_20260925'
    cache=b/'diagnostics/20260925_two_stage_train'; labels=b/'outputs/unified_teacher_labels_20260925'
    cp,cases=verify_cache(data,cache); public=public_data(data)
    label_complete=json.loads((labels/'complete.json').read_text())
    if label_complete['status']!='complete' or label_complete['train_environments']!=864:
        raise ValueError('原版教师标签未完成。')
    lp=json.loads((labels/'protocol.json').read_text())
    if lp['data_manifest_sha256']!=sha256(data/'manifest.json') or lp['uses_test'] or lp['uses_payload']:
        raise ValueError('教师标签协议不匹配。')
    label_records={r['environment_id']:r for r in json.loads((labels/'records.json').read_text())}
    sources=source_record(SOURCES)
    protocol=dict(scope='training-only mechanism screening, not independent test',
        training_environments=24,carriers_ghz=[4,12,20],cases=72,test_used=False,
        methods=METHODS,privileged_reference_indices=[4,5,6],source_sha256=sources,
        cache_protocol_sha256=sha256(cache/'protocol.json'),weight_sha256=cp['weight_sha256'],
        label_complete_sha256=sha256(labels/'complete.json'),
        data_manifest_sha256=sha256(data/'manifest.json'),metric_order=QUALITY_METRICS,
        frame=5,apd_draws=8,seed=0,coordinate_sweeps=2,extra_feedback=0,
        training_environment_ids=cp['training_environment_ids'],
        comparisons='four public 2x2 variants plus three extra-information references')
    jobs=[]
    for i,(row,fc,path,digest) in enumerate(cases):
        lr=label_records[row['environment_id']];teacher=labels/'records'/lr['path']
        if sha256(teacher)!=lr['sha256']:raise ValueError('教师标签哈希改变。')
        hashes={str(path):digest,str(teacher):lr['sha256'],
            str(data/row['path']/'data.npz'):sha256(data/row['path']/'data.npz'),
            str(data/row['path']/'environment.json'):sha256(data/row['path']/'environment.json')}
        jobs.append((data,public,row,fc,path,teacher,output,i,None,hashes))
    protocol['case_sources']=[j[-1] for j in jobs];fp=fingerprint(protocol)
    jobs=[(*j[:8],fp,j[-1]) for j in jobs]
    if preflight:
        if output.exists():raise FileExistsError(output)
        checks=[]
        for job in jobs[:3]:
            data,public,row,fc,path,teacher,_,_,_,_=job
            with np.load(path) as f:h=f['response_estimate'][1];x=f['public_x']
            initial=public['probe_controls'][int(x[1985:2001].argmax())]
            old,_=old_decode(h,fc,initial);new,_=decode(h,fc,initial,'mmse')
            if not np.array_equal(old,new):raise ValueError('旧解码器不一致。')
            model=design(fc);n=model['antenna_payload_variance'].sum(0);dc=model['dc'].sum()
            if score_signal(model,np.zeros(31,complex),n,dc,'ber') != -.5:
                raise ValueError('零信号误码极限错误。')
            arrays,info=calculate(data,public,row,fc,path,teacher)
            checks.append(dict(carrier_ghz=fc,baseline_controls_identical=True,
                baseline_receiver_metrics_reproduced=True,legal_codes=True,zero_signal_limit=True,
                proxy_monotone=all(v['final_proxy']>=v['initial_proxy']-1e-12 for v in info)))
        verify_sources(sources)
        write_json(output,dict(status='passed',at=now(),checks=checks,source_sha256=sources))
        print(json.dumps(dict(status='passed',checks=checks)),flush=True);return
    output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('诊断协议改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in sources:
            dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/name,dst)
    pending=[];completed=0;started=time.perf_counter()
    for job in jobs:
        path=output/'records'/('%03d.npz'%job[7]);marker=path.with_suffix('.json')
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['fingerprint']!=fp or old['sha256']!=sha256(path):raise ValueError('已提交案例改变。')
            completed+=1
        elif path.exists():raise ValueError('有未提交记录，先核查。')
        else:pending.append(job)
    def progress(status):
        state=dict(status=status,completed=completed,total=72,at=now(),pid=os.getpid(),
            workers=workers,seconds=time.perf_counter()-started)
        write_json(output/'progress.json',state);print(json.dumps(state),flush=True)
    progress('running')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one_case,j) for j in pending]
        for future in as_completed(futures):future.result();completed+=1;progress('running')
    arrays=[]
    for i in range(72):
        path=output/'records'/('%03d.npz'%i)
        if sha256(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:
            raise ValueError('汇总前案例哈希不一致。')
        with np.load(path) as f:arrays.append({k:f[k] for k in f.files})
    stacked={k:np.stack([a[k] for a in arrays]) for k in arrays[0]}
    overall,paired=summaries(stacked)
    verify_sources(sources);verify_cache(data,cache)
    atomic_npz(output/'all_metrics.npz',**stacked)
    write_json(output/'summary.json',dict(status='complete',scope=protocol['scope'],records=overall,paired=paired))
    write_json(output/'complete.json',dict(status='complete',fingerprint=fp,
        all_metrics_sha256=sha256(output/'all_metrics.npz'),summary_sha256=sha256(output/'summary.json'),at=now()))
    progress('complete');print(json.dumps(overall),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    try:run(a.project.resolve(),a.output.resolve(),a.workers,a.preflight)
    except BaseException:
        if not a.preflight:
            a.output.mkdir(parents=True,exist_ok=True)
            write_json(a.output/('failure_%d.json'%time.time()),dict(at=now(),traceback=traceback.format_exc()))
        raise
