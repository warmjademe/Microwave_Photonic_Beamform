"""在旧216环境检验单个响应候选＋码本，保留相同64次反馈参考。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
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
from study_full_baselines.evaluate_learned import METRICS,summarize
from study_full_baselines.train_when_ready import alive
from baseline_common.feedback import FeedbackSession
from our_method_response_control.physics import covariance_estimate
from our_method_feedback_candidates.method import optimize

METHODS=['codebook64','covariance_warm64','cnn_warm64']
SOURCES=['our_method_feedback_candidates/'+n for n in ['evaluate_full.py','method.py','FULL_EVALUATION.md']]
SOURCES+=['study_full_baselines/common.py','study_full_baselines/evaluate_learned.py',
    'baseline_common/feedback.py','baseline_codebook/method.py','baseline_common/controls.py',
    'our_method_response_control/physics.py','our_method_two_stage/control.py']
_STATE=None


def decide(engine,obs,h,seed,fc):
    controls=[];seconds=[];sim_seconds=[];trace=[];scores=[];chosen=[];unique=[]
    for estimate in h:
        def measure(u,call):return engine.measure_detailed(u,rng_for(seed,fc,620,call))['score']
        session=FeedbackSession(engine.cfg,obs,measure,64);tick=time.perf_counter()
        u,info=optimize(session,rng_for(0,seed,fc,630),estimate,fc,'warm_codebook')
        seconds.append(max(0.,time.perf_counter()-tick-session.simulation_feedback_seconds))
        sim_seconds.append(session.simulation_feedback_seconds);controls.append(u)
        trace.append(np.rint(np.asarray(session.controls)*LEVELS).astype(np.int16));scores.append(session.scores)
        chosen.append(int(np.argmax(session.scores)));unique.append(info['unique_response_candidates'])
    return np.asarray(controls),dict(seconds=np.asarray(seconds),sim_seconds=np.asarray(sim_seconds),
        trace=np.asarray(trace),scores=np.asarray(scores),chosen=np.asarray(chosen),unique=np.asarray(unique))


def init(data,classic,bundle):
    global _STATE
    _STATE=dict(data=Path(data),classic=Path(classic),public=public_data(data),bundle=bundle,
        covariance=np.load(bundle['covariance']),cnn=np.load(bundle['cnn'],mmap_mode='r'))


def one(row,output,fp):
    state=_STATE;data=state['data'];public=state['public'];classic=state['classic'];output=Path(output)
    path=output/'records'/('environment_%05d.npz'%row['index']);marker=path.with_suffix('.json')
    if marker.exists():
        meta=json.loads(marker.read_text())
        if meta['fingerprint']!=fp or sha256(path)!=meta['sha256']:raise ValueError('旧接收结果身份改变。')
        return meta
    if path.exists():raise ValueError('未提交结果需先核查。')
    reference=classic/'records'/path.name;ref_meta=json.loads(reference.with_suffix('.json').read_text())
    if ref_meta['environment_id']!=row['environment_id'] or sha256(reference)!=ref_meta['sha256']:
        raise ValueError('原码本参考记录身份不同。')
    ref_protocol=json.loads((classic/'protocol.json').read_text());ri=ref_protocol['methods'].index('codebook')
    with np.load(reference) as f:ref_score=f['metrics'][:,ri];ref_code=f['control_code'][:,ri]
    with np.load(data/row['path']/'data.npz') as f:x=f['X']
    env=json.loads((data/row['path']/'environment.json').read_text())
    values=[];codes=[];traces=[];trace_scores=[];chosen=[];unique=[];sim=[];started=time.perf_counter()
    for ci,fc in enumerate(range(4,21)):
        tick=time.perf_counter()
        traditional=covariance_estimate(x[ci],public['pilot_qpsk'],state['covariance'])
        estimate_seconds=time.perf_counter()-tick
        h=np.stack([traditional,state['cnn'][row['index']*17+ci]])
        observed,_=frame_engine(env,fc,public['pilot_qpsk'],0)
        controls,details=decide(observed,observation(x[ci],public),h,row['seed'],fc)
        controls=np.vstack([ref_code[ci]/LEVELS,controls])
        engine,payload=frame_engine(env,fc,public['pilot_qpsk'],5)
        clean=clean_frame_engine(env,fc,public['pilot_qpsk'],payload)
        quality,code=reception_metrics(engine,clean,controls,payload,row['seed'],fc)
        if (not np.array_equal(code[0],ref_code[ci]) or
                not np.allclose(quality[0],ref_score[ci,:10],rtol=1e-12,atol=0)):
            raise ValueError('原码本64次的接收结果不能逐条复现。')
        seconds=np.r_[ref_score[ci,ref_protocol['metric_order'].index('controller_seconds')],details['seconds']]
        seconds[1]+=estimate_seconds
        values.append(np.column_stack([quality,np.full(3,64),seconds]));codes.append(code)
        traces.append(details['trace']);trace_scores.append(details['scores']);chosen.append(details['chosen'])
        unique.append(details['unique']);sim.append(details['sim_seconds'])
    atomic_npz(path,metrics=np.asarray(values),control_code=np.asarray(codes),
        trace_control_code=np.asarray(traces),trace_scores=np.asarray(trace_scores),
        selected_query=np.asarray(chosen),unique_model_queries=np.asarray(unique),
        feedback_simulator_seconds=np.asarray(sim),environment_id=np.asarray(row['environment_id']))
    record=dict(index=row['index'],environment_id=row['environment_id'],path=path.name,sha256=sha256(path),
        fingerprint=fp,reference_record_sha256=ref_meta['sha256'],reference_replay=True,
        seconds=time.perf_counter()-started,at=now())
    write_json(marker,record);return record


def preflight(project,output):
    require_host()
    if output.exists():raise FileExistsError('前置核验不覆盖。')
    base=project/'dataset_simulation';data=base/'outputs/quality_rank_hybrid_20260925';public=public_data(data)
    cache=base/'diagnostics/20260925_two_stage_train';diagnostic=base/'diagnostics/20260926_feedback_candidates_train'
    rows={r['environment_id']:r for r in check_data(data)['environments'] if r['split']=='train'}
    checks=[]
    for index in [0,2]:
        p=cache/'records'/('%03d.npz'%index);meta=json.loads(p.with_suffix('.json').read_text())
        if sha256(p)!=meta['sha256']:raise ValueError('训练缓存哈希不同。')
        row=rows[meta['environment_id']];fc=meta['carrier_ghz']
        with np.load(p) as f:h=f['response_estimate'];x=f['public_x']
        env=json.loads((data/row['path']/'environment.json').read_text())
        engine,_=frame_engine(env,fc,public['pilot_qpsk'],0)
        u,details=decide(engine,observation(x,public),h,row['seed'],fc)
        ref=diagnostic/'records'/p.name;ref_meta=json.loads(ref.with_suffix('.json').read_text())
        if sha256(ref)!=ref_meta['sha256']:raise ValueError('训练诊断参考哈希不同。')
        with np.load(ref) as f:
            if (not np.array_equal(details['trace'],f['trace_control_code'][[1,3]]) or
                    not np.array_equal(details['scores'],f['trace_scores'][[1,3]]) or
                    not np.array_equal(np.rint(u*LEVELS),f['control_code'][[1,3]])):
                raise ValueError('完整评价控制器未复现训练诊断的每次反馈与最终档位。')
        checks.append(dict(training_environment=row['environment_id'],carrier_ghz=fc,
            both_estimators_full64trace_equal=True,selected_control_equal=True))
    write_json(output,dict(status='passed',checks=checks,source_sha256=source_record(SOURCES),at=now()))
    print(json.dumps(dict(status='passed',checks=checks)),flush=True)


def run(project,output,wait_pid,workers):
    require_host();output.mkdir(parents=True,exist_ok=True)
    base=project/'dataset_simulation';data=base/'outputs/quality_rank_hybrid_20260925'
    study=base/'baseline_results/20260925_full_baselines';classic=study/'classic';sources=source_record(SOURCES)
    wait=dict(source_sha256=sources,wait_pid=wait_pid,workers=workers)
    if (output/'wait_protocol.json').exists():
        if json.loads((output/'wait_protocol.json').read_text())!=wait:raise ValueError('等待协议改变。')
    else:write_json(output/'wait_protocol.json',wait)
    while True:
        f=classic/'progress.json';s=json.loads(f.read_text()) if f.exists() else {}
        if s.get('status')=='complete':break
        if not alive(wait_pid) or list(classic.glob('failure*.json')):raise RuntimeError('原基线评分退出或异常。')
        write_json(output/'progress.json',dict(status='waiting_for_classic',pid=os.getpid(),at=now()));time.sleep(20)
    verify_sources(sources);manifest=check_data(data);rows=[r for r in manifest['environments'] if r['split']=='test']
    if len(rows)!=216:raise ValueError('本入口仅使用旧216环境。')
    from study_full_baselines.audit_results import read_phase,check_summary
    ref_info,ref_arrays=read_phase(classic,rows,sha256(data/'manifest.json'),False)
    check_summary(classic,ref_arrays,rows);del ref_arrays
    target=base/'diagnostics/20260925_response_control_targets';cnn=base/'baseline_results/20260925_response_control'
    cov_meta=json.loads((target/'complete.json').read_text());cnn_meta=json.loads((cnn/'inference.json').read_text())
    artifacts={str(target/'training_covariance.npy'):cov_meta['covariance_sha256'],
        str(cnn/'predicted_response.npy'):cnn_meta['predicted_response_sha256'],
        str(classic/'records.json'):sha256(classic/'records.json')}
    for path,digest in artifacts.items():
        if sha256(path)!=digest:raise ValueError('估计产物或参考索引发生变化。')
    bundle=dict(methods=[dict(name=m,probes=64) for m in METHODS],artifact_sha256=artifacts,
        covariance=str(target/'training_covariance.npy'),cnn=str(cnn/'predicted_response.npy'))
    protocol=dict(bundle=bundle,data_manifest_sha256=sha256(data/'manifest.json'),metric_order=METRICS,
        source_sha256=sources,frame=5,apd_draws=8,scope='old216 exploratory; independent confirmation untouched',
        timing='CPU estimation/controller only; cached CNN forward excluded; original codebook cost from source run')
    fp=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('评分协议改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in sources:
            dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dst)
    (output/'records').mkdir(exist_ok=True);records=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(data),str(classic),bundle)) as pool:
        jobs=[pool.submit(one,row,str(output),fp) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result());state=dict(status='running',completed=len(records),total=216,
                pid=os.getpid(),seconds=time.perf_counter()-started,at=now());write_json(output/'progress.json',state)
            if len(records)%10==0:print(json.dumps(state),flush=True)
    write_json(output/'records.json',sorted(records,key=lambda r:r['index']))
    summarize(output,rows,records,bundle['methods']);verify_sources(sources)
    for path,digest in artifacts.items():
        if sha256(path)!=digest:raise ValueError('估计或参考记录在评分中改变。')
    write_json(output/'progress.json',dict(status='complete',completed=216,total=216,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wait-pid',type=int);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    try:
        if a.preflight:preflight(a.project,a.output)
        else:run(a.project,a.output,a.wait_pid,a.workers)
    except BaseException:
        if not a.preflight:
            a.output.mkdir(parents=True,exist_ok=True)
            write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
