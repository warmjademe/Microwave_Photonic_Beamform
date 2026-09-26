"""原六项在线传统方法与教师/MRC参考，在新数据上重新执行完整多指标评价。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
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
from baseline_common.feedback import FeedbackSession
from baseline_common.channel import deserialize_environment, response
from baseline_common.config import KB
from native_sim.control_engine import optimize_teacher

METHODS=ONLINE_CLASSIC+['teacher','mrc']
METRICS=QUALITY_METRICS+['feedback_calls','controller_seconds','feedback_simulator_seconds']


def digital_reference(environment,carrier):
    """实际调用原MRC，并重放其独立QPSK链路以补充SER/块错误率。

    MRC使用不同数字硬件和真实CSI；其随机payload与光子评价不同，单列。
    不把数字MRC的信号/噪声功率写入光电流单位字段。
    """
    from baseline_mrc.method import evaluate
    from baseline_common.metrics import evaluate_output
    cfg=NativeConfig();h=response(deserialize_environment(environment),cfg,carrier*1e9)
    power_dbm=environment['power_dbm'];seed=environment['seed']
    norm=np.sqrt(np.sum(abs(h)**2,axis=0))
    weights=h.conj()/np.maximum(norm,1e-30)
    power=1e-3*10**(power_dbm/10)/cfg.tones
    gain=np.sqrt(power)*np.sum(weights*h,axis=0)
    variance=KB*cfg.temperature_k*10**(.2)*cfg.bandwidth_hz/cfg.tones
    errors=ser=blocks=0;nmse=0.;physical=[]
    for draw in range(8):
        args=(seed,carrier,105,5,draw)
        original=evaluate(h,power_dbm,cfg,rng_for(*args),payload_symbols=1)
        detail=evaluate_output(gain,np.full(31,variance),cfg,rng_for(*args),
                               payload_symbols=1,keep_arrays=True)
        if original['bit_errors']!=detail['bit_errors'] or original['evm_percent']!=detail['evm_percent']:
            raise ValueError('MRC补充指标回放未复现原方法。')
        arr=detail['arrays'];sent=arr['sent_qpsk'];got=arr['equalized']
        wrong=(sent.real>=0)!=(got.real>=0)
        wrong|=(sent.imag>=0)!=(got.imag>=0)
        errors+=original['bit_errors'];ser+=int(wrong.sum());blocks+=int(wrong.any())
        nmse+=(original['evm_percent']/100)**2;physical.append(original['snr_db'])
    v=np.full(len(METRICS),np.nan)
    v[:7]=[errors,496,nmse/8,ser,248,blocks,8]
    return v,float(np.mean(physical))


def one(data_string,output_string,row,fingerprint):
    data,output=Path(data_string),Path(output_string)
    path=output/'records'/('environment_%05d.npz'%row['index']);marker=path.with_suffix('.json')
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['fingerprint']!=fingerprint or sha256(path)!=old['sha256']:
            raise ValueError('旧评测来源或哈希改变。')
        return old
    if path.exists():raise ValueError('未提交产物必须核查：'+str(path))
    env=json.loads((data/row['path']/'environment.json').read_text());public=public_data(data)
    with np.load(data/row['path']/'data.npz') as f:x=f['X']
    values=[];codes=[];mrc_snr=[];teacher_calls=[];started=time.perf_counter()
    for ci,fc in enumerate(range(4,21)):
        observed_engine,_=frame_engine(env,fc,public['pilot_qpsk'],0)
        obs=observation(x[ci],public);controls=[];cost=[]
        for method in ONLINE_CLASSIC:
            def measure(u,call):
                return observed_engine.measure_detailed(u,rng_for(row['seed'],fc,620,call))['score']
            session=FeedbackSession(observed_engine.cfg,obs,measure,64)
            tick=time.perf_counter()
            u=importlib.import_module('baseline_'+method+'.method').optimize(
                session,rng_for(0,row['seed'],fc,630))
            duration=time.perf_counter()-tick
            if session.calls>64:raise ValueError('反馈超预算。')
            controls.append(observed_engine.project(u))
            cost.append([session.calls,max(0.,duration-session.simulation_feedback_seconds),
                         session.simulation_feedback_seconds])
        tick=time.perf_counter()
        teacher,info=optimize_teacher(observed_engine,public['probe_controls'],
            rng_for(0,row['seed'],fc,610),starts=2,sweeps=1)
        controls.append(teacher);cost.append([np.nan,time.perf_counter()-tick,0.])
        teacher_calls.append(info['objective_evaluations'])
        # 所有在线控制已经决定；现在才使用独立第5帧评分。
        scoring_engine,payload=frame_engine(env,fc,public['pilot_qpsk'],5)
        clean=clean_frame_engine(env,fc,public['pilot_qpsk'],payload)
        score,code=reception_metrics(scoring_engine,clean,controls,payload,row['seed'],fc)
        mrc,snr=digital_reference(env,fc)
        values.append(np.vstack([np.column_stack([score,cost]),mrc]))
        codes.append(np.vstack([code,np.full(128,-1,np.int16)]));mrc_snr.append(snr)
    atomic_npz(path,metrics=np.asarray(values),control_code=np.asarray(codes),
        mrc_physical_reference_snr_db=np.asarray(mrc_snr),
        teacher_objective_evaluations=np.asarray(teacher_calls),
        environment_id=np.asarray(row['environment_id']))
    result=dict(index=row['index'],environment_id=row['environment_id'],path=path.name,
        sha256=sha256(path),fingerprint=fingerprint,seconds=time.perf_counter()-started,at=now())
    write_json(marker,result);return result


def summarize(output,rows,records):
    arrays=[]
    for r in sorted(records,key=lambda r:r['index']):
        path=output/'records'/r['path']
        if sha256(path)!=r['sha256']:raise ValueError('原始结果哈希不一致。')
        with np.load(path) as f:
            arrays.append(f['metrics'])
            code=f['control_code'][:,:-1]
            if np.any(code<0) or np.any(code>LEVELS):raise ValueError('非法控制码。')
    values=np.asarray(arrays)
    if values.shape!=(len(rows),17,len(METHODS),len(METRICS)) or not np.isfinite(values[...,:7]).all():
        raise ValueError('接收评分缺失。')
    summaries=[]
    groups=[('all',None,np.ones(len(rows),bool),None)]
    groups += [('carrier_ghz',fc,np.ones(len(rows),bool),fc-4) for fc in range(4,21)]
    for factor in ['rays','max_delay_ns','angular_std_deg','power_bin']:
        for value in sorted({r['factors'][factor] for r in rows}):
            groups.append((factor,value,np.asarray([r['factors'][factor]==value for r in rows]),None))
    for mi,method in enumerate(METHODS):
        for group,group_value,mask,ci in groups:
            a=values[mask,:,mi]
            if ci is not None:a=a[:,ci:ci+1]
            nmse=float(a[...,2].mean())
            record=dict(method=method,group=group,value=group_value,environments=int(mask.sum()),
                ber=float(a[...,0].sum()/a[...,1].sum()),ser=float(a[...,3].sum()/a[...,4].sum()),
                block_error_rate=float(a[...,5].sum()/a[...,6].sum()),mean_nmse=nmse,
                rms_evm_percent=100*np.sqrt(nmse),effective_snr_db=-10*np.log10(max(nmse,1e-30)),
                comparable_hardware=method!='mrc',privileged=method in ['teacher','mrc'])
            if method!='mrc':
                record.update(paired_output_snr_db=float(10*np.log10(a[...,7].sum()/a[...,8].sum())),
                    mean_optical_dc_w=float(a[...,9].mean()),mean_controller_seconds=float(a[...,11].mean()))
            if method not in ['teacher','mrc']:
                record.update(mean_feedback_calls=float(a[...,10].mean()),
                    mean_estimated_control_seconds=float((a[...,11]+a[...,10]*NativeConfig().measurement_s+NativeConfig().switch_s).mean()))
            summaries.append(record)
    write_json(output/'summary.json',dict(status='complete',records=summaries,methods=METHODS,
        metric_order=METRICS,scope='exploratory rerun on previously viewed216 test environments',
        mrc_scope='same propagation, original separate ideal digital hardware, own pilots and payload; no photonic controls',at=now()))
    print(json.dumps([r for r in summaries if r['group']=='all']),flush=True)


def run(data,output,workers):
    require_host();manifest=check_data(data)
    rows=[r for r in manifest['environments'] if r['split']=='test']
    names=['study_full_baselines/common.py','study_full_baselines/evaluate_classic.py',
        'study_full_baselines/PROTOCOL.md','native_sim/control_engine.py','baseline_common/feedback.py',
        'baseline_common/metrics.py']+['baseline_'+m+'/method.py' for m in METHODS]
    protocol=dict(methods=METHODS,metric_order=METRICS,budget=64,seed=0,frame=5,draws=8,
        data_manifest_sha256=sha256(data/'manifest.json'),source_sha256=source_record(names),
        test_ids=[r['environment_id'] for r in rows],original_teacher_backend='native_sim.control_engine.optimize_teacher')
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('评测协议改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in names:
            dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/name,dest)
    started=time.perf_counter();records=[]
    write_json(output/'progress.json',dict(status='running',completed=0,total=len(rows),pid=os.getpid(),at=now()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(one,str(data),str(output),row,fingerprint) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result())
            progress=dict(status='running',completed=len(records),total=len(rows),pid=os.getpid(),
                seconds=time.perf_counter()-started,at=now())
            write_json(output/'progress.json',progress)
            if len(records)%10==0:print(json.dumps(progress),flush=True)
    write_json(output/'records.json',sorted(records,key=lambda r:r['index']))
    summarize(output,rows,records);verify_sources(protocol['source_sha256']);check_data(data)
    write_json(output/'progress.json',dict(status='complete',completed=len(rows),total=len(rows),at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    try:run(a.data,a.output,a.workers)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
