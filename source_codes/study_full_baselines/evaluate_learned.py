"""把重训七网络、现有学习路线、固定先验和规模对照送入同一接收器。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from deep_common.preprocessing import apply
from our_method_quality_rank.common import METHODS as QUALITY_METHODS
from our_method_response_control.physics import ridge_estimate,covariance_estimate,decode
from baseline_frequency_prior.method import fit as fit_prior,predict_indices

METRICS=QUALITY_METRICS+['feedback_calls','decoder_cpu_seconds']
_STATE=None


def build_bundle(project,data,study):
    """只读已完成检查点及训练先验，记录每份预测来源；不生成新模型。"""
    methods=[];artifacts={}
    def bind(path,expected=None):
        path=Path(path);actual=sha256(path)
        if expected is not None and expected!=actual:raise ValueError('预测或训练统计哈希不同：'+str(path))
        artifacts[str(path)]=actual;return str(path)
    training=study/'learned'
    if json.loads((training/'progress.json').read_text())['status']!='complete':
        raise ValueError('七网络训练尚未完成。')
    for method in ['mlp']+DEEP_METHODS:
        f=training/method;meta=json.loads((f/'complete.json').read_text())
        methods.append(dict(name=method,kind='control',path=bind(f/'predictions.npy',meta['predictions_sha256']),probes=16))
    qr=project/'dataset_simulation/baseline_results/20260925_quality_rank_hybrid'
    for method in QUALITY_METHODS:
        f=qr/method;meta=json.loads((f/'inference.json').read_text())
        methods.append(dict(name='quality_'+method,kind='control',path=bind(f/'predictions.npy',meta['predictions_sha256']),probes=16))
    target=project/'dataset_simulation/diagnostics/20260925_response_control_targets'
    metadata=json.loads((target/'complete.json').read_text())
    response=project/'dataset_simulation/baseline_results/20260925_response_control'
    inference=json.loads((response/'inference.json').read_text())
    methods.extend([
        dict(name='ridge_response',kind='ridge',probes=16),
        dict(name='covariance_response',kind='covariance',probes=16,path=bind(target/'training_covariance.npy',metadata['covariance_sha256'])),
        dict(name='complex_response_cnn',kind='response',probes=16,path=bind(response/'predicted_response.npy',inference['predicted_response_sha256']))])
    scale=study/'scale_small'
    if json.loads((scale/'progress.json').read_text())['status']!='complete':raise ValueError('规模模型尚未完成。')
    for n in [216,432]:
        methods.append(dict(name='covariance_n%04d'%n,kind='covariance',probes=16,path=bind(scale/('covariance_%04d.npy'%n))))
        for schedule in ['fixed_epochs','equal_updates']:
            name='response_n%04d_%s'%(n,schedule);folder=scale/name
            meta=json.loads((folder/'complete.json').read_text())
            methods.append(dict(name=name,kind='response',probes=16,path=bind(folder/'predicted_response.npy',meta['prediction_sha256'])))
    _,_,robust,train_rows=load_split(data,'train',labels=True)
    prior=fit_prior(robust.reshape(len(train_rows),17,64))
    indices=predict_indices(prior,np.arange(4,21))
    with np.load(data/'public.npz') as f:catalog=f['catalog_controls']
    for i,name in enumerate(['global_prior','frequency_prior']):
        methods.append(dict(name=name,kind='fixed',probes=0,controls=catalog[indices[:,i]].tolist()))
    methods.append(dict(name='public16',kind='public',probes=16))
    return dict(methods=methods,artifact_sha256=artifacts,prior_fit_training_only=True,
        timing_scope='Per-sample CPU decoder only here; all-method end-to-end software timing collected separately, not treated as zero inference cost',
        train_ids=[r['environment_id'] for r in train_rows])


def init(data_string,bundle):
    global _STATE
    arrays={m['name']:np.load(m['path'],mmap_mode='r') for m in bundle['methods'] if 'path' in m}
    _STATE=dict(data=Path(data_string),public=public_data(data_string),bundle=bundle,arrays=arrays)


def choose_controls(x,sample,public,methods,arrays):
    """在线边界：没有环境、多径真值、数据符号或测试标签参数。"""
    fc=int(x[1984]);initial=public['probe_controls'][int(x[1985:2001].argmax())]
    controls=[];times=[]
    for m in methods:
        tick=time.perf_counter();kind=m['kind']
        if kind=='control':u=arrays[m['name']][sample]
        elif kind=='fixed':u=np.asarray(m['controls'])[fc-4]
        elif kind=='public':u=initial
        else:
            if kind=='ridge':h=ridge_estimate(x,public['pilot_qpsk'])[0]
            elif kind=='covariance':h=covariance_estimate(x,public['pilot_qpsk'],arrays[m['name']])
            elif kind=='response':h=arrays[m['name']][sample]
            else:raise ValueError('未知控制种类。')
            u,_=decode(h,fc,initial,sweeps=2)
        controls.append(np.floor(np.clip(u,0,1)*LEVELS+.5)/LEVELS)
        # 读取缓存预测不是网络推理计时；正式耗时由单独在线回放测量。
        times.append(time.perf_counter()-tick if kind in ['ridge','covariance','response'] else np.nan)
    return np.asarray(controls),np.asarray(times)


def one(row,output_string,fingerprint):
    state=_STATE;data=state['data'];public=state['public'];methods=state['bundle']['methods']
    output=Path(output_string);path=output/'records'/('environment_%05d.npz'%row['index']);marker=path.with_suffix('.json')
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['fingerprint']!=fingerprint or sha256(path)!=old['sha256']:raise ValueError('既有学习方法评分改变。')
        return old
    if path.exists():raise ValueError('有未提交产物，需先核查。')
    with np.load(data/row['path']/'data.npz') as f:x=f['X']
    started=time.perf_counter()
    decisions=[choose_controls(x[ci],row['index']*17+ci,public,methods,state['arrays']) for ci in range(17)]
    # 全载频控制已经冻结后，才打开真值和评价帧。
    env=json.loads((data/row['path']/'environment.json').read_text())
    metrics=[];codes=[]
    for ci,fc in enumerate(range(4,21)):
        controls,times=decisions[ci]
        engine,payload=frame_engine(env,fc,public['pilot_qpsk'],5)
        clean=clean_frame_engine(env,fc,public['pilot_qpsk'],payload)
        scores,code=reception_metrics(engine,clean,controls,payload,row['seed'],fc)
        metrics.append(np.column_stack([scores,[m['probes'] for m in methods],times]));codes.append(code)
    atomic_npz(path,metrics=np.asarray(metrics),control_code=np.asarray(codes),environment_id=np.asarray(row['environment_id']))
    record=dict(index=row['index'],environment_id=row['environment_id'],sha256=sha256(path),path=path.name,
        fingerprint=fingerprint,seconds=time.perf_counter()-started,at=now())
    write_json(marker,record);return record


def summarize(output,rows,records,methods):
    metrics=[];control_codes=[]
    for r in sorted(records,key=lambda r:r['index']):
        path=output/'records'/r['path']
        if sha256(path)!=r['sha256']:raise ValueError('原始评测文件改变。')
        with np.load(path) as f:metrics.append(f['metrics']);control_codes.append(f['control_code'])
    values=np.asarray(metrics);codes=np.asarray(control_codes)
    if values.shape!=(len(rows),17,len(methods),len(METRICS)) or not np.isfinite(values[...,:11]).all():
        raise ValueError('指标不完整或有非有限值。')
    if np.any(codes<0) or np.any(codes>LEVELS):raise ValueError('控制码越界。')
    groups=[('all',None,np.ones(len(rows),bool),None)]
    groups += [('carrier_ghz',fc,np.ones(len(rows),bool),fc-4) for fc in range(4,21)]
    for factor in ['rays','max_delay_ns','angular_std_deg','power_bin']:
        for value in sorted({r['factors'][factor] for r in rows}):
            groups.append((factor,value,np.asarray([r['factors'][factor]==value for r in rows]),None))
    summaries=[]
    for mi,m in enumerate(methods):
        for group,group_value,mask,ci in groups:
            a=values[mask,:,mi]
            if ci is not None:a=a[:,ci:ci+1]
            nmse=float(a[...,2].mean())
            summaries.append(dict(method=m['name'],group=group,value=group_value,environments=int(mask.sum()),
                ber=float(a[...,0].sum()/a[...,1].sum()),ser=float(a[...,3].sum()/a[...,4].sum()),
                block_error_rate=float(a[...,5].sum()/a[...,6].sum()),mean_nmse=nmse,
                rms_evm_percent=100*np.sqrt(nmse),effective_snr_db=-10*np.log10(max(nmse,1e-30)),
                paired_output_snr_db=float(10*np.log10(a[...,7].sum()/a[...,8].sum())),
                mean_optical_dc_w=float(a[...,9].mean()),mean_feedback_calls=float(a[...,10].mean()),
                mean_decoder_seconds=float(a[...,11].mean()) if np.isfinite(a[...,11]).all() else None,
                full_online_timing_pending=True))
    write_json(output/'summary.json',dict(status='complete',records=summaries,
        scope='exploratory; current test already viewed in earlier study',at=now()))
    print(json.dumps([r for r in summaries if r['group']=='all']),flush=True)


def run(project,data,study,output,workers):
    require_host();manifest=check_data(data);bundle=build_bundle(project,data,study)
    rows=[r for r in manifest['environments'] if r['split']=='test']
    protocol=dict(bundle=bundle,data_manifest_sha256=sha256(data/'manifest.json'),metric_order=METRICS,
        source_sha256=source_record(['study_full_baselines/evaluate_learned.py','study_full_baselines/common.py',
            'our_method_response_control/physics.py','baseline_frequency_prior/method.py']),
        frame=5,apd_draws=8)
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('协议发生变化。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in protocol['source_sha256']:
            dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
    records=[];started=time.perf_counter()
    write_json(output/'progress.json',dict(status='running',completed=0,total=len(rows),pid=os.getpid(),at=now()))
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(data),bundle)) as pool:
        jobs=[pool.submit(one,row,str(output),fingerprint) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result())
            progress=dict(status='running',completed=len(records),total=len(rows),
                seconds=time.perf_counter()-started,pid=os.getpid(),at=now())
            write_json(output/'progress.json',progress)
            if len(records)%10==0:print(json.dumps(progress),flush=True)
    write_json(output/'records.json',sorted(records,key=lambda r:r['index']))
    summarize(output,rows,records,bundle['methods']);verify_sources(protocol['source_sha256'])
    for path,digest in bundle['artifact_sha256'].items():
        if sha256(path)!=digest:raise ValueError('预测在评测中改变。')
    write_json(output/'progress.json',dict(status='complete',completed=len(rows),total=len(rows),at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','data','study','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    try:run(a.project,a.data,a.study,a.output,a.workers)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
