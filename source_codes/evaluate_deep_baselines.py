"""用相同冻结器件和独立QPSK数据块，评价六个网络已经输出的控制。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from compact_dataset import CompactDataset, sha256
from native_sim.data import NativeDataset
from native_sim.control_engine import NativeControlEngine
from native_sim.evaluation import evaluate_record
from baseline_common.config import rng_for
from run_compact_baselines import evaluate_one, METRICS
from run_native_baselines import verify_core
from generate_native_dataset import write_json, json_bytes, now

STATE=None


class FixedPrediction:
    def __init__(self, control): self.control=control
    def predict(self, observation, cfg): return self.control.copy()


def initialize(test_root,root,methods):
    global STATE
    data=CompactDataset(test_root); native=NativeDataset(data.source_root)
    STATE=dict(data=data,native=native,rows={r['environment_id']:r for r in native.environments('test')},
        methods=methods,predictions={m:np.load(Path(root)/m/'predictions.npy',mmap_mode='r') for m in methods},
        latency={m:json.loads((Path(root)/m/'metadata.json').read_text())['inference_single_mean_seconds'] for m in methods})


def environment(index,output,fingerprint):
    s=STATE; d=s['data']; root=Path(output)
    path=root/'records'/('environment_%05d.npz'%index)
    if path.with_suffix('.json').exists():
        marker=json.loads(path.with_suffix('.json').read_text())
        if marker['protocol_fingerprint']!=fingerprint or sha256(path)!=marker['sha256']:
            raise ValueError('既有结果协议/SHA不同。')
        return index
    if path.exists(): raise ValueError('存在未提交结果，须核查：'+str(path))
    row=d.environments[index]; native_row=s['rows'][row['environment_id']]
    metrics=np.empty((len(s['methods']),1,17,len(METRICS)))
    controls=np.empty((len(s['methods']),1,17,128),np.int16)
    examples={}; started=time.perf_counter()
    for ci,carrier in enumerate(d.metadata['carriers_ghz']):
        sample=index*17+ci
        if sha256(d.source_record(sample))!=bytes(d.record_sha256[sample]).hex():
            raise ValueError('原始记录与公开输入来源不同。')
        obs=d.observation(sample); truth=s['native'].simulator_truth(native_row,ci)
        engine=NativeControlEngine(d.cfg,truth['branch_band_w'],truth['branch_dc_w'],carrier*1e9,obs['pilot_qpsk'])
        target=np.asarray(d.Y_code[sample],float)/np.r_[np.full(64,76),np.full(64,24)]
        for mi,method in enumerate(s['methods']):
            control=s['predictions'][method][sample]
            # 复用原评测函数的MLP固定输入分支，接口包装只返回预先写盘的网络预测。
            values,code=evaluate_one(engine,obs,truth,target,'mlp',0,native_row['seed'],16,FixedPrediction(control))
            values[METRICS.index('controller_wall_seconds')]=s['latency'][method]
            values[METRICS.index('estimated_control_latency_seconds')]=16*d.cfg.measurement_s+d.cfg.switch_s+s['latency'][method]
            metrics[mi,0,ci]=values; controls[mi,0,ci]=code
            if index==0 and carrier in [4,8,12,20]:
                detail=evaluate_record(engine,engine.project(control),truth['payload_qpsk'],rng_for(native_row['seed'],carrier,0,510))
                prefix=method+'_'+str(carrier)
                examples[prefix+'_received']=detail['received_qpsk']
                examples[prefix+'_iq']=detail['iq_a']
                examples['target_'+str(carrier)]=truth['payload_qpsk']
    if examples:
        np.savez_compressed(root/'signal_examples.npz',**examples)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,metrics=metrics,control_code=controls,protocol_fingerprint=fingerprint,
                            environment_id=row['environment_id'])
    temporary.replace(path)
    write_json(path.with_suffix('.json'),dict(protocol_fingerprint=fingerprint,sha256=sha256(path),
        environment_id=row['environment_id'],seconds=time.perf_counter()-started,completed_at_utc=now()))
    return index


def run(test_root,root,workers=4):
    root=Path(root).resolve(); out=root/'evaluation'; out.mkdir(exist_ok=True);(out/'records').mkdir(exist_ok=True)
    study=json.loads((root/'study_protocol.json').read_text()); methods=study['methods']
    fp_study=hashlib.sha256(json_bytes(study)).hexdigest()
    data=CompactDataset(test_root,verify_hashes=True)
    if data.split!='test' or sha256(data.root/'manifest.json')!=study['test_manifest_sha256']:
        raise ValueError('测试集与预定协议不同。')
    for name,digest in study['source_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest: raise ValueError('源码改变：'+name)
    verify_core(NativeDataset(data.source_root))
    checks={}
    for method in methods:
        marker=json.loads((root/method/'complete.json').read_text())
        if marker['protocol_fingerprint']!=fp_study: raise ValueError('训练尚未完成/版本不同。')
        for name,digest in marker['file_sha256'].items():
            if sha256(root/method/name)!=digest: raise ValueError('模型/预测SHA不符。')
        meta=json.loads((root/method/'metadata.json').read_text())
        if meta['epochs']!=40 or meta['seed']!=0: raise ValueError('训练次数不同。')
        if meta['normalization_sha256']!=sha256(root/'normalization.npz'): raise ValueError('标准化改变。')
        checks[method]=marker
    protocol=dict(methods=methods,seeds=[0],environment_count=len(data.environments),sample_count=len(data),
        metric_order=METRICS,carriers_ghz=data.metadata['carriers_ghz'],checkpoints=checks,
        study_fingerprint=fp_study,test_manifest_sha256=study['test_manifest_sha256'],
        test_dataset=str(data.root),signal_examples='fixed environment index 0, carriers 4/8/12/20; not outcome selected')
    fingerprint=hashlib.sha256(json_bytes(protocol)).hexdigest()
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise ValueError('评分协议改变。')
    write_json(out/'protocol.json',protocol)
    progress=dict(status='running',completed_environments=0,total_environments=len(data.environments),
                  protocol_fingerprint=fingerprint,started_at_utc=now(),workers=workers)
    started=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,
                                 initargs=(str(test_root),str(root),methods)) as pool:
            iterator=iter(range(len(data.environments))); pending={}
            def submit():
                i=next(iterator,None)
                if i is not None: pending[pool.submit(environment,i,str(out),fingerprint)]=i
            for _ in range(workers):submit()
            while pending:
                finished,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for task in finished:
                    i=pending.pop(task)
                    try:task.result()
                    except BaseException:
                        write_json(out/('failure_%05d.json'%i),dict(traceback=traceback.format_exc()));raise
                    progress['completed_environments']+=1;submit()
                progress.update(elapsed_seconds=time.perf_counter()-started,updated_at_utc=now())
                write_json(out/'progress.json',progress)
                if finished:print(json.dumps(progress),flush=True)
        for i in range(len(data.environments)):
            p=out/'records'/('environment_%05d.npz'%i); marker=json.loads(p.with_suffix('.json').read_text())
            if marker['protocol_fingerprint']!=fingerprint or sha256(p)!=marker['sha256']: raise ValueError('最终SHA失败。')
        progress.update(status='complete',final_sha_verification=True,finished_at_utc=now(),
            signal_examples_sha256=sha256(out/'signal_examples.npz'))
    except BaseException:
        progress.update(status='failed_or_interrupted',traceback=traceback.format_exc());raise
    finally:write_json(out/'progress.json',progress)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--test',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.test,a.output,a.workers)
