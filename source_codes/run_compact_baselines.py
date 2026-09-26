"""在同一轻量测试集运行现有基线；原始缓存仅由统一反馈/评分器读取。

逐环境原子保存，绑定协议、输入、源码与检查点；完整测试集和固定种子不挑样本。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import numpy as np
from compact_dataset import CompactDataset,sha256
from native_sim.data import NativeDataset
from native_sim.control_engine import NativeControlEngine
from native_sim.evaluation import evaluate_record
from baseline_common.config import rng_for
from baseline_common.feedback import FeedbackSession
from baseline_common.controls import codes,physical_units
from run_native_baselines import METHODS,verify_core
from generate_native_dataset import write_json,json_bytes,now

SOURCE=Path(__file__).resolve().parent
METRICS=['evm_percent','payload_nmse','effective_snr_db','bit_errors','bits_tested',
         'feedback_calls','controller_wall_seconds','simulation_feedback_seconds',
         'estimated_control_latency_seconds','normalized_control_mse','delay_mae_ps',
         'attenuation_mae_db','physical_reference_snr_db','pilot_objective','optical_dc_w','apd_dc_a']
GROUPS={m:('privileged_digital_reference' if m=='mrc' else
           'privileged_photonic_teacher' if m=='teacher' else
           'fixed_input_learned' if m=='mlp' else
           'fixed_input_geometric' if m=='ttd_das' else 'adaptive_feedback_controller') for m in METHODS}
_STATE=None


def init_worker(test_root,methods,seeds,budget,checkpoints):
    global _STATE
    compact=CompactDataset(test_root)
    source=NativeDataset(compact.source_root)
    source_rows={r['environment_id']:r for r in source.environments('test')}
    models={}
    if 'mlp' in methods:
        from baseline_mlp.method import load
        for seed,path in zip(seeds,checkpoints):models[seed]=load(path)
    _STATE=dict(compact=compact,source=source,source_rows=source_rows,methods=methods,seeds=seeds,
                budget=budget,models=models)


def evaluate_one(engine,observation,truth,target,method,seed,env_seed,budget,model=None):
    """控制器先返回控制；独立payload的答案只在后面的评分器读取。"""
    cfg=engine.cfg;carrier=int(round(observation['carrier_hz']/1e9))
    result=np.full(len(METRICS),np.nan);values={};control_code=np.full(128,-1,np.int16)
    evaluation_rng=rng_for(env_seed,carrier,seed,510)
    if method=='mrc':
        from baseline_mrc.method import evaluate
        started=time.perf_counter()
        metrics=evaluate(truth['channel'],float(truth['power_dbm']),cfg,evaluation_rng,payload_symbols=1)
        values.update(metrics,payload_nmse=(metrics['evm_percent']/100)**2,
            effective_snr_db=-20*np.log10(max(metrics['evm_percent']/100,1e-15)),
            physical_reference_snr_db=metrics['snr_db'])
    else:
        controller_started=time.perf_counter();simulation_feedback_seconds=0.
        if method=='teacher':control=target.copy();calls=np.nan
        else:
            def measure(u,call):
                return engine.measure(u,observation['pilot_qpsk'],rng_for(env_seed,carrier,seed,520,call))
            session=FeedbackSession(cfg,observation,measure,budget)
            if method=='mlp':control=model.predict(observation,cfg)
            else:
                module=importlib.import_module('baseline_'+method+'.method')
                control=module.optimize(session,rng_for(seed,env_seed,carrier,530))
            calls=session.calls;simulation_feedback_seconds=session.simulation_feedback_seconds
        control=engine.project(control)
        wall=max(0.,time.perf_counter()-controller_started-simulation_feedback_seconds)
        # 离线教师只复用已存结果，读取耗时不能冒充教师离线优化计算时间。
        if method=='teacher':wall=np.nan
        metrics=evaluate_record(engine,control,truth['payload_qpsk'],evaluation_rng)
        additional=engine.evaluate(control);control_code=codes(control,cfg)
        delay,attenuation=physical_units(control,cfg);td,ta=physical_units(target,cfg)
        values.update({k:v for k,v in metrics.items() if np.isscalar(v)},
            effective_snr_db=metrics['snr_db'],feedback_calls=calls,controller_wall_seconds=wall,
            simulation_feedback_seconds=simulation_feedback_seconds,
            estimated_control_latency_seconds=(calls*cfg.measurement_s+cfg.switch_s+wall),
            normalized_control_mse=float(np.mean((control-target)**2)),
            delay_mae_ps=float(np.mean(abs(delay-td))),attenuation_mae_db=float(np.mean(abs(attenuation-ta))),
            pilot_objective=additional['objective'],optical_dc_w=additional['optical_dc_w'],
            apd_dc_a=additional['apd_dc_a'])
    for i,key in enumerate(METRICS):
        if key in values:result[i]=values[key]
    if not np.all(np.isfinite(result[:5])) or result[4]!=62 or not 0<=result[3]<=62:
        raise ValueError('接收指标非有限或误码计数无效。')
    return result,control_code


def evaluate_environment(index,output,fingerprint):
    state=_STATE;d=state['compact'];methods=state['methods'];seeds=state['seeds']
    root=Path(output);path=root/'records'/('environment_%05d.npz'%index);marker=path.with_suffix('.json')
    if path.exists() and marker.exists():
        old=json.loads(marker.read_text())
        if old['protocol_fingerprint']!=fingerprint or sha256(path)!=old['sha256']:
            raise ValueError('既有评测结果版本或SHA不同。')
        return dict(index=index,reused=True)
    if path.exists():raise ValueError('存在无提交标记的评测文件，请先核查：'+str(path))
    row=d.environments[index];native_row=state['source_rows'][row['environment_id']]
    metrics=np.empty((len(methods),len(seeds),17,len(METRICS)),np.float64)
    settings=np.empty((len(methods),len(seeds),17,128),np.int16)
    started=time.perf_counter()
    for ci,carrier in enumerate(d.metadata['carriers_ghz']):
        sample=index*17+ci;record=d.source_record(sample)
        if sha256(record)!=bytes(d.record_sha256[sample]).hex():
            raise ValueError('评分器原始记录与轻量输入的来源不一致。')
        observation=d.observation(sample);truth=state['source'].simulator_truth(native_row,ci)
        engine=NativeControlEngine(d.cfg,truth['branch_band_w'],truth['branch_dc_w'],carrier*1e9,
                                   observation['pilot_qpsk'])
        target=np.asarray(d.Y_code[sample],float)/np.r_[np.full(64,76),np.full(64,24)]
        for mi,method in enumerate(methods):
            for si,seed in enumerate(seeds):
                metrics[mi,si,ci],settings[mi,si,ci]=evaluate_one(engine,observation,truth,target,
                    method,seed,native_row['seed'],state['budget'],state['models'].get(seed))
    temp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    with temp.open('wb') as f:
        np.savez_compressed(f,metrics=metrics,control_code=settings,
            protocol_fingerprint=np.array(fingerprint),environment_id=np.array(row['environment_id']))
    temp.replace(path)
    write_json(marker,dict(protocol_fingerprint=fingerprint,sha256=sha256(path),
        environment_id=row['environment_id'],seconds=time.perf_counter()-started,completed_at_utc=now()))
    return dict(index=index,reused=False,seconds=time.perf_counter()-started)


def protocol(test_root,methods,seeds,budget,checkpoints):
    d=CompactDataset(test_root,verify_hashes=True)
    if d.split!='test':raise ValueError('正式评测只接受dataset_test。')
    source=NativeDataset(d.source_root);verify_core(source)
    if not set(methods).issubset(METHODS) or len(set(methods))!=len(methods):raise ValueError('方法列表错误。')
    if budget<16 or not seeds or len(set(seeds))!=len(seeds):raise ValueError('预算或种子无效。')
    if 'mlp' in methods and len(checkpoints)!=len(seeds):raise ValueError('每个随机种子需要一个检查点。')
    checks=[]
    if 'mlp' in methods:
        from baseline_mlp.method import load
        train=CompactDataset(d.root.parent/'dataset_train')
        expected_train_hash=sha256(train.root/'manifest.json')
        expected_train_ids={e['environment_id'] for e in train.environments}
        tests={e['environment_id'] for e in d.environments}
        for seed,path in zip(seeds,checkpoints):
            model=load(path);meta=model.metadata
            train_ids=set(meta.get('training_environment_ids',[]))
            if not train_ids or train_ids&tests:raise ValueError('训练身份缺失或与测试环境交叠。')
            if model.settings['seed']!=seed:raise ValueError('检查点与预定训练种子不同。')
            for key in ('generation_fingerprint','selection_fingerprint','signal_config'):
                if meta.get(key)!=d.metadata[key]:raise ValueError('MLP训练来源/配置与测试集不一致：'+key)
            if meta.get('training_compact_manifest_sha256')!=expected_train_hash or train_ids!=expected_train_ids:
                raise ValueError('MLP训练清单或环境与统一dataset_train不同。')
            checks.append(dict(seed=seed,path=str(Path(path).resolve()),weights_sha256=sha256(Path(path)/'weights.npz'),
                               metadata_sha256=sha256(Path(path)/'checkpoint.json')))
    files=[Path(__file__),SOURCE/'compact_dataset.py',SOURCE/'run_native_baselines.py',
           SOURCE/'baseline_common/feedback.py',SOURCE/'baseline_common/metrics.py',
           SOURCE/'baseline_common/data.py']+[SOURCE/('baseline_'+m)/'method.py' for m in methods]
    return dict(schema='mwp-compact-baselines-v1',test_dataset=str(d.root),
        test_manifest_sha256=sha256(d.root/'manifest.json'),sample_count=len(d),environment_count=len(d.environments),
        methods=methods,groups={m:GROUPS[m] for m in methods},seeds=seeds,budget=budget,
        initial_public_probes=16,max_additional_feedback=budget-16,checkpoints=checks,
        metric_order=METRICS,carriers_ghz=d.metadata['carriers_ghz'],
        generation_fingerprint=d.metadata['generation_fingerprint'],selection_fingerprint=d.metadata['selection_fingerprint'],
        source_sha256={str(p.relative_to(SOURCE)):sha256(p) for p in files},
        verified_native_core=source.manifest['core_source_sha256'],
        analysis=dict(control_label_error='auxiliary; teacher is not global optimum',
            primary='payload NMSE, pooled BER, EVM; measured feedback counts and wall time',
            aggregation='seed means/std and environment-cluster bootstrap CI; all 17 frequencies retained',
            no_validation_split=True,no_test_tuning=True,
            references='teacher uses private branch cache; MRC separate digital hardware and independent payload'),
        noise_scope='frozen initial probes/antenna noise; optimizer and additional APD draws vary by seed',
        python=platform.python_version(),numpy=np.__version__)


def run(test_root,output,methods,seeds,budget=64,workers=4,checkpoints=()):
    output=Path(output).resolve();p=protocol(test_root,methods,seeds,budget,checkpoints)
    fp=hashlib.sha256(json_bytes(p)).hexdigest()
    output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=p:raise ValueError('既有实验协议不同，拒绝混写。')
    else:write_json(output/'protocol.json',p)
    progress=dict(status='running',protocol_fingerprint=fp,completed_environments=0,
        total_environments=p['environment_count'],started_at_utc=now(),methods=methods,seeds=seeds,workers=workers,pid=os.getpid())
    write_json(output/'progress.json',progress);start=time.perf_counter();errors=[]
    try:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,
            initargs=(str(test_root),methods,seeds,budget,checkpoints)) as executor:
            todo=iter(range(p['environment_count']));pending={}
            def submit():
                i=next(todo,None)
                if i is not None:pending[executor.submit(evaluate_environment,i,str(output),fp)]=i
            for _ in range(workers):submit()
            while pending:
                done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    index=pending.pop(future)
                    try:future.result()
                    except Exception as exc:
                        error=dict(index=index,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                        errors.append(error);write_json(output/('failure_%05d.json'%index),error)
                    else:progress['completed_environments']+=1
                    if not errors:submit()
                progress.update(updated_at_utc=now(),elapsed_seconds=time.perf_counter()-start,failures=len(errors))
                write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if errors:raise RuntimeError('基线失败，已保存原始错误。')
        # 每个完成标记与原子输出校验后才置complete。
        for index in range(p['environment_count']):
            path=output/'records'/('environment_%05d.npz'%index);marker=json.loads(path.with_suffix('.json').read_text())
            if sha256(path)!=marker['sha256'] or marker['protocol_fingerprint']!=fp:raise ValueError('最终结果SHA不符。')
        progress.update(status='complete',finished_at_utc=now(),final_sha_verification=True)
    except BaseException as exc:
        progress.update(status='failed_or_interrupted',error_type=type(exc).__name__,error=str(exc));raise
    finally:write_json(output/'progress.json',progress)
    return progress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--budget',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--checkpoints',nargs='*',type=Path,default=[])
    a=p.parse_args();run(a.test,a.output,a.methods,a.seeds,a.budget,a.workers,a.checkpoints)


if __name__=='__main__':main()
