"""分开记录 A/F 的无探针总时间与分段诊断，逐次核对冻结控制。"""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import require_host,sha256,write_json,now,rng_for,atomic_npz,verify_sources
from study_full_baselines.benchmark_online import TraceReplay,negative_checks,resource_gate
from study_final_timing.run import timing_resource_snapshot
from study_full_baselines.confirmation_batch import guard_load
import study_full_baselines.online_controller as online
from study_rq2_af_20261029.run import initialize,STATE,read
from study_uniform64.controllers import Feedback64

METHODS=['complex_response_cnn','cnn_warm64','covariance_warm64','initial_select64']


def identity():
    here=Path(__file__).resolve().parent
    return {n:sha256(here/n) for n in ['benchmark_af.py','P_TIMING_PROTOCOL.md']}


def measure(model,raw,reference,row,fc,instrumented):
    """实际重新推理，只用已记录且完全相同的查询替代物理测量。"""
    callback=TraceReplay(reference['trace_control_code'][16:],reference['trace_scores'][16:]) if 'trace_scores' in reference else None
    spans={'response_estimation':0.,'gpu_network_forward':0.,'discrete_solver':0.}
    base=model.base if isinstance(model,Feedback64) else model
    def wrap(name,fn,gpu=False):
        def call(*args,**kwargs):
            if gpu:torch.cuda.synchronize()
            start=time.perf_counter()
            try:return fn(*args,**kwargs)
            finally:
                if gpu:torch.cuda.synchronize()
                spans[name]+=time.perf_counter()-start
        return call
    with ExitStack() as stack:
        stack.enter_context(patch('numpy.load',guard_load(np.load)))
        if instrumented:
            stack.enter_context(patch.object(base,'estimate',wrap('response_estimation',base.estimate)))
            stack.enter_context(patch.object(online,'decode',wrap('discrete_solver',online.decode)))
            if base.kind=='response':
                stack.enter_context(patch.object(base.model,'forward',wrap('gpu_network_forward',base.model.forward,True)))
        torch.cuda.synchronize();start=time.perf_counter()
        result=model.decide(raw.copy(),rng_for(0,row['seed'],fc,630),callback)
        torch.cuda.synchronize();wall=time.perf_counter()-start
    if callback:callback.finish()
    np.testing.assert_array_equal(result['control_code'],reference['control_code'])
    assert result['feedback_calls']==reference['feedback_calls']
    for name in ['trace_control_code','trace_scores','proposal_control_code']:
        if name in reference:np.testing.assert_array_equal(result[name],reference[name])
    total=float(result['software_seconds']);cb=float(result['feedback_simulator_seconds'])
    assert min(total,cb)>=0 and total+cb<=wall+1e-6
    if instrumented:
        spans['other_control_logic']=total-spans['response_estimation']-spans['discrete_solver']
        spans['response_other']=spans['response_estimation']-spans['gpu_network_forward']
        assert min(spans.values())>=-1e-8,spans
        spans={k:max(0.,v) for k,v in spans.items()}
    return result,dict(software=total,callback=cb,wall=wall,**(spans if instrumented else {}))


def run(project,output,preflight,proof):
    require_host()
    if output.exists():raise FileExistsError(output)
    signature=identity()
    dependencies={n:sha256(project/'source_codes'/n) for n in
        ['study_full_baselines/benchmark_online.py','study_final_timing/run.py']}
    if not preflight:
        passed=read(proof/'complete.json')
        assert passed['status']=='passed' and passed['preflight'] and passed['source_sha256']==signature
        assert passed['timing_dependencies_sha256']==dependencies
    anchor=project/'dataset_simulation/revision_rq2_af_20261029/full'
    p=read(anchor/'protocol.json'); verify_sources(p['source_sha256'])
    rows=[p['rows'][0]] if preflight else [next(r for r in p['rows'] if r['factors']['power_bin']==i) for i in range(6)]
    carriers=[4,12,20] if preflight else list(range(4,21))
    output.mkdir(parents=True);(output/'workers').mkdir();(output/'records').mkdir()
    for name in signature:shutil.copyfile(Path(__file__).parent/name,output/name)
    snapshots=[];input_hashes={};records=[]
    def resource_check(phase):
        s=timing_resource_snapshot(project,preflight);s['phase']=phase;snapshots.append(s)
        write_json(output/'resources.json',snapshots);resource_gate(s,not preflight)
    try:
        resource_check('start')
        start=time.perf_counter();initialize(output,p);torch.cuda.synchronize();load_time=time.perf_counter()-start
        assert torch.get_num_threads()==1
        protocol=dict(source_sha256=signature,timing_dependencies_sha256=dependencies,anchor_protocol_sha256=sha256(anchor/'protocol.json'),
            rows=rows,carriers=carriers,methods=METHODS,preflight=preflight,formal_timing=not preflight,
            warmup=2,repetitions=3,model_loading_seconds=load_time,at=now(),cpu_threads=1,
            torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),
            callback='Exact frozen-query replay; no new physical measurement',
            assumptions=dict(pilot_acquisition_s=1.24e-6,switch_s=1e-6,final_apply_switches=1),
            joint_correction=False)
        write_json(output/'protocol.json',protocol)
        checks=negative_checks()
        for name in METHODS:
            resource_check(name+':before');model=STATE['models'][name]
            for row in rows:
                for fc in carriers:
                    folder=anchor/'records'/f"environment_{row['index']:05d}"
                    marker=read(folder/'complete.json')
                    entry=next(x for x in marker['carriers'] if x['carrier_ghz']==fc)
                    path=anchor/entry['path'];digest=sha256(path);assert digest==entry['sha256']
                    input_hashes[str(path)]=digest
                    with np.load(path,allow_pickle=False) as f:
                        raw=f['public_X'].copy();i=p['methods'].index(name)
                        ref=dict(control_code=f['control_code'][i].copy(),feedback_calls=int(f['metrics'][i,10]))
                        for key in ['trace_control_code','trace_scores','proposal_control_code']:
                            if name+'__'+key in f:ref[key]=f[name+'__'+key].copy()
                    for _ in range(2):measure(model,raw,ref,row,fc,False)
                    plain=[];segmented=[]
                    for _ in range(3):result,t=measure(model,raw,ref,row,fc,False);plain.append(t)
                    for _ in range(3):result,t=measure(model,raw,ref,row,fc,True);segmented.append(t)
                    filename=f"{name}_env{row['index']:05d}_fc{fc:02d}.npz"
                    arrays=dict(public_X=raw,control_code=result['control_code'],
                        feedback_calls=np.asarray(result['feedback_calls']),
                        plain_fields=np.asarray(list(plain[0])),plain=np.asarray([list(t.values()) for t in plain]),
                        segmented_fields=np.asarray(list(segmented[0])),segmented=np.asarray([list(t.values()) for t in segmented]))
                    for key in ['trace_control_code','trace_scores','proposal_control_code']:
                        if key in result:arrays[key]=result[key]
                    atomic_npz(output/'records'/filename,**arrays)
                    records.append(dict(method=name,index=row['index'],carrier=fc,file=filename,sha256=sha256(output/'records'/filename)))
            resource_check(name+':after')
            write_json(output/'progress.json',dict(status='running',records=len(records),pid=os.getpid(),at=now()))
        write_json(output/'inputs.json',input_hashes);write_json(output/'records.json',records)
        summaries=[]
        for name in METHODS:
            grouped={}
            for item in records:
                if item['method']!=name:continue
                path=output/'records'/item['file'];assert sha256(path)==item['sha256']
                with np.load(path,allow_pickle=False) as f:
                    for kind in ['plain','segmented']:
                        for j,key in enumerate(f[kind+'_fields']):grouped.setdefault(kind+'__'+str(key),[]).extend(f[kind][:,j].tolist())
            def stats(vals):
                x=np.asarray(vals)*1000
                return dict(mean_ms=float(x.mean()),median_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),observations=len(x))
            budget=16 if name=='complex_response_cnn' else 64
            summaries.append(dict(method=name,components={k:stats(v) for k,v in grouped.items()},
                nominal_hardware_ms=(budget*2.24+1)*.001,budget=budget))
        write_json(output/'summary.json',dict(rows=summaries,instrumented_and_plain_separate=True,
            acquisition_and_switch='assumption, not measured hardware latency',preflight=preflight))
        for path,h in input_hashes.items():assert sha256(path)==h
        for model in STATE['models'].values():
            (model.base if isinstance(model,Feedback64) else model).verify()
        verify_sources(p['source_sha256']);assert signature==identity()
        for name,h in dependencies.items():assert sha256(project/'source_codes'/name)==h
        assert len(records)==len(METHODS)*len(rows)*len(carriers)
        write_json(output/'complete.json',dict(status='passed',preflight=preflight,formal_timing=not preflight,
            source_sha256=signature,timing_dependencies_sha256=dependencies,at=now(),cases=len(records),joint_correction=False,negative_checks=checks,
            file_sha256={n:sha256(output/n) for n in ['summary.json','records.json','inputs.json','resources.json','protocol.json']}))
    except BaseException:
        write_json(output/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--proof',type=Path);args=parser.parse_args()
    if not args.preflight and not args.proof:parser.error('Formal timing requires a passed preflight')
    run(args.project.resolve(),args.output.resolve(),args.preflight,args.proof)
