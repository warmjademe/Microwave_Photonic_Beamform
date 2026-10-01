"""在华硕计时冻结的最终控制器；实际推理和搜索，严格重放既有反馈。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import require_host,sha256,verify_sources,write_json,now,atomic_npz,rng_for,NativeConfig
from study_full_baselines.benchmark_online import TraceReplay,resource_snapshot,resource_gate,negative_checks,cpu_ticks
from study_full_baselines.confirmation_batch import guard_load
from study_final864.run import initialize
from study_full_baselines.confirmation_execute import STATE
from study_uniform64.controllers import Feedback64,DIRECT,NEW_METHODS

def read(path):return json.loads(Path(path).read_text())

def sources():
    here=Path(__file__).resolve().parent
    return {p.name:sha256(p) for p in here.iterdir() if p.suffix in ['.py','.md']}

def timing_resource_snapshot(project,preflight):
    """按固定时间窗判断背景负载；保留旧200毫秒读数，不择取较低值。"""
    result=resource_snapshot(project)
    result['cpu_busy_fraction_200ms']=result['cpu_busy_fraction']
    window=.2 if preflight else 5.
    before=cpu_ticks();start=time.monotonic();time.sleep(window);after=cpu_ticks()
    result['cpu_sample_seconds']=time.monotonic()-start
    result['cpu_tick_delta']=after[0]-before[0]
    result['cpu_idle_tick_delta']=after[1]-before[1]
    result['cpu_busy_fraction']=1-(after[1]-before[1])/max(1,after[0]-before[0])
    return result

def run(project,output,preflight,proof):
    require_host()
    if output.exists():raise FileExistsError(output)
    old=project/'dataset_simulation/baseline_results/20260926_final864_selected'
    new=project/'dataset_simulation/baseline_results/20260927_uniform64_all13'
    identity=read(old/'protocol.json');extension=read(new/'protocol.json')
    methods=identity['ordinary_methods']+NEW_METHODS
    assert len(methods)==27 and len(set(methods))==27
    verify_sources(identity['source_sha256']);verify_sources(extension['source_sha256'])
    hashes=sources()
    if not preflight:
        passed=read(proof/'complete.json')
        assert passed['status']=='passed' and passed['preflight'] and passed['source_sha256']==hashes
        assert passed['cases']==81 and passed['methods']==methods
    rows=[identity['rows'][0]] if preflight else [next(r for r in identity['rows'] if r['factors']['power_bin']==b) for b in range(6)]
    carriers=[4,12,20] if preflight else list(range(4,21))
    output.mkdir(parents=True);(output/'records').mkdir();(output/'workers').mkdir()
    for f in hashes:shutil.copyfile(Path(__file__).resolve().parent/f,output/f)
    snapshots=[]
    def check_resources(phase):
        snapshot=timing_resource_snapshot(project,preflight)
        snapshot['phase']=phase;snapshots.append(snapshot)
        write_json(output/'resources.json',snapshots)
        if len(snapshots)==1:write_json(output/'resource_start.json',snapshot)
        resource_gate(snapshot,not preflight)
    try:
        check_resources('start')
        tick=time.perf_counter();initialize(output,identity)
        public=STATE['public'];models=dict(STATE['models'])
        models['initial_select64']=Feedback64(models['ttd_das'],public,True)
        for n in DIRECT:models[n+'_feedback64']=Feedback64(models[n],public)
        torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
        assert list(models)==methods and torch.get_num_threads()==1
        protocol=dict(preflight=preflight,final_fair_timing=not preflight,methods=methods,rows=rows,
            carriers=carriers,warmup=2,repetitions=3,selection='first environment per received-power bin; all carriers',
            source_sha256=hashes,original_protocol_sha256=sha256(old/'protocol.json'),
            extension_protocol_sha256=sha256(new/'protocol.json'),runtime_manifest_sha256=identity['runtime_manifest_sha256'],
            measurement_seconds=NativeConfig().measurement_s,switch_seconds=NativeConfig().switch_s,
            model_loading_seconds=load_seconds,torch=torch.__version__,numpy=np.__version__,
            gpu=torch.cuda.get_device_name(),cpu_threads=torch.get_num_threads(),new_training=False,
            test_quality_or_control_changes=False,cpu_sampling_seconds=.2 if preflight else 5.,
            cpu_busy_limit=.6,at=now())
        write_json(output/'protocol.json',protocol)
        inputs={};input_hashes={}
        for row in rows:
            for fc in carriers:
                merged={}
                for folder,plan in [(old,identity),(new,extension)]:
                    path=folder/'records'/('environment_%05d'%row['index'])/('carrier_%02d.npz'%fc)
                    marker=read(path.parent/'complete.json')
                    expected=next(r for r in marker['carriers'] if r['carrier_ghz']==fc)['sha256']
                    assert sha256(path)==expected
                    input_hashes[str(path)]=expected
                    with np.load(path,allow_pickle=False) as a:arr={k:a[k].copy() for k in a.files}
                    if 'raw' in merged:np.testing.assert_array_equal(merged['raw'],arr['public_X'])
                    merged['raw']=arr['public_X']
                    for i,n in enumerate(plan['methods']):
                        if n not in methods:continue
                        merged[n]=dict(control_code=arr['control_code'][i],feedback_calls=int(arr['metrics'][i,10]))
                        for field in ['trace_control_code','trace_scores','proposal_control_code']:
                            if n+'__'+field in arr:merged[n][field]=arr[n+'__'+field]
                inputs[row['index'],fc]=merged
        write_json(output/'inputs.json',input_hashes)
        checks=negative_checks();records=[]
        for mi,n in enumerate(methods):
            check_resources(n+':before')
            model=models[n]
            for row in rows:
                for fc in carriers:
                    data=inputs[row['index'],fc];reference=data[n];raw=data['raw']
                    def execute():
                        extra=reference['feedback_calls']==64
                        callback=TraceReplay(reference['trace_control_code'][16:],reference['trace_scores'][16:]) if extra else None
                        torch.cuda.synchronize();start=time.perf_counter()
                        with patch('numpy.load',guard_load(np.load)):
                            result=model.decide(raw.copy(),rng_for(0,row['seed'],fc,630),callback)
                        torch.cuda.synchronize();wall=time.perf_counter()-start
                        if callback:callback.finish()
                        np.testing.assert_array_equal(result['control_code'],reference['control_code'])
                        assert result['feedback_calls']==reference['feedback_calls']
                        for field in ['trace_control_code','trace_scores','proposal_control_code']:
                            if field in reference:np.testing.assert_array_equal(result[field],reference[field])
                        sw=result['software_seconds'];cb=result['feedback_simulator_seconds']
                        assert np.isfinite([sw,cb,wall]).all() and min(sw,cb)>=0 and sw+cb<=wall+1e-6
                        return result,[sw,cb,wall]
                    for _ in range(2):execute()
                    measured=[]
                    for _ in range(3):result,times=execute();measured.append(times)
                    name='%s_env%05d_fc%02d.npz'%(n,row['index'],fc)
                    arrays=dict(times=np.asarray(measured),control_code=result['control_code'],public_X=raw,
                        feedback_calls=np.asarray(result['feedback_calls']))
                    for field in ['trace_control_code','trace_scores','proposal_control_code']:
                        if field in result:arrays[field]=result[field]
                    atomic_npz(output/'records'/name,**arrays)
                    records.append(dict(method=n,environment_index=row['index'],carrier_ghz=fc,file=name,
                        sha256=sha256(output/'records'/name)))
            check_resources(n+':after')
            write_json(output/'progress.json',dict(status='running',completed_methods=mi+1,total_methods=len(methods),cases=len(records),pid=os.getpid(),at=now()))
            print(json.dumps(dict(method=n,complete=mi+1,total=len(methods),cases=len(records))),flush=True)
        write_json(output/'records.json',records)
        for p,h in input_hashes.items():assert sha256(p)==h
        for model in STATE['models'].values():model.verify()
        assert sources()==hashes
        verify_sources(identity['source_sha256']);verify_sources(extension['source_sha256'])
        from study_final_timing.audit import audit
        result=audit(output)
        write_json(output/'complete.json',dict(status='passed',at=now(),preflight=preflight,
            final_fair_timing=not preflight,source_sha256=hashes,methods=methods,cases=len(records),
            negative_checks=checks,summary_sha256=sha256(output/'summary.json'),audit=result))
        write_json(output/'progress.json',dict(status='complete',cases=len(records),at=now()))
    except BaseException:
        write_json(output/'failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--final',action='store_true')
    p.add_argument('--proof',type=Path);a=p.parse_args()
    if a.final and a.proof is None:p.error('--final requires --proof')
    run(a.project.resolve(),a.output.resolve(),a.preflight,a.proof.resolve() if a.proof else None)
