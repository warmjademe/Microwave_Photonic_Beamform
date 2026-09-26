"""固定训练案例上核查GPU FFT后端；独立保存数值差异与诊断耗时。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from baseline_common.config import qpsk
from compact_dataset import pack_observation
from our_method_quality_rank.common import candidate_metrics
from hybrid_centered import approximate_cache
from gpu_fft_hybrid import approximate_cache_cuda


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.finfo(float).tiny))


def observe(engine,controls,seed,fc):
    measured=[engine.measure_detailed(u,rng_for(seed,fc,103,index)) for index,u in enumerate(controls)]
    x=pack_observation(dict(combined_iq_a=np.stack([m['symbols'] for m in measured[:16]]),
        quality=np.asarray([m['score'] for m in measured[:16]]),
        noise_symbol_var_a2=np.stack([m['noise_symbol_var'] for m in measured[:16]]),
        probe_apd_dc_a=np.asarray([m['apd_dc_a'] for m in measured[:16]])),fc)
    return x,int(np.argmax([m['score'] for m in measured]))


def run(project,output):
    require_host()
    if not torch.cuda.is_available():raise RuntimeError('仅允许华硕CUDA进行加速诊断。')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    data=project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    manifest=check_data(data);public=public_data(data)
    with np.load(data/'public.npz') as f:controls=f['catalog_controls'].copy()
    train=[r for r in manifest['environments'] if r['split']=='train']
    rows=[min(train,key=lambda r:(r['factors']['power_bin'],r['index'])),
          min(train,key=lambda r:(-r['factors']['power_bin'],r['index']))]
    sources={**manifest['source_sha256'],**source_record([
        'study_full_baselines/check_gpu_fft.py','diagnostics/gpu_fft_hybrid.py',
        'study_full_baselines/common.py'])}
    gates=dict(band_relative=2e-8,dc_relative=1e-10,observation_relative=1e-6,
               nmse_relative=1e-6,identical_measured_choice=True,identical_bit_errors=True)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',dict(at=now(),source_sha256=sources,gates=gates,
        data_manifest_sha256=sha256(data/'manifest.json'),
        environment_ids=[r['environment_id'] for r in rows],carriers=[4,12,20],frames=[0,1],
        activation='diagnostic only; not used by current generation or evaluation',
        device=torch.cuda.get_device_name(),torch=torch.__version__,precision='float64/complex128 FFT',
        load_average_start=list(os.getloadavg()),final_fair_timing=False))
    # 两种FFT先做计划初始化，正式比较中仍计CPU/GPU数据拷贝。
    n=NativeConfig().sample_count*4
    z=torch.zeros((8,n//2+1),dtype=torch.complex128,device='cuda')
    y=torch.fft.irfft(z,n=n);torch.fft.rfft(y);torch.cuda.synchronize()
    del z,y
    records=[]
    for row in rows:
        env=json.loads((data/row['path']/'environment.json').read_text())
        for fc in [4,12,20]:
            for frame in [0,1]:
                payload=qpsk(rng_for(row['seed'],fc,101,frame),(31,));cfg=NativeConfig()
                outputs={};seconds={}
                order=['cpu','gpu'] if len(records)%2==0 else ['gpu','cpu']
                for name in order:
                    fn=approximate_cache if name=='cpu' else approximate_cache_cuda
                    tick=time.perf_counter()
                    outputs[name]=fn(env,fc*1e9,public['pilot_qpsk'],payload,
                                    rng_for(row['seed'],fc,102,frame),cfg,return_details=True)
                    seconds[name]=time.perf_counter()-tick
                a,b=outputs['cpu'],outputs['gpu']
                assert a[2]==b[2],'GPU后端改变了非线性路由选择'
                engines=[NativeControlEngine(cfg,v[0],v[1],fc*1e9,public['pilot_qpsk']) for v in [a,b]]
                xa,ia=observe(engines[0],controls,row['seed'],fc)
                xb,ib=observe(engines[1],controls,row['seed'],fc)
                qa,qb=[candidate_metrics(e,controls,payload,row['seed'],fc,104,frame,draws=8) for e in engines]
                observation_groups={name:relative(xa[start:end],xb[start:end]) for name,start,end in
                    [('iq',0,1984),('pilot_score',1985,2001),('noise_variance',2001,2497),('apd_dc',2497,2513)]}
                assert xa[1984]==xb[1984], 'GPU后端改变了载频'
                record=dict(environment_id=row['environment_id'],power_bin=row['factors']['power_bin'],
                    carrier_ghz=fc,frame=frame,nonlinear_routes=a[2]['nonlinear_routes'],
                    cpu_seconds=seconds['cpu'],gpu_fft_seconds=seconds['gpu'],
                    band_relative=relative(a[0],b[0]),dc_relative=relative(a[1],b[1]),
                    observation_relative=max(observation_groups.values()),observation_groups=observation_groups,
                    nmse_relative=relative(qa['nmse'],qb['nmse']),
                    identical_measured_choice=ia==ib,
                    identical_bit_errors=bool(np.array_equal(qa['bit_errors'],qb['bit_errors'])),
                    packed_float32_identical=bool(np.array_equal(xa.astype(np.float32),xb.astype(np.float32))))
                record['passed']=all(record[k]<=gates[k] for k in ['band_relative','dc_relative','observation_relative','nmse_relative']) and record['identical_measured_choice'] and record['identical_bit_errors']
                records.append(record)
                write_json(output/'progress.json',dict(status='running',completed=len(records),total=12,pid=os.getpid(),at=now()))
                write_json(output/'records.json',records)
    verify_sources(sources)
    write_json(output/'summary.json',dict(status='complete',passed_declared_gates=all(r['passed'] for r in records),
        records=records,source_sha256=sources,load_average_end=list(os.getloadavg()),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        current_formal_backend_changed=False,full_17_carrier_validation_pending=True,at=now()))
    write_json(output/'progress.json',dict(status='complete',completed=12,total=12,at=now()))
    print(json.dumps(dict(status='complete',passed=all(r['passed'] for r in records))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:run(a.project,a.output)
    except BaseException:
        if a.output.exists():
            write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
