"""使用GPU FFT重放六个训练环境的完整生成过程，与冻结CPU数据逐字段比较。"""
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
from compact_dataset import pack_observation
from baseline_common.config import qpsk
from our_method_quality_rank.common import candidate_metrics
from gpu_fft_hybrid import approximate_cache_cuda


def generate_arrays(environment,public,progress_callback=None):
    """保持原one_environment的计算顺序和随机流，仅传入另一FFT后端。"""
    cfg=NativeConfig();seed=environment['seed'];pilots=public['pilot_qpsk']
    controls=public['catalog_controls'];xs=[];singles=[];robusts=[];choices=[];counts=[]
    def engine_for(fc,frame):
        payload=qpsk(rng_for(seed,fc,101,frame),(31,))
        band,dc,detail=approximate_cache_cuda(environment,fc*1e9,pilots,payload,
            rng_for(seed,fc,102,frame),cfg,return_details=True)
        return NativeControlEngine(cfg,band,dc,fc*1e9,pilots),payload,detail
    for fc in range(4,21):
        engine,_,detail=engine_for(fc,0);frame_counts=[detail['nonlinear_routes']]
        measured=[engine.measure_detailed(u,rng_for(seed,fc,103,index)) for index,u in enumerate(controls)]
        xs.append(pack_observation(dict(combined_iq_a=np.stack([v['symbols'] for v in measured[:16]]),
            quality=np.asarray([v['score'] for v in measured[:16]]),
            noise_symbol_var_a2=np.stack([v['noise_symbol_var'] for v in measured[:16]]),
            probe_apd_dc_a=np.asarray([v['apd_dc_a'] for v in measured[:16]])),fc))
        choices.append(int(np.argmax([v['score'] for v in measured])))
        labels=[]
        for frame in range(1,5):
            engine,payload,detail=engine_for(fc,frame)
            frame_counts.append(detail['nonlinear_routes'])
            labels.append(candidate_metrics(engine,controls,payload,seed,fc,104,frame,draws=8)['nmse'])
        singles.append(labels[0]);robusts.append(np.mean(labels,axis=0));counts.append(frame_counts)
        if progress_callback:progress_callback(fc)
    return dict(X=np.asarray(xs,np.float32),single_nmse=np.asarray(singles,np.float32),
        robust_nmse=np.asarray(robusts,np.float32),measured64_choice=np.asarray(choices,np.int16),
        nonlinear_route_counts=np.asarray(counts,np.int16))


def relative(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.finfo(float).tiny))


def run(project,output,preflight):
    require_host()
    if not torch.cuda.is_available():raise RuntimeError('要求华硕CUDA。')
    pre=json.loads((preflight/'summary.json').read_text())
    if not pre['passed_declared_gates'] or len(pre['records'])!=12:
        raise ValueError('12案例GPU逐段核查未通过。')
    verify_sources(pre['source_sha256'])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    data=project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    manifest=check_data(data)
    train=[r for r in manifest['environments'] if r['split']=='train']
    rows=[min([r for r in train if r['factors']['power_bin']==v],key=lambda r:r['index'])
          for v in sorted({r['factors']['power_bin'] for r in train})]
    source_records={r['environment_id']:r for r in json.loads((data/'records.json').read_text())}
    with np.load(data/'public.npz') as f:public={k:f[k].copy() for k in f.files}
    sources={**manifest['source_sha256'],**source_record([
        'study_full_baselines/replay_gpu_generation.py','diagnostics/gpu_fft_hybrid.py',
        'study_full_baselines/GPU_FFT_PROTOCOL.md','study_full_baselines/common.py'])}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',dict(at=now(),source_sha256=sources,
        data_manifest_sha256=sha256(data/'manifest.json'),preflight_summary_sha256=sha256(preflight/'summary.json'),
        environments=rows,carriers=list(range(4,21)),frames=list(range(5)),relative_gate=1e-6,
        device=torch.cuda.get_device_name(),torch=torch.__version__,load_average_start=list(os.getloadavg()),
        formal_backend_changed=False))
    records=[]
    for index,row in enumerate(rows):
        folder=data/row['path'];original=source_records[row['environment_id']]
        for name,digest in original['file_sha256'].items():
            if sha256(folder/name)!=digest:raise ValueError('原CPU样本哈希改变。')
        env=json.loads((folder/'environment.json').read_text());started=time.perf_counter()
        def progress(fc):
            write_json(output/'progress.json',dict(status='running',completed_environments=index,
                total_environments=len(rows),current_environment=row['environment_id'],
                completed_carriers=fc-3,pid=os.getpid(),at=now()))
        got=generate_arrays(env,public,progress);duration=time.perf_counter()-started
        with np.load(folder/'data.npz') as f:reference={k:f[k].copy() for k in f.files}
        assert set(got)==set(reference)
        for key in got:
            assert got[key].shape==reference[key].shape and np.isfinite(got[key]).all(),key
        groups={name:relative(reference['X'][:,start:end],got['X'][:,start:end]) for name,start,end in
            [('iq',0,1984),('pilot_score',1985,2001),('noise_variance',2001,2497),('apd_dc',2497,2513)]}
        for name in ['single_nmse','robust_nmse']:groups[name]=relative(reference[name],got[name])
        equal={key:bool(np.array_equal(got[key],reference[key])) for key in got}
        carrier_equal=bool(np.array_equal(got['X'][:,1984],reference['X'][:,1984]))
        path=output/('environment_%05d.npz'%row['index']);atomic_npz(path,**got)
        rec=dict(environment_id=row['environment_id'],factors=row['factors'],
            gpu_generation_seconds=duration,original_recorded_seconds=original['seconds'],
            original_time_comparable=False,relative_errors=groups,exact_array_equality=equal,
            carrier_equal=carrier_equal,result_file=path.name,result_sha256=sha256(path),
            source_files_sha256=original['file_sha256'],
            passed=max(groups.values())<=1e-6 and carrier_equal and
                equal['measured64_choice'] and equal['nonlinear_route_counts'])
        records.append(rec);write_json(output/'records.json',records)
        print(json.dumps(dict(environment=row['environment_id'],passed=rec['passed'],seconds=duration)),flush=True)
    verify_sources(sources);check_data(data)
    write_json(output/'summary.json',dict(status='complete',passed_declared_gates=all(r['passed'] for r in records),
        all_arrays_bitwise_equal=all(all(r['exact_array_equality'].values()) for r in records),
        records=records,at=now(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        formal_backend_changed=False,load_average_end=list(os.getloadavg())))
    write_json(output/'progress.json',dict(status='complete',completed_environments=len(rows),
        total_environments=len(rows),at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','output','preflight']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:run(a.project,a.output,a.preflight)
    except BaseException:
        if a.output.exists():write_json(a.output/('failure_%d.json'%time.time()),
            dict(traceback=traceback.format_exc(),at=now()))
        raise
