"""只分析固定训练输入的生成耗时，不改生成器、数据集或测试输入。"""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys
import time
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from our_method_quality_rank.common import candidate_metrics


def run(project, output):
    require_host()
    data = project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    manifest = check_data(data); public = public_data(data)
    with np.load(data/'public.npz') as f:
        catalog_controls = f['catalog_controls'].copy()
    train = [r for r in manifest['environments'] if r['split']=='train']
    # 只按输入分层固定低/高功率各一个训练环境；不看接收成绩选案例。
    selected = [min(train, key=lambda r: (r['factors']['power_bin'], r['index'])),
                min(train, key=lambda r: (-r['factors']['power_bin'], r['index']))]
    sources = {**manifest['source_sha256'], **source_record([
        'study_full_baselines/profile_generation.py', 'study_full_baselines/common.py'])}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', dict(at=now(), source_sha256=sources,
        data_manifest_sha256=sha256(data/'manifest.json'),
        environment_ids=[r['environment_id'] for r in selected], carriers=[4,12,20],
        frames=[0,1], purpose='hotspot diagnostic only; no final method timing',
        load_average=list(os.getloadavg())))
    records = []; profiler = cProfile.Profile(); profiler.enable()
    for row in selected:
        env = json.loads((data/row['path']/'environment.json').read_text())
        for fc in [4,12,20]:
            for frame in [0,1]:
                tick = time.perf_counter()
                engine, payload = frame_engine(env, fc, public['pilot_qpsk'], frame)
                physical_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                if frame==0:
                    for index,u in enumerate(catalog_controls):
                        engine.measure_detailed(u, rng_for(row['seed'],fc,103,index))
                else:
                    candidate_metrics(engine, catalog_controls, payload,
                                      row['seed'],fc,104,frame,draws=8)
                records.append(dict(environment_id=row['environment_id'],
                    power_bin=row['factors']['power_bin'],carrier_ghz=fc,frame=frame,
                    physical_seconds=physical_seconds,
                    measurement_seconds=time.perf_counter()-tick,
                    nonlinear_routes=engine.simulation_details['nonlinear_routes']))
                write_json(output/'progress.json', dict(status='running',
                    completed=len(records),total=12,pid=os.getpid(),at=now()))
    profiler.disable(); profiler.dump_stats(str(output/'generation.prof'))
    statistics = pstats.Stats(profiler)
    functions = [dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],
        total_calls=v[1],own_seconds=v[2],cumulative_seconds=v[3])
        for k,v in statistics.stats.items()]
    verify_sources(sources)
    write_json(output/'summary.json', dict(status='complete', records=records,
        top_own_time=sorted(functions,key=lambda r:-r['own_seconds'])[:30],
        top_cumulative_time=sorted(functions,key=lambda r:-r['cumulative_seconds'])[:30],
        profiler_sha256=sha256(output/'generation.prof'),
        load_average_end=list(os.getloadavg()),at=now()))
    write_json(output/'progress.json',dict(status='complete',completed=12,total=12,at=now()))
    print(json.dumps(dict(status='complete',cases=12)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        run(a.project,a.output)
    except BaseException:
        if a.output.exists():
            write_json(a.output/('failure_%d.json'%time.time()),
                       dict(status='failed',traceback=traceback.format_exc(),at=now()))
        raise
