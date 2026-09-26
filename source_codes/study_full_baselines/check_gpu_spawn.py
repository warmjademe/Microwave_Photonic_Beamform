"""在独立目录验证两个spawn工作进程的实际数据文件，不接触在生成的数据目录。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.expand_training_gpu import one,validate_backend,conflicting_generators


def run(project,output,gates):
    require_host();validate_backend(gates)
    data=project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    check_data(data)
    protocol=json.loads((gates/'protocol.json').read_text())
    # 一条低功率和一条已核查的非线性训练环境，验证spawn并发及实际写盘。
    rows=[protocol['environments'][0],protocol['environments'][3]]
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(data/'public.npz',output/'public.npz')
    sources=source_record(['study_full_baselines/check_gpu_spawn.py',
        'study_full_baselines/expand_training_gpu.py','study_full_baselines/replay_gpu_generation.py',
        'diagnostics/gpu_fft_hybrid.py'])
    write_json(output/'protocol.json',dict(at=now(),environments=rows,workers=2,context='spawn',source_sha256=sources))
    # 隔离检查启动器读回真实进程信息的冲突识别，不能仅凭锁文件判定。
    if conflicting_generators(output):raise ValueError('检查目录意外存在其他生成器。')
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(one,[str(output)]*len(rows),rows))
    checks=[]
    for row,record in zip(rows,results):
        folder=output/row['path'];original=data/row['path']
        for name,digest in record['file_sha256'].items():
            assert sha256(folder/name)==digest
        assert json.loads((folder/'environment.json').read_text())==json.loads((original/'environment.json').read_text())
        with np.load(folder/'data.npz') as a,np.load(original/'data.npz') as b:
            equal={key:bool(np.array_equal(a[key],b[key])) for key in a.files}
        assert all(equal.values()),equal
        checks.append(dict(environment_id=row['environment_id'],exact_array_equality=equal,
            result_record=record))
    verify_sources(sources)
    write_json(output/'summary.json',dict(status='complete',passed=True,checks=checks,at=now()))
    print(json.dumps(dict(status='complete',passed=True,workers=2)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','output','gates']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.project,a.output,a.gates)
