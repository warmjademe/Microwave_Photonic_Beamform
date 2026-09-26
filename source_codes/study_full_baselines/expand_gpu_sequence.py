"""使用已核查的GPU FFT后端依次完成同一1728/3456嵌套训练计划。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.expand_training_gpu import validate_backend,EXTRA_SOURCES


def run(project,queue,gates,workers):
    require_host();validation=validate_backend(gates)
    queue.mkdir(parents=True,exist_ok=False)
    frozen=source_record(EXTRA_SOURCES+['study_full_baselines/expand_gpu_sequence.py'])
    write_json(queue/'protocol.json',dict(at=now(),source_sha256=frozen,validation=validation,
        targets=[1728,3456],seeds=[2026092507,2026092508],workers=workers,
        scope='same sampling and numerical model; validated CUDA FFT execution backend'))
    base=project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    for target,seed in [(1728,2026092507),(3456,2026092508)]:
        verify_sources(frozen)
        output=project/('dataset_simulation/outputs/scaling_train_%d_20260925'%target)
        command=[sys.executable,'-u','study_full_baselines/expand_training_gpu.py',
            '--project',str(project),'--base',str(base),'--output',str(output),'--target',str(target),
            '--seed',str(seed),'--workers',str(workers),'--gates',str(gates)]
        write_json(queue/'progress.json',dict(status='expanding',target=target,pid=os.getpid(),
            command=command,backend='cuda_fft_cpu_rk4_v1',at=now()))
        with (queue/('target_%d.log'%target)).open('x') as stream:
            subprocess.run(command,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True)
        base=output
    verify_sources(frozen)
    write_json(queue/'progress.json',dict(status='complete',at=now(),
        scope='expanded training data only; method comparison and fresh confirmation remain required'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','queue','gates']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    try:run(a.project,a.queue,a.gates,a.workers)
    except BaseException:
        if a.queue.exists():write_json(a.queue/'failure.json',dict(traceback=traceback.format_exc(),at=now()))
        raise
