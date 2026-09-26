"""扩展数据完成且七个基线释放GPU后，执行两个大训练规模。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.train_when_ready import alive


def state(folder):
    p = folder/'progress.json'
    return json.loads(p.read_text()) if p.exists() else {}


def run(a):
    require_host(); a.queue.mkdir(parents=True, exist_ok=False)
    frozen = source_record(['study_full_baselines/scale_large_when_ready.py',
        'study_full_baselines/scale_large.py', 'study_full_baselines/SCALE_LARGE_PROTOCOL.md',
        'study_full_baselines/common.py'])
    write_json(a.queue/'protocol.json', dict(source_sha256=frozen,
        expansion_pid=a.expansion_pid, training_pid=a.training_pid,
        counts=[1728, 3456], schedules=['fixed_epochs', 'equal_updates'], at=now()))
    base = a.project/'dataset_simulation'; old = base/'outputs/quality_rank_hybrid_20260925'
    training = base/'baseline_results/20260925_full_baselines/learned'
    expansion = base/'ops/full_baselines_20260925/expansion_queue'
    for count in [1728, 3456]:
        data = base/('outputs/scaling_train_%d_20260925'%count)
        while True:
            data_ready = state(data).get('status')=='complete'
            training_ready = state(training).get('status')=='complete'
            if data_ready and training_ready:
                break
            if list(expansion.glob('failure*.json')) or list(training.rglob('failure*.json')):
                raise RuntimeError('前置扩展或基线训练异常。')
            if not data_ready and not alive(a.expansion_pid):
                raise RuntimeError('数据尚未完成，但扩展队列已经退出。')
            if not training_ready and not alive(a.training_pid):
                raise RuntimeError('基线训练尚未完成，但训练队列已经退出。')
            write_json(a.queue/'progress.json', dict(status='waiting', count=count,
                data_ready=data_ready, baseline_training_ready=training_ready, pid=os.getpid(), at=now()))
            time.sleep(20)
        verify_sources(frozen)
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0])
            if free >= 12000:
                break
            write_json(a.queue/'progress.json', dict(status='waiting_for_GPU_memory',
                count=count, free_mib=free, pid=os.getpid(), at=now()))
            time.sleep(20)
        output = base/('baseline_results/20260925_full_baselines/scale_%d'%count)
        cmd = [sys.executable, '-u', 'study_full_baselines/scale_large.py', '--data', str(data),
            '--exploratory-data', str(old), '--output', str(output)]
        write_json(a.queue/'progress.json', dict(status='running', count=count,
            command=cmd, pid=os.getpid(), at=now()))
        subprocess.run(cmd, cwd=SOURCE, check=True)
    write_json(a.queue/'progress.json', dict(status='complete', at=now(),
        scope='response models and public predictions only; reception evaluation and final confirmation still required'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'queue']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expansion-pid', type=int, required=True)
    p.add_argument('--training-pid', type=int, required=True); a = p.parse_args()
    try:
        run(a)
    except BaseException:
        a.queue.mkdir(parents=True, exist_ok=True)
        write_json(a.queue/'failure.json', dict(traceback=traceback.format_exc(), at=now()))
        raise
