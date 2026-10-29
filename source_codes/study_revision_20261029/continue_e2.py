"""E2持久队列：训练产物提交后逐种子评价；失败停止，不选择最好种子。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def run(project, workers):
    root = project/'dataset_simulation/revision_20261029'
    with (root/'e2_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in (1, 2):
            model = root/'e2_training'/('seed_%d'%seed)
            while not (model/'complete.json').exists():
                failed = list((root/'e2_training').glob('failure_*.json'))
                if failed:
                    raise RuntimeError('Training failed; inspect '+str(failed[-1]))
                pid = json.loads((root/'e2_launch.json').read_text())['pid']
                os.kill(pid, 0)
                state = dict(status='waiting_for_training', seed=seed, at=time.time())
                (root/'e2_sequence_progress.json').write_text(json.dumps(state,indent=2))
                time.sleep(15)
            output = root/('e2_seed_%d_test'%seed)
            command = [str(project/'.venv_dl/bin/python'), str(Path(__file__).with_name('evaluate_e2.py')),
                       '--project', str(project), '--output', str(output), '--seed', str(seed),
                       '--workers', str(workers)]
            (root/'e2_sequence_progress.json').write_text(json.dumps(dict(status='evaluating',seed=seed,at=time.time()),indent=2))
            subprocess.run(command, cwd=project, check=True)
        (root/'e2_sequence_progress.json').write_text(json.dumps(dict(status='evaluations_complete',seeds=[1,2],at=time.time()),indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args(); run(a.project.resolve(), a.workers)
