"""已有完整接收评测释放工作进程后，接续组件与大规模接收比较。"""
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


def run(a):
    require_host(); a.queue.mkdir(parents=True, exist_ok=False)
    frozen = source_record(['study_full_baselines/variants_when_ready.py',
        'study_full_baselines/evaluate_response_variants.py',
        'study_full_baselines/evaluate_learned.py', 'study_full_baselines/common.py',
        'our_method_two_stage/control.py', 'our_method_two_stage/decode_multistart.py',
        'our_method_response_control/physics.py'])
    write_json(a.queue/'protocol.json', dict(source_sha256=frozen,
        evaluation_queue_pid=a.evaluation_queue_pid, scale_queue_pid=a.scale_queue_pid,
        suites=['components', 'scaling_1728', 'scaling_3456'], workers=4, at=now()))
    base = a.project/'dataset_simulation'; study = base/'baseline_results/20260925_full_baselines'
    for suite, prerequisite, prerequisite_pid in [
        ('components', study/'learned_evaluation', a.evaluation_queue_pid),
        ('scaling_1728', study/'scale_1728', a.scale_queue_pid),
        ('scaling_3456', study/'scale_3456', a.scale_queue_pid)]:
        while True:
            path = prerequisite/'progress.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get('status')=='complete':
                break
            if list(prerequisite.rglob('failure*.json')) or not alive(prerequisite_pid):
                raise RuntimeError('前置任务尚未完成但出现异常或队列退出。')
            write_json(a.queue/'progress.json', dict(status='waiting', suite=suite,
                prerequisite=str(prerequisite), pid=os.getpid(), at=now()))
            time.sleep(20)
        verify_sources(frozen)
        output = study/('evaluation_'+suite)
        cmd = [sys.executable, '-u', 'study_full_baselines/evaluate_response_variants.py',
            '--project', str(a.project), '--data', str(base/'outputs/quality_rank_hybrid_20260925'),
            '--study', str(study), '--output', str(output), '--suite', suite, '--workers', '4']
        write_json(a.queue/'progress.json', dict(status='evaluating', suite=suite,
            command=cmd, pid=os.getpid(), at=now()))
        subprocess.run(cmd, cwd=SOURCE, check=True)
    verify_sources(frozen)
    write_json(a.queue/'progress.json', dict(status='complete', at=now(),
        scope='exploratory component and scale comparisons only; independent confirmation and website still required'))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'queue']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--evaluation-queue-pid', type=int, required=True)
    p.add_argument('--scale-queue-pid', type=int, required=True); a = p.parse_args()
    try:
        run(a)
    except BaseException:
        a.queue.mkdir(parents=True, exist_ok=True)
        write_json(a.queue/'failure.json', dict(traceback=traceback.format_exc(), at=now()))
        raise
