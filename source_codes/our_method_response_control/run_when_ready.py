"""冻结补充方法，等待已有质量排序实验完成，再串行执行完整控制实验。"""
import argparse
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from compact_dataset import sha256
from generate_native_dataset import write_json, now


def run(project, output):
    if platform.node() != 'qyb-HuaShuo':
        raise RuntimeError('仅华硕执行。')
    output.mkdir(parents=True, exist_ok=False)
    ds = project/'dataset_simulation'
    data = ds/'outputs/quality_rank_hybrid_20260925'
    targets = ds/'diagnostics/20260925_response_control_targets'
    first = ds/'baseline_results/20260925_quality_rank_hybrid'
    results = ds/'baseline_results/20260925_response_control'
    files = [p for p in (SOURCE/'our_method_response_control').glob('*') if p.suffix in ['.py', '.md']]
    frozen = {str(p.relative_to(SOURCE)): sha256(p) for p in files}
    write_json(output/'frozen_plan.json', dict(at=now(), source_sha256=frozen,
        expected_prior=str(first), output=str(results), seed=0, epochs=40,
        data=str(data), targets=str(targets), sequential=True, stop_on_failure=True,
        maximum_wait_hours=12, new_test_results_seen=False))
    for p in files:
        destination = output/'source_snapshot'/p.name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, destination)
    def verify():
        for name, digest in frozen.items():
            if sha256(SOURCE/name) != digest:
                raise ValueError('排队后方法源码改变：'+name)
    failures = [data/'failure.json', first/'failure.json', first/'evaluation_failure.json']
    started = time.monotonic()
    while True:
        verify()
        for path in failures:
            if path.exists():
                raise RuntimeError('前置流水线失败，保留现场：'+str(path))
        progress, audit = first/'evaluation/progress.json', first/'evaluation/audit.json'
        ready = (progress.exists() and audit.exists()
            and json.loads(progress.read_text())['status'] == 'complete'
            and json.loads(audit.read_text())['status'] == 'complete')
        phase = 'waiting_for_quality_rank'
        if ready:
            compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip()
            if not compute:
                break
            phase = 'waiting_for_gpu_release'
        write_json(output/'progress.json', dict(status=phase, at=now(), seconds=time.monotonic()-started))
        if time.monotonic()-started > 12*3600:
            raise TimeoutError('等待12小时仍未满足前置条件；没有启动补充训练。')
        time.sleep(45)
    if results.exists():
        raise FileExistsError('补充实验目录已经存在，不覆盖或重复运行。')
    stages = [
        ('training', 'train.py', ['--data', str(data), '--targets', str(targets), '--output', str(results)]),
        ('evaluation', 'evaluate.py', ['--data', str(data), '--targets', str(targets), '--run-root', str(results), '--workers', '6']),
        ('audit', 'audit.py', ['--data', str(data), '--targets', str(targets), '--run-root', str(results), '--workers', '6'])]
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = '1'
    for stage, script, args in stages:
        verify()
        write_json(output/'progress.json', dict(status=stage, at=now()))
        with (output/(stage+'.log')).open('x') as log:
            subprocess.run([sys.executable, str(SOURCE/'our_method_response_control'/script), *args],
                cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify()
    final = results/'evaluation/control_replay_audit.json'
    if json.loads(final.read_text())['status'] != 'complete':
        raise ValueError('完整控制回放未完成。')
    write_json(output/'progress.json', dict(status='complete', at=now(), audit_sha256=sha256(final),
        note='This supplementary experiment completed; research goal still requires synthesis and delivery.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.project, a.output)
    except BaseException:
        if a.output.exists():
            write_json(a.output/'failure.json', dict(at=now(), traceback=traceback.format_exc()))
        raise
