"""等已有正式计时结束后，运行一次独立 A+F 内存测量。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(project):
    root = project / 'dataset_simulation/revision_af_20261029/P'
    here = Path(__file__).resolve().parent
    proof = read(root / 'memory_preflight/complete.json')
    assert proof['status'] == 'passed' and proof['preflight'] and proof['cases'] == 3
    for name, digest in proof['source_sha256'].items():
        assert sha(here / name) == digest
    for name, digest in proof['file_sha256'].items():
        assert sha(root / 'memory_preflight' / name) == digest
    output = root / 'memory_full'
    assert not output.exists()
    with (root / 'memory_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def state(status, **kw):
            value = dict(status=status, pid=os.getpid(), at=datetime.now(timezone.utc).isoformat(),
                         source_sha256=proof['source_sha256'], **kw)
            temp = root / 'memory_queue_state.json.tmp'
            temp.write_text(json.dumps(value, indent=2) + '\n')
            temp.replace(root / 'memory_queue_state.json')
        try:
            while True:
                timing_state = read(root / 'timing_queue_state.json')
                if timing_state['status'] == 'failed':
                    raise RuntimeError('Formal timing prerequisite failed')
                marker = root / 'timing_full/complete.json'
                if marker.exists():
                    result = read(marker)
                    assert result['status'] == 'passed' and result['formal_timing'] and not result['preflight']
                    break
                state('waiting_for_existing_formal_timing', timing_queue_pid=timing_state['pid'])
                time.sleep(30)
            # Import only after timing is complete; no CUDA context while waiting.
            sys.path.insert(0, str(here.parent))
            from study_final_timing.run import timing_resource_snapshot
            from study_full_baselines.benchmark_online import resource_gate
            while True:
                snapshot = timing_resource_snapshot(project, False)
                try:
                    resource_gate(snapshot, True)
                    break
                except RuntimeError:
                    state('waiting_for_exclusive_resources', resource_snapshot=snapshot)
                    time.sleep(30)
            for name, digest in proof['source_sha256'].items():
                assert sha(here / name) == digest
            with (root / 'memory_full.log').open('a') as log:
                child = subprocess.Popen([sys.executable, str(here / 'profile_memory.py'),
                    '--project', str(project), '--output', str(output)], cwd=project,
                    stdout=log, stderr=subprocess.STDOUT)
                state('running', child_pid=child.pid)
                code = child.wait()
            if code:
                raise RuntimeError('Memory measurement exit ' + str(code))
            result = read(output / 'complete.json')
            assert result['status'] == 'passed' and result['formal_memory'] and result['cases'] == 102
            state('complete', evidence_sha256=sha(output / 'complete.json'))
        except Exception as error:
            state('failed', error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    run(parser.parse_args().project.resolve())
