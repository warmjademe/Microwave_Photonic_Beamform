"""等待七条真实评分任务完成后，导出同一个固定案例的全部方法和17载频。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, source_record, verify_sources, write_json, now, sha256
from study_full_baselines.export_signal_examples import PHASES, SOURCES


def alive(pid):
    path = Path('/proc')/str(pid)/'stat'
    if not path.exists():
        return False
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def run(project, queue, output, owners):
    require_host(); study = project/'dataset_simulation/baseline_results/20260925_full_baselines'
    queue.mkdir(parents=True, exist_ok=False)
    sources = source_record(SOURCES+['study_full_baselines/export_examples_when_ready.py',
                                    'study_full_baselines/render_signal_examples.py'])
    if set(owners) != set(PHASES):
        raise ValueError('每条评分流水线都需要绑定当前真实拥有者PID。')
    write_json(queue/'protocol.json', dict(owners=owners, output=str(output),
        source_sha256=sources, frame=5, fixed_test_index=0, carriers=list(range(4, 21)), at=now()))
    while True:
        verify_sources(sources); pending = []
        for phase in PHASES:
            progress = study/phase/'progress.json'
            state = json.loads(progress.read_text()) if progress.exists() else {}
            if state.get('status') == 'complete':
                continue
            pending.append(phase)
            if not alive(owners[phase]):
                raise RuntimeError('评分没有完成且绑定任务已结束：'+phase)
            if list((study/phase).glob('failure*.json')):
                raise RuntimeError('评分出现失败记录：'+phase)
        if not pending:
            break
        write_json(queue/'progress.json', dict(status='waiting', pending=pending, pid=os.getpid(), at=now()))
        time.sleep(30)
    commands = [
        [sys.executable, '-u', 'study_full_baselines/export_signal_examples.py', '--project', str(project), '--output', str(output)],
        [sys.executable, '-u', 'study_full_baselines/render_signal_examples.py', '--data', str(output), '--output', str(output/'figures')]]
    for stage, command in zip(['export', 'render'], commands):
        write_json(queue/'progress.json', dict(status=stage, pid=os.getpid(), command=command, at=now()))
        with (queue/(stage+'.log')).open('x') as stream:
            subprocess.run(command, cwd=project/'source_codes', stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        verify_sources(sources)
    write_json(queue/'progress.json', dict(status='complete',
        scope='old-test fixed examples exported and rendered; visual review and NAS publishing still required',
        summary_sha256=sha256(output/'summary.json'), at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'queue', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--owner', nargs=2, action='append', required=True, metavar=('PHASE', 'PID'))
    a = p.parse_args()
    try:
        run(a.project, a.queue, a.output, {name: int(pid) for name, pid in a.owner})
    except BaseException:
        if a.queue.exists():
            write_json(a.queue/('failure_%d.json' % os.getpid()), dict(traceback=traceback.format_exc(), at=now()))
        raise
