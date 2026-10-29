"""等待既有质量网络评价结束，再运行同响应监督补测；不重启旧任务。"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(p)


def run(project):
    source=project/'source_codes/revision_af_20261029';base=project/'dataset_simulation/revision_af_20261029'
    output=base/'C_response';output.mkdir(exist_ok=True)
    names=['evaluate_real_response.py','C_RESPONSE_PROTOCOL.md','analyze_real_response.py']
    frozen={n:digest(source/n) for n in names}
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=read(output/'preflight/complete.json')
        assert proof['code_sha256']==frozen['evaluate_real_response.py'] and proof['preflight']
        while not (base/'C/test/complete.json').exists():
            state=read(base/'C/queue_state.json')
            if state['status']=='failed':raise RuntimeError('Existing C evaluation failed; inspect before continuing')
            pid=state.get('child_pid');cmd=Path(f'/proc/{pid}/cmdline')
            if not cmd.exists():raise RuntimeError('Existing C child no longer live and completion absent')
            assert 'evaluate_quality.py' in cmd.read_text()
            write(output/'queue_state.json',dict(status='waiting_for_existing_C',pid=os.getpid(),
                observed_child_pid=pid,source_sha256=frozen,at=datetime.now(timezone.utc).isoformat()))
            time.sleep(30)
        assert read(base/'C/test/complete.json')['environments']==864
        for task,args,done in [
            ('evaluate',[str(source/'evaluate_real_response.py'),'--project',str(project),'--workers','4'],output/'full/complete.json'),
            ('analyze',[str(source/'analyze_real_response.py'),'--project',str(project)],output/'analysis/complete.json')]:
            assert all(digest(source/n)==h for n,h in frozen.items())
            if done.exists():continue
            with (output/(task+'.log')).open('a') as log:
                child=subprocess.Popen([sys.executable]+args,cwd=project,stdout=log,stderr=subprocess.STDOUT)
                write(output/'queue_state.json',dict(status='running',task=task,pid=os.getpid(),child_pid=child.pid,
                    source_sha256=frozen,at=datetime.now(timezone.utc).isoformat()))
                rc=child.wait()
            if rc:raise RuntimeError(f'{task} failed with code {rc}')
            assert done.exists()
        write(output/'queue_state.json',dict(status='complete',source_sha256=frozen,at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    try:run(a.project.resolve())
    except BaseException:
        import traceback
        output=a.project/'dataset_simulation/revision_af_20261029/C_response'
        write(output/'queue_failure.json',dict(at=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
