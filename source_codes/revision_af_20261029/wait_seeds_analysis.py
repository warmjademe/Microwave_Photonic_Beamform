"""等待现有 S 队列结束后汇总；不重新训练或重启评价。"""
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


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def run(project):
    root=project/'dataset_simulation/revision_af_20261029/S'
    script=project/'source_codes/revision_af_20261029/summarize_seeds.py'
    digest=hashlib.sha256(script.read_bytes()).hexdigest()
    with (root/'analysis_wait.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (root/'analysis/complete.json').exists():raise RuntimeError('Analysis already complete')
            while True:
                state=json.loads((root/'queue_state.json').read_text())
                if state['status']=='evaluations_complete':break
                if state['status']!='running':raise RuntimeError('Evaluation queue is not running: '+state['status'])
                os.kill(state['queue_pid'],0)
                write(root/'analysis_state.json',dict(status='waiting_for_existing_evaluation',pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),summary_code_sha256=digest))
                time.sleep(30)
            assert hashlib.sha256(script.read_bytes()).hexdigest()==digest
            with (root/'analysis.log').open('a') as log:
                child=subprocess.Popen([sys.executable,str(script),'--project',str(project)],cwd=project,stdout=log,stderr=subprocess.STDOUT)
                write(root/'analysis_state.json',dict(status='running',pid=os.getpid(),child_pid=child.pid,summary_code_sha256=digest,at=datetime.now(timezone.utc).isoformat()))
                rc=child.wait()
            if rc:raise RuntimeError('Summary failed: '+str(rc))
            assert (root/'analysis/complete.json').exists()
            write(root/'analysis_state.json',dict(status='complete',summary_code_sha256=digest,at=datetime.now(timezone.utc).isoformat()))
        except Exception as e:
            write(root/'analysis_state.json',dict(status='failed',error=str(e),at=datetime.now(timezone.utc).isoformat()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
