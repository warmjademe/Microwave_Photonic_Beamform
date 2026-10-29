"""只等待已经启动的 R/C 评价，再执行冻结统计脚本；不重启实验。"""
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


def write(path,obj):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(path)


def run(project,package):
    root=project/'dataset_simulation/revision_af_20261029'/package
    script=project/'source_codes/revision_af_20261029'/('summarize_r.py' if package=='R' else 'summarize_quality.py')
    digest=hashlib.sha256(script.read_bytes()).hexdigest()
    destination=root/('full' if package=='R' else 'test')
    launch=json.loads((root/('full_launch.json' if package=='R' else 'evaluation_launch.json')).read_text())
    expected='evaluate_r.py' if package=='R' else 'quality_queue.py'
    with (root/'analysis_wait.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (root/'analysis/complete.json').exists():raise RuntimeError('Analysis already complete')
            while not (destination/'complete.json').exists():
                if (destination/'failure.json').exists():raise RuntimeError('Evaluation failure marker exists')
                if package=='C':
                    state=json.loads((root/'queue_state.json').read_text())
                    if state['status']=='failed':raise RuntimeError('Evaluation queue failed')
                command=Path('/proc')/str(launch['pid'])/'cmdline'
                if not command.exists() or expected.encode() not in command.read_bytes():raise RuntimeError('Original evaluation handle is no longer live')
                write(root/'analysis_state.json',dict(status='waiting_for_existing_evaluation',pid=os.getpid(),summary_code_sha256=digest,at=datetime.now(timezone.utc).isoformat()))
                time.sleep(30)
            assert hashlib.sha256(script.read_bytes()).hexdigest()==digest
            with (root/'analysis.log').open('a') as log:
                child=subprocess.Popen([sys.executable,str(script),'--project',str(project)],cwd=project,stdout=log,stderr=subprocess.STDOUT)
                write(root/'analysis_state.json',dict(status='running',pid=os.getpid(),child_pid=child.pid,summary_code_sha256=digest,at=datetime.now(timezone.utc).isoformat()))
                code=child.wait()
            if code:raise RuntimeError('Analysis exit '+str(code))
            assert (root/'analysis/complete.json').exists()
            write(root/'analysis_state.json',dict(status='complete',summary_code_sha256=digest,at=datetime.now(timezone.utc).isoformat()))
        except Exception as e:
            write(root/'analysis_state.json',dict(status='failed',error=str(e),at=datetime.now(timezone.utc).isoformat()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--package',choices=['R','C'],required=True)
    a=p.parse_args();run(a.project.resolve(),a.package)
