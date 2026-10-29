"""H 全量评价与完整性汇总的单实例队列；不改变已冻结配置。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def run(project):
    root=project/'dataset_simulation/revision_af_20261029/H'
    source=project/'source_codes/revision_af_20261029'
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scripts={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['evaluate_h.py','summarize_h.py']}
        pre=json.loads((root/'preflight/complete.json').read_text())
        assert pre['status']=='complete' and pre['code_sha256']==scripts['evaluate_h.py']
        for label,script,extra in [('evaluation','evaluate_h.py',['--output',str(root/'full'),'--workers','4']),('analysis','summarize_h.py',[])]:
            destination=root/('full' if label=='evaluation' else 'analysis')
            if (destination/'complete.json').exists():
                raise RuntimeError('Inspect completed output before restarting this queue')
            with (root/(label+'.log')).open('a') as log:
                child=subprocess.Popen([sys.executable,str(source/script),'--project',str(project)]+extra,stdout=log,stderr=subprocess.STDOUT,cwd=project)
                write(root/'queue_state.json',dict(status='running',task=label,pid=os.getpid(),child_pid=child.pid,source_sha256=scripts,at=datetime.now(timezone.utc).isoformat()))
                rc=child.wait()
            if rc:
                write(root/'queue_state.json',dict(status='failed',task=label,exit_code=rc,at=datetime.now(timezone.utc).isoformat()))
                raise RuntimeError(f'{label} failed: {rc}')
        write(root/'queue_state.json',dict(status='complete',source_sha256=scripts,at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
