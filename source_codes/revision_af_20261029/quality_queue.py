"""四个质量训练配置按冻结顺序完成验证与既有测试集评价。"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def write(path,item):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(item,indent=2));t.replace(path)


def run(project):
    root=project/'dataset_simulation/revision_af_20261029/C';source=project/'source_codes/revision_af_20261029/evaluate_quality.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for role,preflight in [('validation',True),('validation',False),('test',True),('test',False)]:
            name=role+'_preflight' if preflight else role;done=root/name/'complete.json'
            if done.exists():
                p=json.loads(done.read_text());assert p['code_sha256']==digest and p['role']==role and p['preflight']==preflight
                continue
            args=[sys.executable,str(source),'--project',str(project),'--role',role,'--workers','4']
            if preflight:args+=['--preflight']
            with (root/(name+'.log')).open('a') as log:
                child=subprocess.Popen(args,cwd=project,stdout=log,stderr=subprocess.STDOUT)
                write(root/'queue_state.json',dict(status='running',task=name,pid=os.getpid(),child_pid=child.pid,code_sha256=digest,at=datetime.now(timezone.utc).isoformat()))
                rc=child.wait()
            if rc:
                write(root/'queue_state.json',dict(status='failed',task=name,exit_code=rc,at=datetime.now(timezone.utc).isoformat()))
                raise RuntimeError(name+' failed')
        write(root/'queue_state.json',dict(status='evaluations_complete',code_sha256=digest,analysis_complete=False,at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);run(p.parse_args().project.resolve())
