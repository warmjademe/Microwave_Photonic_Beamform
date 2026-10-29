"""等待已有补测与统计结束、资源空闲后，执行一次 A/F 正式计时。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from revision_af_20261029.benchmark_af import identity
from study_final_timing.run import timing_resource_snapshot
from study_full_baselines.benchmark_online import resource_gate
from study_full_baselines.common import write_json,now,sha256


def read(p):return json.loads(p.read_text())


def run(project):
    root=project/'dataset_simulation/revision_af_20261029';folder=root/'P'
    proof=folder/'timing_preflight03';out=folder/'timing_full'
    signature=identity();passed=read(proof/'complete.json')
    assert passed['status']=='passed' and passed['source_sha256']==signature and passed['preflight']
    with (folder/'timing_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not out.exists()
        def state(status,**kwargs):
            write_json(folder/'timing_queue_state.json',dict(status=status,pid=os.getpid(),at=now(),source_sha256=signature,**kwargs))
        try:
            while True:
                pending=[]
                for pkg in ['H','R','C']:
                    for n in ['queue_state.json','analysis_state.json']:
                        p=root/pkg/n
                        if p.exists() and read(p).get('status')=='failed':raise RuntimeError(pkg+' prerequisite failed')
                    if not (root/pkg/'analysis/complete.json').exists():pending.append(pkg)
                if pending:
                    state('waiting_for_existing_experiments',packages=pending);time.sleep(30);continue
                snapshot=timing_resource_snapshot(project,False)
                try:resource_gate(snapshot,True)
                except RuntimeError:
                    state('waiting_for_exclusive_resources',resource_snapshot=snapshot);time.sleep(30);continue
                break
            assert identity()==signature
            script=project/'source_codes/revision_af_20261029/benchmark_af.py'
            with (folder/'timing_full.log').open('a') as log:
                child=subprocess.Popen([sys.executable,str(script),'--project',str(project),
                    '--output',str(out),'--proof',str(proof)],cwd=project,stdout=log,stderr=subprocess.STDOUT)
                state('running',child_pid=child.pid);code=child.wait()
            if code:raise RuntimeError('Timing exit '+str(code))
            result=read(out/'complete.json')
            assert result['status']=='passed' and result['formal_timing']
            state('complete',evidence_sha256=sha256(out/'complete.json'))
        except Exception as error:
            state('failed',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
