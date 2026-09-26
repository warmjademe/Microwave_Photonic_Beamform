"""等统一标签完整、规模对照释放GPU后，执行七项基线训练；不重复启动。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE,require_host,source_record,verify_sources,write_json,now


def alive(pid):
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    stat=Path('/proc')/str(pid)/'stat'
    return not stat.exists() or stat.read_text().split(') ',1)[1].split()[0]!='Z'


def run(a):
    require_host();a.queue.mkdir(parents=True,exist_ok=False)
    files=['study_full_baselines/train.py','study_full_baselines/common.py',
        'study_full_baselines/train_when_ready.py']
    frozen=source_record(files)
    write_json(a.queue/'protocol.json',dict(source_sha256=frozen,label_pid=a.labels_pid,
        scale_pid=a.scale_pid,labels=str(a.labels),scale=str(a.scale),output=str(a.output),at=now()))
    dependencies=[(a.labels,a.labels_pid),(a.scale,a.scale_pid)]
    while True:
        complete=[]
        for path,pid in dependencies:
            progress=path/'progress.json'
            state=json.loads(progress.read_text()) if progress.exists() else {}
            done=state.get('status')=='complete';complete.append(done)
            if not done and not alive(pid):
                raise RuntimeError('前序进程已终止而尚未完成：'+str(path))
            if list(path.glob('failure_*.json')):
                raise RuntimeError('前序任务留下失败记录，先核查：'+str(path))
        if all(complete):break
        write_json(a.queue/'progress.json',dict(status='waiting_for_labels_and_scale',
            labels_complete=complete[0],scale_complete=complete[1],pid=os.getpid(),at=now()))
        time.sleep(20)
    verify_sources(frozen)
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free>=10000:break
        write_json(a.queue/'progress.json',dict(status='waiting_for_free_GPU',free_MiB=free,at=now()))
        time.sleep(20)
    cmd=[sys.executable,'-u','study_full_baselines/train.py','--data',str(a.data),
        '--labels',str(a.labels),'--output',str(a.output)]
    write_json(a.queue/'progress.json',dict(status='training',command=cmd,pid=os.getpid(),at=now()))
    subprocess.run(cmd,cwd=SOURCE,check=True)
    write_json(a.queue/'progress.json',dict(status='complete',scope='seven baseline models trained; evaluation still required',at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','labels','scale','output','queue']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--labels-pid',type=int,required=True);p.add_argument('--scale-pid',type=int,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException:
        a.queue.mkdir(parents=True,exist_ok=True)
        write_json(a.queue/'failure.json',dict(traceback=traceback.format_exc(),at=now()))
        raise
