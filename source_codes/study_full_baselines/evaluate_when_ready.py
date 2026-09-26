"""七网络训练完成后自动执行统一接收评测；失败保留，不自动覆盖重跑。"""
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
from study_full_baselines.train_when_ready import alive


def run(a):
    require_host();a.queue.mkdir(parents=True,exist_ok=False)
    frozen=source_record(['study_full_baselines/evaluate_learned.py',
        'study_full_baselines/common.py','study_full_baselines/evaluate_when_ready.py'])
    write_json(a.queue/'protocol.json',dict(source_sha256=frozen,training_pid=a.training_pid,
        training_queue=str(a.training_queue),study=str(a.study),at=now()))
    while True:
        f=a.training_queue/'progress.json'
        progress=json.loads(f.read_text()) if f.exists() else {}
        if progress.get('status')=='complete':break
        if (a.training_queue/'failure.json').exists() or not alive(a.training_pid):
            raise RuntimeError('训练队列失败或终止，需要检查原始记录。')
        write_json(a.queue/'progress.json',dict(status='waiting_for_seven_models',pid=os.getpid(),at=now()))
        time.sleep(20)
    verify_sources(frozen)
    command=[sys.executable,'-u','study_full_baselines/evaluate_learned.py',
        '--project',str(a.project),'--data',str(a.data),'--study',str(a.study),
        '--output',str(a.study/'learned_evaluation'),'--workers','6']
    write_json(a.queue/'progress.json',dict(status='evaluation',command=command,pid=os.getpid(),at=now()))
    subprocess.run(command,cwd=SOURCE,check=True)
    write_json(a.queue/'progress.json',dict(status='complete',at=now(),
        remaining='independent audit, complete software timing, expanded data, method optimization, fresh confirmation and website publication'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','data','study','queue','training-queue']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-pid',type=int,required=True);a=p.parse_args()
    try:run(a)
    except BaseException:
        a.queue.mkdir(parents=True,exist_ok=True)
        write_json(a.queue/'failure.json',dict(traceback=traceback.format_exc(),at=now()))
        raise
