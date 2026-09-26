"""标签任务释放CPU后，嵌套扩展到1728和3456训练环境；不运行最终测试。"""
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
    frozen=source_record(['study_full_baselines/expand_training.py',
        'study_full_baselines/common.py','study_full_baselines/expand_when_ready.py'])
    write_json(a.queue/'protocol.json',dict(source_sha256=frozen,labels_pid=a.labels_pid,
        targets=[1728,3456],additional_seeds=[2026092507,2026092508],workers=[4,6],at=now()))
    while True:
        progress=a.labels/'progress.json'
        state=json.loads(progress.read_text()) if progress.exists() else {}
        if state.get('status')=='complete':break
        if not alive(a.labels_pid) or list(a.labels.glob('failure_*.json')):
            raise RuntimeError('标签任务异常，需要先核查。')
        write_json(a.queue/'progress.json',dict(status='waiting_for_CPU_release',pid=os.getpid(),at=now()))
        time.sleep(20)
    verify_sources(frozen)
    base=a.project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    for target,seed,workers in [(1728,2026092507,4),(3456,2026092508,6)]:
        output=a.project/('dataset_simulation/outputs/scaling_train_%d_20260925'%target)
        cmd=[sys.executable,'-u','study_full_baselines/expand_training.py','--project',str(a.project),
            '--base',str(base),'--output',str(output),'--target',str(target),
            '--seed',str(seed),'--workers',str(workers)]
        write_json(a.queue/'progress.json',dict(status='expanding',target=target,command=cmd,pid=os.getpid(),at=now()))
        subprocess.run(cmd,cwd=SOURCE,check=True)
        base=output
    write_json(a.queue/'progress.json',dict(status='complete',at=now(),
        scope='expanded training observations and quality labels only; large-scale training/evaluation and final confirmation remain required'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','labels','queue']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--labels-pid',type=int,required=True);a=p.parse_args()
    try:run(a)
    except BaseException:
        a.queue.mkdir(parents=True,exist_ok=True)
        write_json(a.queue/'failure.json',dict(traceback=traceback.format_exc(),at=now()))
        raise
