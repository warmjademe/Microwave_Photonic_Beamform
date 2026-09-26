"""等全部六阶段评分完成后执行统一审计/统计；不启动或重启上游实验。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.train_when_ready import alive


def run(a):
    require_host();a.queue.mkdir(parents=True,exist_ok=False)
    base=a.project/'dataset_simulation';study=base/'baseline_results/20260925_full_baselines'
    paths=dict(classic=a.classic_pid,learned_evaluation=a.learned_queue_pid,
        evaluation_components=a.variants_queue_pid,evaluation_real_response_cnn=a.real_queue_pid,
        evaluation_scaling_1728=a.variants_queue_pid,evaluation_scaling_3456=a.variants_queue_pid)
    files=['study_full_baselines/'+name for name in ['analyze_when_ready.py','analyze_reception.py',
        'audit_reception_v2.py','audit_results.py','paired_statistics.py',
        'EXPLORATORY_COMPARISONS.json','ANALYSIS_PROTOCOL.md','common.py']]
    frozen=source_record(files)
    write_json(a.queue/'protocol.json',dict(source_sha256=frozen,prerequisite_owner_pids=paths,
        output=str(a.output),scope='old216 exploration only; does not finish the full research goal',at=now()))
    while True:
        pending=[]
        for name,pid in paths.items():
            folder=study/name;path=folder/'progress.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            if list(folder.glob('failure*.json')):raise RuntimeError('前置评分出现失败记录：'+name)
            if state.get('status')=='complete':continue
            if not alive(pid):raise RuntimeError('前置评分未完成且归属进程退出：'+name)
            pending.append(name)
        if not pending:break
        write_json(a.queue/'progress.json',dict(status='waiting_for_full_reception',pending=pending,
            pid=os.getpid(),at=now()))
        time.sleep(20)
    verify_sources(frozen)
    cmd=[sys.executable,'-u','study_full_baselines/analyze_reception.py',
        '--data',str(base/'outputs/quality_rank_hybrid_20260925'),'--study',str(study),
        '--output',str(a.output)]
    write_json(a.queue/'progress.json',dict(status='analyzing',pid=os.getpid(),command=cmd,at=now()))
    subprocess.run(cmd,cwd=SOURCE,check=True)
    verify_sources(frozen)
    write_json(a.queue/'progress.json',dict(status='complete',scope='exploratory audit/statistics only',at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','queue','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['classic-pid','learned-queue-pid','variants-queue-pid','real-queue-pid']:
        p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException:
        a.queue.mkdir(parents=True,exist_ok=True)
        write_json(a.queue/'failure.json',dict(traceback=traceback.format_exc(),at=now()));raise
