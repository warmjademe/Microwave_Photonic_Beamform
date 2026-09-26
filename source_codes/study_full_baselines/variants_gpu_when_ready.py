"""保留正在执行的组件评分；其结束后接续新GPU生成队列的大规模评分。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.train_when_ready import alive


def run(a):
    require_host();a.queue.mkdir(parents=True,exist_ok=False)
    frozen=source_record(['study_full_baselines/variants_gpu_when_ready.py',
        'study_full_baselines/evaluate_response_variants.py','study_full_baselines/evaluate_learned.py',
        'study_full_baselines/common.py','our_method_two_stage/control.py',
        'our_method_two_stage/decode_multistart.py','our_method_response_control/physics.py'])
    write_json(a.queue/'protocol.json',dict(source_sha256=frozen,at=now(),
        existing_component_pid=a.component_pid,retiring_supervisor_pid=a.old_supervisor_pid,
        retiring_supervisor_start_ticks=a.old_supervisor_start_ticks,
        new_scale_queue_pid=a.scale_queue_pid,suites=['scaling_1728','scaling_3456'],workers=4))
    base=a.project/'dataset_simulation';study=base/'baseline_results/20260925_full_baselines'
    component=study/'evaluation_components'
    while True:
        progress=json.loads((component/'progress.json').read_text())
        if list(component.glob('failure*.json')):raise RuntimeError('原组件评测失败。')
        if progress.get('status')=='complete' and not alive(a.component_pid):break
        if not alive(a.component_pid) and progress.get('status')!='complete':
            raise RuntimeError('组件计算已退出但没有完整结果。')
        write_json(a.queue/'progress.json',dict(status='waiting_for_existing_components',
            completed=progress.get('completed'),component_pid=a.component_pid,pid=os.getpid(),at=now()))
        time.sleep(10)
    # 旧监管进程先前被SIGSTOP，子评测一直运行；现在只结束旧监管者。
    if alive(a.old_supervisor_pid):
        root=Path('/proc')/str(a.old_supervisor_pid)
        parts=(root/'stat').read_text().rsplit(')',1)[1].split()
        command=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if (int(parts[19])!=a.old_supervisor_start_ticks or parts[0] not in ['T','t']
                or 'study_full_baselines/variants_when_ready.py' not in command):
            raise RuntimeError('旧监管者身份或暂停状态不符，禁止发送信号。')
        os.kill(a.old_supervisor_pid,signal.SIGTERM);os.kill(a.old_supervisor_pid,signal.SIGCONT)
        for _ in range(50):
            if not alive(a.old_supervisor_pid):break
            time.sleep(.1)
        if alive(a.old_supervisor_pid):raise RuntimeError('旧监管者未终止。')
    old_queue=base/'ops/full_baselines_20260925/variants_queue'
    write_json(old_queue/'progress.json',dict(status='superseded',replacement_queue=str(a.queue),
        replacement_pid=os.getpid(),at=now(),completed_component_results_preserved=True))
    for count in [1728,3456]:
        suite='scaling_%d'%count;prerequisite=study/('scale_%d'%count)
        while True:
            f=prerequisite/'progress.json';state=json.loads(f.read_text()) if f.exists() else {}
            if state.get('status')=='complete':break
            if list(prerequisite.rglob('failure*.json')) or not alive(a.scale_queue_pid):
                raise RuntimeError('大规模训练未完成且前置队列异常。')
            write_json(a.queue/'progress.json',dict(status='waiting',suite=suite,
                scale_queue_pid=a.scale_queue_pid,pid=os.getpid(),at=now()))
            time.sleep(20)
        verify_sources(frozen)
        command=[sys.executable,'-u','study_full_baselines/evaluate_response_variants.py',
            '--project',str(a.project),'--data',str(base/'outputs/quality_rank_hybrid_20260925'),
            '--study',str(study),'--output',str(study/('evaluation_'+suite)),
            '--suite',suite,'--workers','4']
        write_json(a.queue/'progress.json',dict(status='evaluating',suite=suite,
            command=command,pid=os.getpid(),at=now()))
        with (a.queue/(suite+'.log')).open('x') as stream:
            subprocess.run(command,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True)
    verify_sources(frozen)
    write_json(a.queue/'progress.json',dict(status='complete',at=now(),
        scope='existing components preserved and expanded response evaluations complete; fresh confirmation remains'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','queue']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['component-pid','old-supervisor-pid','old-supervisor-start-ticks','scale-queue-pid']:
        p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException:
        if a.queue.exists():write_json(a.queue/'failure.json',dict(traceback=traceback.format_exc(),at=now()))
        raise
