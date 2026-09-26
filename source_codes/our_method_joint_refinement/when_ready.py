"""先等待全基线审核完成，再执行预设的一轮训练内组件诊断。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,write_json,now,source_record,verify_sources,SOURCE,sha256


def process_identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return None if fields[0]=='Z' else dict(pid=pid,start_ticks=fields[19])


def run(a):
    require_host();project=a.project.resolve();queue=a.queue.resolve();evaluation=a.evaluation.resolve()
    owner=process_identity(a.evaluation_pid)
    if owner is None and not (evaluation/'analysis/complete.json').exists():
        raise ValueError('全基线作业不存在且尚未完成。')
    sources=source_record(['our_method_joint_refinement/'+n for n in
        ['when_ready.py','experiment.py','method.py','PROTOCOL.md']])
    queue.mkdir(parents=True,exist_ok=False)
    output=project/'dataset_simulation/diagnostics/20260926_joint_refinement_train3456'
    protocol=dict(at=now(),source_sha256=sources,evaluation=str(evaluation),owner=owner,
        output=str(output),scope='one predefined training-only hypothesis after full baseline analysis',
        new_test_opened=False,network_updates=0)
    write_json(queue/'protocol.json',protocol)
    for name in sources:
        dest=queue/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
    while not (evaluation/'analysis/complete.json').exists():
        verify_sources(sources)
        if list(evaluation.glob('failure*')) or process_identity(a.evaluation_pid)!=owner:
            raise RuntimeError('全基线评测或分析未完成即停止；不跳过前置步骤。')
        progress=json.loads((evaluation/'progress.json').read_text())
        write_json(queue/'progress.json',dict(status='waiting_for_full_baseline_analysis',at=now(),
            pid=os.getpid(),evaluation_completed=progress['completed'],evaluation_total=progress['total']))
        time.sleep(20)
    analysis=json.loads((evaluation/'analysis/complete.json').read_text())
    if analysis['status']!='complete_exploratory_analysis': raise ValueError('前置分析完成格式不同。')
    summary=json.loads((evaluation/'analysis/summary.json').read_text())
    winners={str(b):min([r for r in summary if r['measurement_budget']==b],key=lambda r:r['ber'])
             for b in [16,64]}
    write_json(queue/'baseline_context.json',dict(at=now(),winners=winners,
        analysis_complete_sha256=sha256(evaluation/'analysis/complete.json'),
        next_hypothesis='joint frequency residual correlation; predeclared, no new final test'))
    while True:
        available=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        memory={line.split(':')[0]:line.split(':')[1] for line in Path('/proc/meminfo').read_text().splitlines()}
        if available>=12288 and int(memory['MemAvailable'].split()[0])>=16*1024**2:break
        write_json(queue/'progress.json',dict(status='waiting_for_resources',at=now(),pid=os.getpid(),free_gpu_mib=available))
        time.sleep(20)
    verify_sources(sources)
    command=[str(project/'.venv_dl/bin/python'),'-u','our_method_joint_refinement/experiment.py',
        '--project',str(project),'--output',str(output),'--after-evaluation',str(evaluation),'--workers','4']
    with (queue/'experiment.log').open('x') as log:
        child=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
        write_json(queue/'progress.json',dict(status='running_training_only_hypothesis',at=now(),
            pid=os.getpid(),child_pid=child.pid,command=command))
        code=child.wait()
    if code!=0:raise RuntimeError('训练内候选检验失败，保留原始记录。')
    verify_sources(sources)
    write_json(queue/'progress.json',dict(status='complete_training_only_hypothesis',at=now(),
        experiment_complete_sha256=sha256(output/'complete.json'),
        overall_research_complete=False,requires_result_interpretation=True,final_confirmation_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['project','queue','evaluation']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--evaluation-pid',type=int,required=True);a=p.parse_args()
    try:run(a)
    except BaseException:
        if a.queue.exists():write_json(a.queue/'failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
