"""记录并接续本轮CPU生成队列；不结束正在执行的组件或反馈评分。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.expand_training_gpu import validate_backend,conflicting_generators
from study_full_baselines.train_when_ready import alive

TARGETS={924050:'study_full_baselines/expand_when_ready.py',
    929175:'study_full_baselines/expand_training.py',
    929187:'study_full_baselines/expand_training.py',929188:'study_full_baselines/expand_training.py',
    929189:'study_full_baselines/expand_training.py',929190:'study_full_baselines/expand_training.py',
    925425:'study_full_baselines/scale_large_when_ready.py',
    926739:'study_full_baselines/variants_when_ready.py',
    932053:'study_full_baselines/analyze_when_ready.py',
    936439:'study_full_baselines/export_examples_when_ready.py'}
KEEP={935938:'study_full_baselines/evaluate_response_variants.py',
      927914:'baseline_response_realcnn/evaluate.py',933590:'our_method_feedback_candidates/evaluate_full.py',
      937277:'our_method_feedback_candidates/analyze_full.py'}


def process(pid,expected):
    root=Path('/proc')/str(pid)
    args=(root/'cmdline').read_bytes().split(b'\0')
    args=[s.decode() for s in args if s]
    stat=(root/'stat').read_text().rsplit(')',1)[1].split()
    if expected not in args or stat[0]=='Z':raise ValueError('进程身份或运行状态不同：'+str(pid))
    return dict(pid=pid,args=args,state=stat[0],ppid=int(stat[1]),start_ticks=int(stat[19]))


def inspect(project,output):
    require_host();base=project/'dataset_simulation';ops=base/'ops/full_baselines_20260925'
    gates=base/'diagnostics/20260926_gpu_generation_replay';validation=validate_backend(gates)
    spawn=base/'diagnostics/20260926_gpu_spawn_check'
    if not json.loads((spawn/'summary.json').read_text())['passed']:raise ValueError('spawn检查未通过。')
    verify_sources(json.loads((spawn/'protocol.json').read_text())['source_sha256'])
    targets={str(pid):process(pid,name) for pid,name in TARGETS.items()}
    keep={str(pid):process(pid,name) for pid,name in KEEP.items()}
    expected={929175,929187,929188,929189,929190}
    data=base/'outputs/scaling_train_1728_20260925'
    if set(conflicting_generators(data))!=expected:raise ValueError('活动生成进程集合改变。')
    statuses={name:json.loads((ops/name/'progress.json').read_text()) for name in
              ['expansion_queue','scale_large_queue','variants_queue','analysis_queue','example_export_queue']}
    if (statuses['scale_large_queue']['status']!='waiting' or statuses['scale_large_queue']['data_ready']
            or statuses['analysis_queue']['status']!='waiting_for_full_reception'
            or statuses['example_export_queue']['status']!='waiting'):
        raise ValueError('将被替换的等待队列已经进入其他阶段。')
    frozen=source_record(['study_full_baselines/'+n for n in [
        'migrate_gpu_expansion.py','expand_gpu_sequence.py','expand_training_gpu.py',
        'variants_gpu_when_ready.py','scale_large_when_ready.py','analyze_when_ready.py',
        'export_examples_when_ready.py']])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'preflight.json',dict(at=now(),targets=targets,keep=keep,queue_states=statuses,
        source_sha256=frozen,validation=validation,spawn_summary_sha256=sha256(spawn/'summary.json'),
        expansion_manifest_sha256=sha256(data/'manifest.json')))
    print(json.dumps(dict(status='preflight_passed',replace_processes=len(targets),
        preserve_processes=len(keep),output=str(output))),flush=True)


def execute(project,output):
    require_host();pre=json.loads((output/'preflight.json').read_text());verify_sources(pre['source_sha256'])
    base=project/'dataset_simulation';ops=base/'ops/full_baselines_20260925'
    study=base/'baseline_results/20260925_full_baselines';data=base/'outputs/scaling_train_1728_20260925'
    gates=base/'diagnostics/20260926_gpu_generation_replay';validate_backend(gates)
    for group,names in [('targets',TARGETS),('keep',KEEP)]:
        for pid,name in names.items():
            current=process(pid,name)
            if current['start_ticks']!=pre[group][str(pid)]['start_ticks']:
                raise ValueError('PID已被重用：'+str(pid))
    if (output/'journal.json').exists():raise FileExistsError('迁移已开始，禁止盲目重复执行。')
    actions=[];jobs={}
    def record(action,**fields):
        actions.append(dict(action=action,at=now(),**fields))
        write_json(output/'journal.json',dict(actions=actions,jobs=jobs))
    def terminate(pid):
        current=process(pid,TARGETS[pid])
        if current['start_ticks']!=pre['targets'][str(pid)]['start_ticks']:raise ValueError('PID身份改变。')
        os.kill(pid,signal.SIGTERM)
        for _ in range(50):
            if not alive(pid):break
            time.sleep(.1)
        if alive(pid):raise RuntimeError('指定进程没有退出：'+str(pid))
        record('terminated_superseded_process',pid=pid)
    def spawn(name,script,arguments):
        command=[sys.executable,'-u',script]+list(map(str,arguments))
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with (output/(name+'.log')).open('xb') as stream:
            child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        jobs[name]=dict(pid=child.pid,command=command)
        record('started_replacement',name=name,pid=child.pid)
        return child.pid
    # 只暂挂监管者。其组件子进程不接收任何信号，继续计算。
    os.kill(926739,signal.SIGSTOP);record('supervisor_suspended_child_continues',pid=926739,child_pid=935938)
    for name in pre['queue_states']:
        shutil.copyfile(ops/name/'progress.json',output/(name+'_before.json'))
    for pid in [932053,936439,925425,924050,929175,929187,929188,929189,929190]:terminate(pid)
    if conflicting_generators(data):raise RuntimeError('旧生成器仍然存在。')
    manifest=json.loads((data/'manifest.json').read_text());members={r['path']:r for r in manifest['environments']}
    complete=[];quarantined=[]
    for folder in sorted((data/'train').glob('environment_*')):
        rel=str(folder.relative_to(data))
        if rel not in members:raise ValueError('出现计划外目录，停止迁移。')
        marker=folder/'complete.json'
        if marker.exists():
            r=json.loads(marker.read_text())
            if r['environment_id']!=members[rel]['environment_id'] or r['path']!=rel:raise ValueError('环境身份不符。')
            for name,digest in r['file_sha256'].items():
                if sha256(folder/name)!=digest:raise ValueError('已完成样本哈希改变。')
            complete.append(r)
        else:
            quarantine=output/'quarantine'/folder.name;quarantine.parent.mkdir(exist_ok=True)
            folder.rename(quarantine);quarantined.append(rel)
    write_json(output/'preserved_records.json',complete)
    record('verified_checkpoint_boundary',completed=len(complete),quarantined=quarantined,
        preserved_records_sha256=sha256(output/'preserved_records.json'))
    expansion_queue=ops/'expansion_gpu_queue';scale_queue=ops/'scale_gpu_queue'
    variants_queue=ops/'variants_gpu_queue';analysis_queue=ops/'analysis_gpu_queue'
    examples_queue=ops/'example_export_gpu_queue'
    for folder in [expansion_queue,scale_queue,variants_queue,analysis_queue,examples_queue]:
        if folder.exists():raise FileExistsError('新队列目录已经存在：'+str(folder))
    expansion=spawn('expansion','study_full_baselines/expand_gpu_sequence.py',[
        '--project',project,'--queue',expansion_queue,'--gates',gates,'--workers',4])
    scale=spawn('scale','study_full_baselines/scale_large_when_ready.py',[
        '--project',project,'--queue',scale_queue,'--expansion-pid',expansion,'--training-pid',923115])
    variants=spawn('variants','study_full_baselines/variants_gpu_when_ready.py',[
        '--project',project,'--queue',variants_queue,'--component-pid',935938,
        '--old-supervisor-pid',926739,'--old-supervisor-start-ticks',pre['targets']['926739']['start_ticks'],
        '--scale-queue-pid',scale])
    analysis=spawn('analysis','study_full_baselines/analyze_when_ready.py',[
        '--project',project,'--queue',analysis_queue,'--output',study/'analysis',
        '--classic-pid',923113,'--learned-queue-pid',923572,'--variants-queue-pid',variants,'--real-queue-pid',927914])
    owner_args=[]
    for name,pid in dict(classic=923113,learned_evaluation=923572,evaluation_components=variants,
        evaluation_real_response_cnn=927914,evaluation_scaling_1728=variants,
        evaluation_scaling_3456=variants,evaluation_feedback_warm=933590).items():
        owner_args+=['--owner',name,pid]
    examples=spawn('examples','study_full_baselines/export_examples_when_ready.py',[
        '--project',project,'--queue',examples_queue,'--output',base/'diagnostics/20260926_all_method_signal_examples',*owner_args])
    for old,new,pid in [('expansion_queue',expansion_queue,expansion),('scale_large_queue',scale_queue,scale),
        ('analysis_queue',analysis_queue,analysis),('example_export_queue',examples_queue,examples)]:
        write_json(ops/old/'progress.json',dict(status='superseded',replacement_queue=str(new),replacement_pid=pid,at=now()))
    write_json(ops/'variants_queue/progress.json',dict(status='supervisor_suspended_for_handoff',
        component_child_pid=935938,component_continues=True,replacement_queue=str(variants_queue),replacement_pid=variants,at=now()))
    # 启动后的短检查只确认拥有者确实存活；逐环境进度由后续观察核验。
    time.sleep(3)
    for name,info in jobs.items():
        if not alive(info['pid']):raise RuntimeError('替换队列启动后退出：'+name)
    for pid,name in KEEP.items():
        if process(pid,name)['start_ticks']!=pre['keep'][str(pid)]['start_ticks']:
            raise RuntimeError('被保留的评测进程身份改变。')
    write_json(output/'complete.json',dict(status='handoff_started',at=now(),jobs=jobs,
        preserved_environments=len(complete),quarantined=quarantined,
        old_supervisor_pending_retirement=926739,component_child_preserved=935938,
        next_check='new CUDA worker processes and additional committed environments'))
    print(json.dumps(dict(status='handoff_started',jobs={k:v['pid'] for k,v in jobs.items()},
        preserved_environments=len(complete))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    try:(execute if a.execute else inspect)(a.project,a.output)
    except BaseException:
        if a.output.exists():write_json(a.output/('failure_%d.json'%os.getpid()),dict(traceback=traceback.format_exc(),at=now()))
        raise
