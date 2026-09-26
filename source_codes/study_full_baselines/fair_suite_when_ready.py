"""等待扩展数据，然后以同一批成员训练全部既有学习基线。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (SOURCE, require_host, check_data, sha256,
    source_record, verify_sources, write_json, now)
from study_full_baselines.runtime_bundle import runtime_sources, verify as verify_bundle
from study_full_baselines.fair_training_common import cohort_metadata
from study_full_baselines.train import verify_labels
from our_method_response_control.train import verify_targets


def process_identity(pid):
    """PID和启动时间共同识别原作业；不保存可能含敏感数据的命令行。"""
    path = Path('/proc')/str(pid)
    try:
        fields = (path/'stat').read_text().split(') ', 1)[1].split()
        command = (path/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError: return None
    if fields[0] == 'Z': return None
    scripts = [Path(x.decode()).name for x in command if x.endswith(b'.py')]
    return dict(pid=pid, start_ticks=int(fields[19]), scripts=scripts)


def state(folder):
    path = Path(folder)/'progress.json'
    return json.loads(path.read_text()) if path.exists() else {}


def memory_available_mib():
    rows = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(rows['MemAvailable'].split()[0])/1024


def source_files():
    result = runtime_sources()
    result.update(source_record(['study_full_baselines/'+name for name in [
        'fair_suite_when_ready.py', 'FAIR_SUITE_QUEUE_PROTOCOL.md', 'prepare_fair_labels.py',
        'prepare_labels.py', 'bind_fair_targets.py', 'scale_large.py',
        'SCALED_FAIR_TRAINING_PROTOCOL.md', 'FAIR_COHORT_PROTOCOL.md', 'SCALE_LARGE_PROTOCOL.md']]))
    return result


def make_plan(project, count):
    """目录和命令预先列明，不读取测试分数挑选训练配置。"""
    if count != 3456: raise ValueError('本流水线只准备既定3456环境的完整同规模对照。')
    root = project/'dataset_simulation'; study = root/'baseline_results/20260925_full_baselines'
    original = root/'outputs/quality_rank_hybrid_20260925'
    data = root/'outputs/scaling_train_3456_20260925'
    view = root/'outputs/fair_view_3456_20260926'
    suite = study/'fair_3456'; scale = study/'scale_3456'
    labels = suite/'teacher_labels'; targets = suite/'targets'
    def command(script, **kwargs):
        args = [sys.executable, '-u', script]
        for name, value in kwargs.items(): args += ['--'+name.replace('_', '-'), str(value)]
        return args
    steps = dict(
        view=command('study_full_baselines/build_fair_view.py', train_data=data, old_test_data=original, output=view),
        labels=command('study_full_baselines/prepare_fair_labels.py', data=view, source_data=original,
            source_labels=root/'outputs/unified_teacher_labels_20260925', output=labels),
        targets=command('study_full_baselines/bind_fair_targets.py', source_data=data,
            source_targets=scale/'targets', view=view, output=targets),
        direct=command('study_full_baselines/train.py', data=view, labels=labels, output=suite/'direct'),
        quality=command('our_method_quality_rank/train_scaled.py', data=view, output=suite/'quality'),
        real=command('baseline_response_realcnn/train_scaled.py', data=view, targets=targets, output=suite/'real'),
        linear=command('baseline_joint_response_linear/train_scaled.py', data=view, targets=targets, output=suite/'linear'),
        bundle=command('study_full_baselines/runtime_bundle.py', data=view, spec=suite/'runtime_spec.json', output=suite/'runtime_bundle'))
    spec = {group:dict(path=str(suite/group), training_data=str(view))
            for group in ['direct', 'quality', 'real', 'linear']}
    spec['direct']['control_labels'] = str(labels)
    for group in ['real', 'linear']: spec[group]['targets'] = str(targets)
    spec['response'] = dict(path=str(scale/'response_n3456_fixed_epochs'), training_data=str(data), targets=str(scale/'targets'))
    return dict(count=count, data=str(data), view=str(view), suite=str(suite), scale=str(scale),
        labels=str(labels), targets=str(targets), steps=steps, runtime_spec=spec,
        final_method_selected=False, fresh_confirmation_started=False)


def run(args):
    require_host(); project = args.project.resolve(); queue = args.queue.resolve()
    if not 1 <= args.workers <= 8: raise ValueError('教师并发须在1到8之间。')
    plan = make_plan(project, args.count)
    expansion = process_identity(args.expansion_pid); scale_owner = process_identity(args.scale_pid)
    if expansion is None or 'expand_gpu_sequence.py' not in expansion['scripts']:
        raise ValueError('原扩展队列身份无法核验。')
    if scale_owner is None or 'scale_large_when_ready.py' not in scale_owner['scripts']:
        raise ValueError('原规模训练队列身份无法核验。')
    queue.mkdir(parents=True, exist_ok=False)
    args.queue_created = True
    frozen = source_files()
    protocol = dict(at=now(), source_sha256=frozen, plan=plan, expansion_owner=expansion,
        scale_owner=scale_owner, max_teacher_workers=args.workers, cpu_threads=1,
        seed=0, epochs=40, gpu_free_minimum_mib=12288, host_available_minimum_mib=12288)
    write_json(queue/'protocol.json', protocol)
    for name in frozen:
        dest = queue/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    def progress(status, **fields):
        write_json(queue/'progress.json', dict(status=status, pid=os.getpid(), at=now(), **fields))
    def wait_dependency(folder, owner, step):
        while True:
            current = state(folder)
            if current.get('status') == 'complete': return
            if list(Path(folder).glob('failure*.json')):
                raise RuntimeError('前置作业有失败记录：'+str(folder))
            if process_identity(owner['pid']) != owner:
                raise RuntimeError('前置尚未完成且原进程身份已消失或改变：'+str(folder))
            progress('waiting', step=step, dependency=str(folder), dependency_state=current,
                dependency_identity_verified=True)
            time.sleep(20)
    def execute(name, command=None):
        verify_sources(frozen)
        command = command or plan['steps'][name]
        with (queue/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
            start = dict(step=name, command=command, pid=child.pid, started_at=now(),
                process_identity=process_identity(child.pid), log=name+'.log')
            write_json(queue/(name+'_started.json'), start)
            progress('running', step=name, child_pid=child.pid, command=command)
            code = child.wait()
        write_json(queue/(name+'_exit.json'), dict(**start, exit_code=code, ended_at=now()))
        if code != 0: raise RuntimeError('子步骤失败，保留原目录：'+name)
        verify_sources(frozen)
    wait_dependency(Path(plan['data']), expansion, 'complete_training_data')
    manifest = check_data(Path(plan['data']))
    if manifest['train_samples'] != args.count*17 or manifest['test_samples'] != 0:
        raise ValueError('扩充数据规模或划分不同。')
    # 本队列只创建自己的新运行；不覆盖其他会话已创建的共同视图/产物。
    if Path(plan['view']).exists() or Path(plan['suite']).exists():
        raise FileExistsError('共同训练目录已存在，应核对后显式续跑，不自动覆盖。')
    Path(plan['suite']).mkdir(parents=True)
    execute('view'); cohort = cohort_metadata(Path(plan['view']))
    if cohort['train_environments'] != args.count: raise ValueError('共同视图规模不符。')
    while memory_available_mib() < 12288:
        progress('waiting_for_host_memory', step='labels', available_mib=memory_available_mib()); time.sleep(20)
    workers = min(args.workers, max(1, len(os.sched_getaffinity(0))//2),
                  max(1, int((memory_available_mib()-8192)//1536)))
    write_json(queue/'teacher_resources.json', dict(at=now(), workers=workers,
        available_mib=memory_available_mib(), affinity_cpus=len(os.sched_getaffinity(0)), load=list(os.getloadavg())))
    execute('labels', plan['steps']['labels']+['--workers', str(workers)])
    verify_labels(Path(plan['view']), Path(plan['labels']))
    scale_queue = project/'dataset_simulation/ops/full_baselines_20260925/scale_gpu_queue'
    wait_dependency(scale_queue, scale_owner, 'complete_existing_scale_training')
    verify_targets(Path(plan['data']), Path(plan['scale'])/'targets')
    execute('targets'); verify_targets(Path(plan['view']), Path(plan['targets']))
    for group in ['direct', 'quality', 'real', 'linear']:
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).splitlines()[0])
            memory = memory_available_mib()
            if free >= 12288 and memory >= 12288: break
            progress('waiting_for_resources', step=group, free_gpu_mib=free, available_host_mib=memory); time.sleep(20)
        execute(group)
    write_json(Path(plan['suite'])/'runtime_spec.json', plan['runtime_spec'])
    execute('bundle'); bundle = verify_bundle(Path(plan['suite'])/'runtime_bundle')
    verify_sources(frozen)
    write_json(queue/'complete.json', dict(status='complete_training_artifacts_only', at=now(),
        train_environments=args.count, methods=bundle['methods'],
        runtime_manifest_sha256=sha256(Path(plan['suite'])/'runtime_bundle/manifest.json'),
        protocol_sha256=sha256(queue/'protocol.json'), final_method_selected=False,
        fresh_confirmation_started=False, quality_evaluation_and_timing_and_website_pending=True))
    progress('complete', scope='common training artifacts and runtime bundle only; final evaluation remains pending')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--count', type=int, choices=[3456], default=3456)
    parser.add_argument('--expansion-pid', type=int, required=True)
    parser.add_argument('--scale-pid', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    try: run(args)
    except BaseException:
        if getattr(args, 'queue_created', False):
            write_json(args.queue/('failure_%d.json'%time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
