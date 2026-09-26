"""等待本轮残差拟合结束，通过共同规模前置核验后自动进行完整探索评分。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,write_json,now,source_record,verify_sources,sha256


def identity(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.read_text().rsplit(')',1)[1].split()[19] if p.exists() else None


def run(project,queue,fit_pid):
    require_host();queue.mkdir(parents=True,exist_ok=False)
    b=project/'dataset_simulation';fit=b/'diagnostics/20260926_measurement_refinement_fit_3456'
    output=b/'baseline_results/20260925_full_baselines/evaluation_measurement_refinement_3456'
    preflight=b/'diagnostics/20260926_measurement_refinement_scaled_preflight.json'
    start=identity(fit_pid)
    if start is None and not (fit/'complete.json').exists():raise ValueError('拟合进程不存在且未完成。')
    sources=source_record(['our_method_measurement_refinement/'+n for n in ['when_ready.py','evaluate_scaled.py','method.py','SCALED_PROTOCOL.md','fit_scaled.py']]+
        ['study_full_baselines/common.py','study_full_baselines/evaluate_learned.py','our_method_response_control/physics.py'])
    write_json(queue/'protocol.json',dict(fit_pid=fit_pid,fit_start_ticks=start,source_sha256=sources,
        train_environments=3456,scope='old216 exploration only; no new final test',at=now()))
    for name in sources:
        dest=queue/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((project/'source_codes'/name).read_bytes())
    while not (fit/'complete.json').exists():
        verify_sources(sources)
        if list(fit.glob('failure*')) or identity(fit_pid)!=start:
            raise RuntimeError('残差拟合失败或原进程已退出，保留现场。')
        write_json(queue/'progress.json',dict(status='waiting_for_fit',pid=os.getpid(),fit_pid=fit_pid,at=now()))
        time.sleep(20)
    if json.loads((fit/'complete.json').read_text())['status']!='complete':raise ValueError('拟合没有通过完成检查。')
    env=os.environ.copy()
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
    for step,dest,extra in [('preflight',preflight,['--preflight']),('evaluation',output,['--workers','2'])]:
        verify_sources(sources)
        cmd=[str(project/'.venv_dl/bin/python'),'-u','our_method_measurement_refinement/evaluate_scaled.py',
            '--project',str(project),'--fit',str(fit),'--output',str(dest),*extra]
        with (queue/(step+'.log')).open('xb') as log:
            child=subprocess.Popen(cmd,cwd=project/'source_codes',env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(queue/'progress.json',dict(status='running',step=step,pid=os.getpid(),child_pid=child.pid,command=cmd,at=now()))
            code=child.wait()
        if code!=0:raise RuntimeError(step+'失败，退出码'+str(code))
        if step=='preflight':
            result=json.loads(preflight.read_text())
            if result['status']!='passed' or not result['original_controls_equal'] or not result['original_metrics_equal']:
                raise ValueError('共同规模前置核验失败。')
    verify_sources(sources)
    write_json(queue/'progress.json',dict(status='complete',at=now(),scope='exploratory reception only; final confirmation, timing, report and website remain',
        evaluation_complete_sha256=sha256(output/'complete.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);p.add_argument('--fit-pid',type=int,required=True);a=p.parse_args()
    try:run(a.project.resolve(),a.queue.resolve(),a.fit_pid)
    except BaseException:
        a.queue.mkdir(parents=True,exist_ok=True);write_json(a.queue/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
