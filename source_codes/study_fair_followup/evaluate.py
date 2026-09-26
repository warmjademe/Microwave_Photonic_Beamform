"""共同3,456模型的全部旧216探索评测；不接受新留出环境。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (SOURCE, require_host, check_data, sha256,
    source_record, verify_sources, write_json, now, LEVELS)
from study_full_baselines.runtime_bundle import verify, load_models, fingerprint
from study_full_baselines import confirmation_batch as batch
from study_full_baselines.confirmation_freeze import budget
from study_full_baselines.confirmation_receiver import METRIC_ORDER
from study_full_baselines.analyze_confirmation import check_values
from study_full_baselines.expand_training_gpu import validate_backend
from our_method_response_control.train import precision
from our_method_response_control.physics import decode
from our_method_measurement_refinement.method import refine
from study_fair_followup.controllers import binding, augment, EXTRA


def read(path):
    return json.loads(Path(path).read_text())


def identity_for(project, preflight):
    root = project/'dataset_simulation'
    bundle = root/'baseline_results/20260925_full_baselines/fair_3456/runtime_bundle'
    package = verify(bundle)
    if package['cohort']['train_environments'] != 3456:
        raise ValueError('必须使用共同3,456模型。')
    data = root/'outputs/quality_rank_hybrid_20260925'
    manifest = check_data(data)
    rows = [r for r in manifest['environments'] if r['split'] == 'test']
    if len(rows) != 216 or [r['environment_id'] for r in rows] != package['cohort']['test_ids']:
        raise ValueError('只能评测已使用的旧216探索环境。')
    bound = binding(project, package)
    proof = read(root/'diagnostics/20260926_confirmation_receiver_preflight/summary.json')
    sources = dict(package['source_sha256'])
    for other in [proof['source_sha256'], bound['source_sha256']]:
        for name, digest in other.items():
            if name in sources and sources[name] != digest:
                raise ValueError('已有核验源码身份冲突。')
            sources[name] = digest
    sources.update(source_record(['study_fair_followup/'+n for n in
        ['evaluate.py', 'controllers.py', 'analyze.py', 'PROTOCOL.md']]+
        ['study_full_baselines/'+n for n in ['confirmation_batch.py', 'confirmation_receiver.py',
         'analyze_confirmation.py', 'paired_statistics.py', 'analyze_reception.py']]))
    verify_sources(sources)
    selected = rows[:1] if preflight else rows
    records = {r['environment_id']: r for r in read(data/'records.json')}
    ordinary = package['methods']+list(EXTRA)
    return dict(schema='fair3456-full-exploratory-v1', project=str(project),
        scope='old216_preflight' if preflight else 'old216_exploratory',
        rows=selected, carriers=[4, 12, 20] if preflight else list(range(4, 21)),
        data_path=str(data), data_manifest_sha256=sha256(data/'manifest.json'),
        runtime_bundle=str(bundle), runtime_manifest_sha256=sha256(bundle/'manifest.json'),
        common_training_environments=3456, ordinary_methods=ordinary, methods=ordinary+['teacher', 'mrc'],
        metric_order=METRIC_ORDER, measurement_budgets={n: budget(n) for n in ordinary},
        source_sha256=sources, refinement=bound, final_confirmation=False,
        new_environment_signals_generated=0,
        input_sha256={r['environment_id']: records[r['environment_id']]['file_sha256'] for r in selected},
        backend='cuda_fft_cpu_rk4_v1',
        backend_proof=validate_backend(root/'diagnostics/20260926_gpu_generation_replay'))


def initialize(output, identity):
    require_host(); precision()
    verify_sources(identity['source_sha256'])
    bundle = Path(identity['runtime_bundle'])
    if sha256(bundle/'manifest.json') != identity['runtime_manifest_sha256']:
        raise ValueError('模型包改变。')
    with np.load(bundle/'public.npz') as f:
        public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    with patch('numpy.load', batch.guard_load(np.load)):
        models = augment(load_models(bundle, public), identity['refinement'])
    if list(models) != identity['ordinary_methods']:
        raise ValueError('方法清单不同。')
    batch.STATE.update(output=Path(output), identity=identity, public=public, models=models)
    write_json(Path(output)/'workers'/('%d.json'%os.getpid()), dict(pid=os.getpid(), at=now(),
        cuda_device=torch.cuda.get_device_name(), threads=torch.get_num_threads(), methods=len(models)))


def preflight_checks(output, identity):
    initialize(output, identity)
    names = identity['methods']; models = batch.STATE['models']; public = batch.STATE['public']
    changes = []; checks = 0; trace_count = 0
    old_folder = Path(identity['project'])/'dataset_simulation/baseline_results/20260925_full_baselines/evaluation_measurement_refinement_3456'
    old_path = old_folder/'records/environment_00000.npz'
    old_mark = read(old_path.with_suffix('.json'))
    if sha256(old_path) != old_mark['sha256']:
        raise ValueError('原校正实验记录改变。')
    old_names = [a+'__'+v for a in ['covariance', 'cnn'] for v in ['original', 'isotropic', 'spatial']]
    with np.load(old_path) as f:
        old_codes = f['control_code'].copy(); old_metrics = f['metrics'].copy()
    with torch.inference_mode():
        for fc in identity['carriers']:
            path = output/'records/environment_00000'/('carrier_%02d.npz'%fc)
            with np.load(path) as f:
                arrays = {k: f[k].copy() for k in f.files}
            trace_count += check_values(arrays, fc, identity, public)
            x = arrays['public_X']
            for name, (base, _, mode) in EXTRA.items():
                model = models[name]
                h = models[base].estimate(x)
                corrected, detail = refine(x, public['pilot_qpsk'], h, model.error_covariance, mode)
                start = public['probe_controls'][int(x[1985:2001].argmax())]
                u, _ = decode(corrected, fc, start, sweeps=2)
                np.testing.assert_array_equal(np.rint(u*LEVELS), arrays['control_code'][names.index(name)])
                if detail['extra_feedback'] != 0:
                    raise ValueError('校正引入额外反馈。')
                checks += 1
            for oi, old_name in enumerate(old_names):
                name = {'covariance__original': 'covariance_response',
                        'cnn__original': 'complex_response_cnn'}.get(old_name, old_name)
                ni = names.index(name)
                different = int(np.sum(arrays['control_code'][ni] != old_codes[fc-4, oi]))
                if different == 0:
                    np.testing.assert_allclose(arrays['metrics'][ni, :10], old_metrics[fc-4, oi, :10],
                                               rtol=2e-10, atol=1e-28)
                changes.append(dict(carrier=fc, method=name, changed_codes_vs_cached_batch=different,
                    bit_error_difference=float(arrays['metrics'][ni, 0]-old_metrics[fc-4, oi, 0])))
    result = dict(status='passed', at=now(), cases=3*len(names),
        explicit_refinement_path_checks=checks, feedback_trace_rows=trace_count,
        prior_batch_comparison=changes, identity_sha256=fingerprint(identity),
        source_sha256=identity['source_sha256'], runtime_manifest_sha256=identity['runtime_manifest_sha256'])
    write_json(output/'preflight_checks.json', result)
    return result


def run(args):
    require_host(); project=args.project.resolve(); output=args.output.resolve()
    identity=identity_for(project, args.preflight)
    if not args.preflight:
        if args.preflight_evidence is None:
            raise ValueError('完整评测需要本轮前置检查。')
        evidence=read(args.preflight_evidence)
        if (evidence['status'] != 'passed' or evidence['cases'] != 129
                or evidence['source_sha256'] != identity['source_sha256']
                or evidence['runtime_manifest_sha256'] != identity['runtime_manifest_sha256']):
            raise ValueError('前置检查与本次代码/模型不同。')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ['records', 'workers', 'attempts']:
            (output/name).mkdir(exist_ok=True)
        protocol=output/'protocol.json'
        if protocol.exists():
            if read(protocol) != identity:
                raise ValueError('续跑身份不同，保留已有结果。')
        else:
            write_json(protocol, identity)
            for name in identity['source_sha256']:
                dest=output/'source_snapshot'/name
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE/name, dest)
        completed=[]; pending=[]
        for row in identity['rows']:
            old=batch.committed_environment(output, row, identity)
            if old is None: pending.append(row)
            else: completed.append(old)
        selected=pending[:args.stop_after] if args.stop_after else pending
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        mem={line.split(':')[0]:line.split(':')[1] for line in Path('/proc/meminfo').read_text().splitlines()}
        if selected and (free < args.workers*3072 or int(mem['MemAvailable'].split()[0]) < (args.workers*2048+4096)*1024):
            raise RuntimeError('显存或主存不足，保留断点并停止。')
        attempt=output/'attempts'/('%d_%d.json'%(time.time_ns(),os.getpid()))
        state=dict(status='running', pid=os.getpid(), workers=args.workers, at=now(),
            completed=len(completed), total=len(identity['rows']), reused_environments=len(completed),
            scope=identity['scope'], final_confirmation=False, identity_sha256=fingerprint(identity))
        write_json(attempt,state); write_json(output/'progress.json',state)
        if selected:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(str(output),identity)) as pool:
                jobs=[pool.submit(batch.one,row) for row in selected]
                for job in as_completed(jobs):
                    completed.append(job.result())
                    state.update(completed=len(completed),at=now())
                    write_json(output/'progress.json',state); print(json.dumps(state),flush=True)
        verify_sources(identity['source_sha256']); verify(Path(identity['runtime_bundle']))
        complete=len(completed)==len(identity['rows'])
        state.update(status='complete' if complete else 'stopped_at_requested_environment_boundary', at=now())
        write_json(attempt,state); write_json(output/'progress.json',state)
        write_json(output/'records.json',sorted(completed,key=lambda r:r['index']))
        if complete:
            write_json(output/'complete.json',dict(status='complete_exploratory_reception',at=now(),
                environments=len(completed),carriers=len(identity['carriers']),methods=len(identity['methods']),
                cases=len(completed)*len(identity['carriers'])*len(identity['methods']),
                identity_sha256=fingerprint(identity),records_sha256=sha256(output/'records.json'),
                final_confirmation=False,scope=identity['scope']))
            if args.preflight: preflight_checks(output,identity)
            else:
                subprocess.run([sys.executable,'-u','study_fair_followup/analyze.py',
                    '--evaluation',str(output),'--output',str(output/'analysis')],cwd=SOURCE,check=True)
        print(json.dumps(state),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project','output']: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--stop-after',type=int,default=0)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--preflight-evidence',type=Path)
    args=parser.parse_args()
    if args.stop_after < 0: parser.error('stop-after必须非负')
    try: run(args)
    except BaseException:
        if args.output.exists():
            write_json(args.output/('failure_%d.json'%time.time()),dict(at=now(),traceback=traceback.format_exc()))
        raise
