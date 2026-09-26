"""只用已知旧案例重建完整确认链路，并核对CPU/GPU接收后端。不会生成新留出信号。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.online_controller import OnlineController, method_names
from study_full_baselines.confirmation_receiver import public_observation, decide_all, score_decisions, METRIC_ORDER
from study_full_baselines.expand_training_gpu import validate_backend
from study_full_baselines.check_online_controller import references
from our_method_response_control.train import precision
from our_method_quality_rank.generate import require_checks


def run(project, output):
    require_host(); precision()
    if output.exists(): raise FileExistsError('不覆盖已有确认接收器核查。')
    if not torch.cuda.is_available(): raise RuntimeError('要求华硕GPU。')
    root = project/'dataset_simulation'; data = root/'outputs/quality_rank_hybrid_20260925'
    study = root/'baseline_results/20260925_full_baselines'
    # 验证新计划的文件身份，但不调用它的任何环境种子或传播生成器。
    fresh_plan = root/'ops/full_baselines_20260925/fresh_confirmation_environment_plan.json'
    fresh_digest = sha256(fresh_plan)
    if fresh_digest != '06c44a38ad6ac36a213671181574d415814cee6c7f26a5cb5885a94e3c3b8378':
        raise ValueError('尚未解封的新环境计划发生变化。')
    backend_proof = validate_backend(root/'diagnostics/20260926_gpu_generation_replay')
    numerical = require_checks(project)
    manifest = check_data(data)
    row = next(r for r in manifest['environments'] if r['split'] == 'test' and r['index'] == 0)
    folder = data/row['path']; env = json.loads((folder/'environment.json').read_text())
    committed = next(r for r in json.loads((data/'records.json').read_text()) if r['environment_id'] == row['environment_id'])
    for name, digest in committed['file_sha256'].items():
        if sha256(folder/name) != digest: raise ValueError('旧案例文件改变。')
    with np.load(folder/'data.npz') as f: cached_x = f['X'].copy()
    with np.load(data/'public.npz') as f:
        public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    prior = root/'diagnostics/20260926_online_controller_preflight'
    prior_summary = json.loads((prior/'summary.json').read_text())
    single = {(r['method'], r['carrier_ghz']): r for r in prior_summary['cases']}
    classic, reference_files = references(study, row)
    names = method_names([216, 432])
    source_names = list(manifest['source_sha256'])
    source_names += ['study_full_baselines/'+n for n in ['confirmation_receiver.py',
        'check_confirmation_receiver.py', 'CONFIRMATION_RECEIVER_PROTOCOL.md',
        'online_controller.py', 'common.py', 'evaluate_classic.py', 'check_online_controller.py',
        'expand_training_gpu.py']]+['diagnostics/gpu_fft_hybrid.py']
    for group in ['deep_common', 'baseline_common', 'baseline_mlp', 'our_method_quality_rank',
        'our_method_response_control', 'baseline_response_realcnn', 'baseline_joint_response_linear',
        'our_method_two_stage', 'our_method_feedback_candidates']+['baseline_'+n for n in ONLINE_CLASSIC+DEEP_METHODS]:
        source_names += [str(f.relative_to(SOURCE)) for f in (SOURCE/group).glob('*.py')]
    sources = source_record(source_names)
    output.mkdir(parents=True); (output/'records').mkdir()
    for name in sources:
        dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    write_json(output/'protocol.json', dict(methods=names+['teacher', 'mrc'], environment=row, carriers=[4, 12, 20],
        source_sha256=sources, data_manifest_sha256=sha256(data/'manifest.json'), input_sha256=committed['file_sha256'],
        backend_proof=backend_proof, numerical_checks=numerical, public_sha256=sha256(data/'public.npz'),
        cpu_gpu_metric_relative_tolerance=1e-9, cpu_gpu_metric_absolute_tolerance=0.,
        bit_symbol_block_counts_exact=True, fresh_plan_sha256=fresh_digest, new_environment_signals_generated=0,
        metric_order=METRIC_ORDER, final_confirmation=False, at=now()))
    old_load = np.load; reads = []
    def guarded_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            path = Path(path)
            if path.name in ['data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy']:
                raise RuntimeError('控制器尝试读取隐藏数据或缓存预测。')
            reads.append(str(path))
        return old_load(path, *args, **kwargs)
    with patch('numpy.load', guarded_load): models = {name: OnlineController(project, name, public) for name in names}
    details = []; files = {}; before = time.perf_counter()
    for fc in [4, 12, 20]:
        raw, observed = public_observation(env, fc, public, 'cuda_fft_cpu_rk4_v1')
        # 独立重建输入，不从缓存X进入本次控制流程；缓存仅在此处做相等核对。
        np.testing.assert_array_equal(raw, cached_x[fc-4])
        with patch('numpy.load', guarded_load):
            decisions = decide_all(models, raw, observed, row['seed'], fc, public)
        # 到此全部控制已固定，才创建独立评分帧。
        gpu = score_decisions(env, fc, public, decisions, 'cuda_fft_cpu_rk4_v1')
        cpu = score_decisions(env, fc, public, decisions, 'cpu')
        np.testing.assert_array_equal(gpu['control_code'], cpu['control_code'])
        np.testing.assert_array_equal(gpu['metrics'][:, [0, 1, 3, 4, 5, 6]], cpu['metrics'][:, [0, 1, 3, 4, 5, 6]])
        np.testing.assert_allclose(gpu['metrics'][:, :10], cpu['metrics'][:, :10], rtol=1e-9, atol=0., equal_nan=True)
        arrays = dict(public_X=raw, control_code=gpu['control_code'], metrics_gpu=gpu['metrics'], metrics_cpu=cpu['metrics'],
            teacher_objective_evaluations=np.asarray(gpu['teacher_objective_evaluations']),
            mrc_physical_reference_snr_db=np.asarray(gpu['mrc_physical_reference_snr_db']))
        for mi, name in enumerate(gpu['methods']):
            if name in ['teacher', 'mrc']:
                refcode = classic[name]['control'][fc-4]; refquality = classic[name]['quality'][fc-4]
            else:
                ref = single[name, fc]; refpath = prior/ref['result_file']
                if sha256(refpath) != ref['result_sha256']: raise ValueError('旧在线核对参考改变。')
                files[str(refpath)] = ref['result_sha256']
                with np.load(refpath) as f:
                    refcode = f['control_code'][1].copy(); refquality = f['quality'][1].copy()
                    if 'trace_control_code' in decisions['feedback'][name]:
                        np.testing.assert_array_equal(decisions['feedback'][name]['trace_control_code'], f['trace_control_code'])
                        np.testing.assert_allclose(decisions['feedback'][name]['trace_scores'],
                                                   f['trace_scores'], rtol=1e-9, atol=0.)
            np.testing.assert_array_equal(gpu['control_code'][mi], refcode)
            np.testing.assert_allclose(gpu['metrics'][mi, :10], refquality, rtol=1e-9, atol=0., equal_nan=True)
            for field, values in decisions['feedback'].get(name, {}).items(): arrays[name+'__'+field] = values
            finite = np.isfinite(refquality)
            error = np.max(abs(gpu['metrics'][mi, :10][finite]-refquality[finite])/
                           np.maximum(abs(refquality[finite]), np.finfo(float).tiny))
            details.append(dict(method=name, carrier_ghz=fc, controls_equal=True,
                quality_relative_difference_max=float(error), cpu_gpu_counts_equal=True))
        path = output/'records'/('carrier_%02d.npz' % fc); atomic_npz(path, **arrays)
        files[str(path)] = sha256(path)
        write_json(output/'progress.json', dict(status='running', completed_carriers=fc//8+1,
            carrier=fc, cases=len(details), total_cases=(len(names)+2)*3, pid=os.getpid(), at=now()))
        print(json.dumps(dict(carrier=fc, methods=len(gpu['methods']), public_input_bitwise_equal=True,
            same_controls_and_quality_as_previous_receiver=True, cpu_gpu_quality_gate_passed=True)), flush=True)
    for model in models.values(): model.verify()
    verify_sources(sources)
    if sha256(fresh_plan) != fresh_digest: raise ValueError('新计划在核查期间改变。')
    write_json(output/'summary.json', dict(status='passed_confirmation_receiver_preflight', cases=details,
        cases_count=len(details), methods=len(names)+2, new_environment_signals_generated=0,
        all_inputs_bitwise_equal=True, all_controls_equal=True, cpu_gpu_scoring_passed=True,
        original_quality_reproduced=True, source_sha256=sources, reference_files=reference_files,
        result_and_reference_sha256=files, model_artifacts={name: model.artifacts for name, model in models.items()},
        controller_numeric_reads=sorted(set(reads)), final_confirmation=False,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.perf_counter()-before, at=now()))
    write_json(output/'progress.json', dict(status='complete', methods=len(names)+2, cases=len(details), at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try: run(args.project.resolve(), args.output.resolve())
    except BaseException:
        if args.output.exists(): write_json(args.output/('failure_%d.json' % time.time()),
                                           dict(traceback=traceback.format_exc(), at=now()))
        raise
