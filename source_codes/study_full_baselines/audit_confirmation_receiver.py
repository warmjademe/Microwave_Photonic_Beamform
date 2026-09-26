"""逐文件重算旧案例确认链路核查，验证来源、预算、控制和接收计数。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, write_json, now, LEVELS, ONLINE_CLASSIC


def run(project, folder):
    require_host()
    dest = folder/'integrity_verification.json'
    if dest.exists(): raise FileExistsError('不覆盖原核查。')
    protocol = json.loads((folder/'protocol.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    root = project/'dataset_simulation'; data = root/'outputs/quality_rank_hybrid_20260925'
    row = protocol['environment']; names = protocol['methods']
    if row['index'] != 0 or row['split'] != 'test' or protocol['carriers'] != [4, 12, 20]:
        raise ValueError('不是固定旧案例前置核查。')
    if summary['final_confirmation'] or protocol['final_confirmation'] or summary['new_environment_signals_generated'] != 0:
        raise ValueError('前置核查被误标为新留出结果。')
    if len(names) != 45 or len(set(names)) != 45 or len(summary['cases']) != 135:
        raise ValueError('核查方法或案例缺失。')
    for name, digest in protocol['source_sha256'].items():
        if sha256(folder/'source_snapshot'/name) != digest or sha256(project/'source_codes'/name) != digest:
            raise ValueError('源码或快照改变。')
    for name, digest in protocol['input_sha256'].items():
        if sha256(data/row['path']/name) != digest: raise ValueError('旧案例输入改变。')
    for path, digest in summary['result_and_reference_sha256'].items():
        if sha256(path) != digest: raise ValueError('接收结果或单条参考改变。')
    for path, digest in summary['reference_files'].items():
        if sha256(path) != digest: raise ValueError('教师/MRC参考文件改变。')
    for artifacts in summary['model_artifacts'].values():
        for path, digest in artifacts.items():
            if sha256(path) != digest: raise ValueError('模型或训练统计参数改变。')
    plan = root/'ops/full_baselines_20260925/fresh_confirmation_environment_plan.json'
    if sha256(plan) != protocol['fresh_plan_sha256']: raise ValueError('新留出清单改变。')
    with np.load(data/row['path']/'data.npz') as f: original_x = f['X'].copy()
    trace_rows = 0; max_relative = 0.; cases = 0
    prior = root/'diagnostics/20260926_online_controller_preflight'
    old = json.loads((prior/'summary.json').read_text())
    index = {(r['method'], r['carrier_ghz']): r for r in old['cases']}
    classic_dir = root/'baseline_results/20260925_full_baselines/classic'
    classic_names = json.loads((classic_dir/'protocol.json').read_text())['methods']
    with np.load(classic_dir/'records/environment_00000.npz') as f:
        classic_codes = f['control_code'].copy(); classic_quality = f['metrics'][:, :, :10].copy()
    for fc in protocol['carriers']:
        with np.load(folder/'records'/('carrier_%02d.npz' % fc)) as f: arrays = {k: f[k].copy() for k in f.files}
        np.testing.assert_array_equal(arrays['public_X'], original_x[fc-4])
        code = arrays['control_code']; gpu = arrays['metrics_gpu']; cpu = arrays['metrics_cpu']
        if code.shape != (45, 128) or gpu.shape != (45, 13) or cpu.shape != gpu.shape:
            raise ValueError('接收数组形状错误。')
        if np.any(code[:-1] < 0) or np.any(code[:-1] > LEVELS) or np.any(code[:-1] != np.rint(code[:-1])):
            raise ValueError('光子控制码非法。')
        if names[-1] != 'mrc' or np.any(code[-1] != -1) or not np.isnan(gpu[-1, 7:]).all():
            raise ValueError('数字MRC被写成光子硬件结果。')
        np.testing.assert_array_equal(gpu[:, [1, 4, 6]], np.tile([496, 248, 8], (45, 1)))
        for error_col, total_col in [(0, 1), (3, 4), (5, 6)]:
            if np.any(gpu[:, error_col] < 0) or np.any(gpu[:, error_col] > gpu[:, total_col]):
                raise ValueError('错误计数范围错误。')
        np.testing.assert_array_equal(gpu[:, [0, 1, 3, 4, 5, 6]], cpu[:, [0, 1, 3, 4, 5, 6]])
        np.testing.assert_allclose(gpu[:, :10], cpu[:, :10], rtol=1e-9, atol=0., equal_nan=True)
        for mi, name in enumerate(names):
            if name in ['teacher', 'mrc']:
                j = classic_names.index(name)
                refcode = classic_codes[fc-4, j]; refquality = classic_quality[fc-4, j]
                if np.isfinite(gpu[mi, 10]): raise ValueError('额外信息参考不应伪装为普通预算。')
            else:
                ref = index[name, fc]
                with np.load(prior/ref['result_file']) as f:
                    refcode = f['control_code'][1].copy(); refquality = f['quality'][1].copy()
                    if 'trace_control_code' in f:
                        a = arrays[name+'__trace_control_code']; scores = arrays[name+'__trace_scores']
                        np.testing.assert_array_equal(a, f['trace_control_code'])
                        np.testing.assert_allclose(scores, f['trace_scores'], rtol=1e-9, atol=0.)
                        trace_rows += len(a)
                expected = 0 if name.endswith('_prior') else 64 if name in ONLINE_CLASSIC[1:] or name.endswith('_warm64') else 16
                if gpu[mi, 10] != expected: raise ValueError('在线预算不同。')
            np.testing.assert_array_equal(code[mi], refcode)
            np.testing.assert_allclose(gpu[mi, :10], refquality, rtol=1e-9, atol=0., equal_nan=True)
            finite = np.isfinite(refquality)
            delta = abs(gpu[mi, :10][finite]-refquality[finite])/np.maximum(abs(refquality[finite]), np.finfo(float).tiny)
            max_relative = max(max_relative, float(delta.max())); cases += 1
    if any(Path(p).name in ['data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy']
           for p in summary['controller_numeric_reads']): raise ValueError('在线控制读取隐藏数据。')
    result = dict(status='passed_artifact_controls_and_quality_audit', cases=cases, methods=len(names),
        metric_fields_per_case=10, exact_integer_metric_fields_per_case=6, feedback_trace_rows_checked=trace_rows,
        maximum_relative_quality_difference=max_relative, final_confirmation=False, new_environment_signals_generated=0,
        summary_sha256=sha256(folder/'summary.json'), auditor_sha256=sha256(Path(__file__)), at=now())
    write_json(dest, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); run(args.project.resolve(), args.input.resolve())
