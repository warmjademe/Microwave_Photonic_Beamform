"""从逐次耗时、控制、来源文件独立核对在线计时产物；不依据status判定通过。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, LEVELS, sha256, write_json, now, NativeConfig


def audit(project, folder):
    require_host()
    protocol = json.loads((folder/'protocol.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    data = project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    if summary['protocol_sha256'] != sha256(folder/'protocol.json'):
        raise ValueError('计时协议改变。')
    if protocol['data_manifest_sha256'] != sha256(data/'manifest.json'):
        raise ValueError('计时数据来源改变。')
    if protocol['public_sha256'] != sha256(data/'public.npz'): raise ValueError('公开配置改变。')
    if protocol['cpu_threads'] != 1 or set(protocol['cpu_thread_environment'].values()) != {'1'}:
        raise ValueError('CPU线程未统一固定。')
    if protocol['repetitions'] != 3 or protocol['batch_size'] != 1: raise ValueError('计时批量或重复数不符。')
    if summary['final_fair_timing'] != protocol['final_fair_timing']: raise ValueError('正式标记不一致。')
    manifest = json.loads((data/'manifest.json').read_text())
    test = sorted([r for r in manifest['environments'] if r['split'] == 'test'], key=lambda r: r['index'])
    expected_rows = [test[0]] if protocol['preflight'] else [next(r for r in test if r['factors']['power_bin'] == p) for p in range(6)]
    if protocol['rows'] != expected_rows: raise ValueError('选样不是预先约定的固定成员。')
    carriers = [4, 12, 20] if protocol['preflight'] else list(range(4, 21))
    if protocol['carriers'] != carriers: raise ValueError('频率覆盖不符。')
    for row in expected_rows:
        for name, digest in protocol['input_file_sha256'][row['environment_id']].items():
            if sha256(data/row['path']/name) != digest: raise ValueError('单环境输入改变。')
    for name, digest in protocol['source_sha256'].items():
        if sha256(folder/'source_snapshot'/name) != digest: raise ValueError('源码快照改变。')
        if sha256(project/'source_codes'/name) != digest: raise ValueError('当前代码已不同于计时版本。')
    if any(Path(p).name in ['data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy']
           for p in summary['numerical_artifacts_loaded']): raise ValueError('在线阶段读入了禁止文件。')
    expected = {(m, r['index'], fc) for m in protocol['methods'] for r in expected_rows for fc in carriers}
    observed = {(r['method'], r['index'], r['carrier_ghz']) for r in summary['cases']}
    if observed != expected or len(summary['cases']) != len(expected): raise ValueError('缺失或重复计时案例。')
    if summary['cases_count'] != len(expected): raise ValueError('案例计数不符。')
    preflight_reference = project/'dataset_simulation/diagnostics/20260926_online_controller_preflight'
    references = {}; hashes = {}
    if protocol['preflight']:
        previous = json.loads((preflight_reference/'summary.json').read_text())
        references = {(r['method'], r['carrier_ghz']): r for r in previous['cases']}
    times = {}; reference_count = 0; trace_count = 0; repetition_count = 0
    for record in summary['cases']:
        path = folder/record['file']
        if sha256(path) != record['sha256']: raise ValueError('逐次计时文件改变。')
        with np.load(path) as f: arrays = {k: f[k].copy() for k in f.files}
        code = arrays['control_code']
        if code.shape != (128,) or not np.isfinite(code).all() or np.any(code != np.rint(code)) or np.any(code < 0) or np.any(code > LEVELS):
            raise ValueError('器件码非法。')
        for field in ['software_seconds', 'replay_callback_seconds', 'total_wall_seconds']:
            values = arrays[field]
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('逐次时间非法。')
        np.testing.assert_allclose(arrays['software_seconds']+arrays['replay_callback_seconds'],
                                   arrays['total_wall_seconds'], rtol=1e-12, atol=1e-12)
        if not record['repeated_controls_and_all_queries_identical']: raise ValueError('在线重放不一致。')
        if record['measurement_calls'] == 64:
            if arrays['query_control_code'].shape != (48, 128) or arrays['query_scores'].shape != (48,):
                raise ValueError('追加测量轨迹数量不符。')
            if not np.isfinite(arrays['query_scores']).all(): raise ValueError('测量值非有限。')
            trace_count += 48
        elif record['measurement_calls'] not in [0, 16]: raise ValueError('未知测量预算。')
        if references:
            reference = references[record['method'], record['carrier_ghz']]
            source = preflight_reference/reference['result_file']
            if sha256(source) != reference['result_sha256']: raise ValueError('独立接收核查参考改变。')
            hashes[str(source)] = reference['result_sha256']
            with np.load(source) as f:
                np.testing.assert_array_equal(code, f['control_code'][1])
                if record['measurement_calls'] == 64:
                    np.testing.assert_array_equal(arrays['query_control_code'], f['trace_control_code'][16:])
                    np.testing.assert_array_equal(arrays['query_scores'], f['trace_scores'][16:])
            reference_count += 1
        times.setdefault(record['method'], []).extend(arrays['software_seconds'].tolist())
        repetition_count += 3
    method_names = [r['method'] for r in summary['methods']]
    if method_names != protocol['methods']: raise ValueError('方法汇总有缺失或顺序不同。')
    for model in summary['methods']:
        values = np.asarray(times[model['method']])*1000
        expected_values = [values.mean(), np.median(values), np.quantile(values, .95)]
        observed_values = [model[f] for f in ['mean_software_ms', 'median_software_ms', 'p95_software_ms']]
        np.testing.assert_allclose(observed_values, expected_values, rtol=1e-12, atol=1e-12)
        for path, digest in model['model_artifacts'].items():
            if sha256(path) != digest: raise ValueError('模型权重或统计参数改变。')
        nominal = (model['measurement_calls']*NativeConfig().measurement_s+NativeConfig().switch_s)*1000
        np.testing.assert_allclose(model['estimated_measurement_switch_ms'], nominal, rtol=1e-12)
        np.testing.assert_allclose(model['estimated_total_mean_ms'], values.mean()+nominal, rtol=1e-12)
        if summary['final_fair_timing']:
            if model['contention_reasons']: raise ValueError('正式计时存在已知竞争。')
            for snapshot in [model['load_before'], model['load_after']]:
                if snapshot['project_numeric_processes'] or snapshot['cpu_busy_fraction'] > .6:
                    raise ValueError('正式计时CPU资源门限不符。')
                if len(snapshot['gpu_processes']) > 1: raise ValueError('GPU存在其他进程。')
    report = dict(status='passed_artifact_and_reference_audit', cases=len(expected),
        timing_repetitions_checked=repetition_count, additional_query_rows_checked=trace_count,
        controls_matched_to_independent_receiver_preflight=reference_count, reference_file_sha256=hashes,
        final_fair_timing=summary['final_fair_timing'], summary_sha256=sha256(folder/'summary.json'),
        auditor_sha256=sha256(Path(__file__)), at=now())
    dest = folder/'integrity_verification.json'
    if dest.exists(): raise FileExistsError('不覆盖已有审计。')
    write_json(dest, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'reference_file_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path); parser.add_argument('--input', required=True, type=Path)
    args = parser.parse_args(); audit(args.project.resolve(), args.input.resolve())
