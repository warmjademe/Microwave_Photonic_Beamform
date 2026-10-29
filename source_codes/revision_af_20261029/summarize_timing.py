"""从完整调用边界汇总正式时延，保留内部计时和回放开销的区别。"""
import argparse
import json
import os
from pathlib import Path
import sys
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host, sha256, write_json, now


def read(path):
    return json.loads(path.read_text())


def run(project):
    require_host()
    source = project / 'dataset_simulation/revision_af_20261029/P/timing_full'
    output = source.parent / 'timing_analysis'
    assert not output.exists(), output
    proof = read(source / 'complete.json')
    assert proof['status'] == 'passed' and proof['formal_timing'] and not proof['preflight']
    for name, digest in proof['file_sha256'].items():
        assert sha256(source / name) == digest
    protocol = read(source / 'protocol.json')
    records = read(source / 'records.json')
    assert len(records) == proof['cases'] == 408
    assert protocol['repetitions'] == 3 and protocol['warmup'] == 2
    methods = protocol['methods']
    collected = {name: {} for name in methods}
    seen = set()
    for record in records:
        key = (record['method'], record['index'], record['carrier'])
        assert key not in seen
        seen.add(key)
        path = source / 'records' / record['file']
        assert sha256(path) == record['sha256']
        with np.load(path, allow_pickle=False) as values:
            assert values['control_code'].shape == (128,)
            for kind in ['plain', 'segmented']:
                names = values[kind + '_fields'].tolist()
                matrix = values[kind]
                assert matrix.shape == (3, len(names)) and np.isfinite(matrix).all()
                fields = {name: matrix[:, i] for i, name in enumerate(names)}
                total = fields['wall'] - fields['callback']
                extra = total - fields['software']
                assert (total >= 0).all() and (extra >= -1e-8).all()
                fields['total_software_call'] = total
                fields['outside_internal_timer'] = extra
                if kind == 'segmented':
                    other = total - fields['response_estimation'] - fields['discrete_solver']
                    assert (other >= -1e-8).all()
                    assert (fields['response_estimation'] >= fields['gpu_network_forward'] - 1e-8).all()
                    fields['other_control_full_boundary'] = other
                for name, data in fields.items():
                    collected[record['method']].setdefault(kind + '__' + name, []).extend(data.tolist())
    expected = {(method, row['index'], carrier) for method in methods
                for row in protocol['rows'] for carrier in protocol['carriers']}
    assert seen == expected
    output.mkdir()
    rows = []
    for method in methods:
        def stats(values):
            data = np.asarray(values) * 1000
            assert len(data) == 306
            return dict(mean_ms=float(data.mean()), median_ms=float(np.median(data)),
                        p95_ms=float(np.quantile(data, .95)), repetitions=len(data))
        components = {name: stats(data) for name, data in collected[method].items()}
        budget = 16 if method == 'complex_response_cnn' else 64
        assumptions = protocol['assumptions']
        measurement = 1000 * (budget * (assumptions['pilot_acquisition_s'] + assumptions['switch_s'])
                              + assumptions['final_apply_switches'] * assumptions['switch_s'])
        rows.append(dict(method=method, budget=budget, inputs=102, components=components,
                         assumed_measurement_ms=measurement,
                         mean_software_plus_assumed_measurement_ms=components['plain__total_software_call']['mean_ms'] + measurement))
    write_json(output / 'summary.json', dict(rows=rows,
        boundary='Entire controller call including result packaging, minus exact-query replay callback; assertions are outside the timer.',
        instrumented_diagnostics='Separate executions; network forward is nested in response estimation.',
        hardware_scope='Acquisition and switch times are assumptions; communication and extra settling are not measured.'))
    write_json(output / 'protocol.json', dict(source_complete_sha256=sha256(source / 'complete.json'),
        source_file_sha256=proof['file_sha256'], code_sha256=sha256(Path(__file__)),
        changes_to_execution=False, new_timing_samples=False,
        reason='Internal controller software_seconds ends before all return-value packaging. '
               'Use the already recorded outer wall-minus-callback for total call latency. '
               'Compute medians and percentiles from per-repetition differences, not differences of summaries.'))
    lines = ['# 正式在线时间：完整控制调用边界', '',
        '原计时记录不改动；完整软件调用等于外层墙钟减测量回放回调。包含结果整理，不包含计时区间外的正确性断言。',
        '102 条预定输入，每条预热两次，正式三次；另三次分段诊断。单位均为毫秒。', '',
        '| 方法 | 预算 | 完整软件均值 | 中位数 | P95 | 假设采集与切换 | 均值加假设测量 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        s = row['components']['plain__total_software_call']
        lines.append('| ' + row['method'] + ' | ' + str(row['budget']) + ' | ' +
                     ' | '.join(f'{x:.6f}' for x in [s['mean_ms'], s['median_ms'], s['p95_ms'],
                     row['assumed_measurement_ms'], row['mean_software_plus_assumed_measurement_ms']]) + ' |')
    lines += ['', '## A+F 分段诊断均值', '']
    af = next(row for row in rows if row['method'] == 'cnn_warm64')
    for name in ['response_estimation', 'gpu_network_forward', 'response_other', 'discrete_solver',
                 'other_control_full_boundary', 'total_software_call', 'outside_internal_timer']:
        lines.append('- ' + name + ': ' + f"{af['components']['segmented__'+name]['mean_ms']:.6f}")
    lines += ['', '网络前向已经包含在响应估计中，不能再次相加。分段总和不替代无探针总时间；二者来源为不同重复执行。', '']
    (output / 'REPORT_zh.md').write_text('\n'.join(lines))
    for name, digest in proof['file_sha256'].items():
        assert sha256(source / name) == digest
    write_json(output / 'complete.json', dict(status='complete', at=now(), raw_records_verified=408,
        total_call_repetitions=1224, instrumented_repetitions=1224, new_timing_samples=False,
        file_sha256={name: sha256(output / name) for name in ['summary.json', 'protocol.json', 'REPORT_zh.md']}))
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    run(parser.parse_args().project.resolve())
