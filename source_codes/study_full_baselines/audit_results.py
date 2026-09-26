"""独立重算全部接收汇总，检查身份、控制码、错误计数及跨流水线一致性。

部分审计只能报告已提交记录，不记为完整实验通过；不生成或修改预测。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *

PHASES = ['classic', 'learned_evaluation', 'evaluation_components',
          'evaluation_scaling_1728', 'evaluation_scaling_3456']
SUMMARY_FIELDS = ['ber', 'ser', 'block_error_rate', 'mean_nmse', 'rms_evm_percent',
                  'effective_snr_db', 'paired_output_snr_db', 'mean_optical_dc_w']


def read_phase(folder, rows, data_digest, partial=False):
    if not (folder/'protocol.json').exists():
        if partial:
            return dict(status='not_started', count=0, folder=folder.name), None
        raise ValueError('缺少实验协议：'+str(folder))
    protocol = json.loads((folder/'protocol.json').read_text())
    if protocol['data_manifest_sha256'] != data_digest:
        raise ValueError('方法没有使用同一份数据：'+folder.name)
    verify_sources(protocol['source_sha256'])
    for path, digest in protocol.get('bundle', {}).get('artifact_sha256', {}).items():
        if sha256(path) != digest:
            raise ValueError('评分使用的预测产物发生变化。')
    methods = protocol.get('methods') or [m['name'] for m in protocol['bundle']['methods']]
    metrics = protocol['metric_order']
    if metrics[:10] != QUALITY_METRICS or len(set(methods)) != len(methods):
        raise ValueError('指标顺序或方法命名不符合统一评价定义。')
    fp = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    # 只读取已原子提交的单环境标记。进行中的半个环境不参与审计。
    records = [json.loads(p.read_text()) for p in sorted((folder/'records').glob('environment_*.json'))]
    indices = [r['index'] for r in records]
    if len(set(indices)) != len(indices):
        raise ValueError('一个环境存在多个提交。')
    expected = {r['index']:r for r in rows}; arrays = []; controls = []; ids = []
    for record in records:
        index = record['index']
        if index not in expected or record['environment_id'] != expected[index]['environment_id']:
            raise ValueError('评分记录环境身份不同。')
        if record['fingerprint'] != fp:
            raise ValueError('评分指纹与协议不同。')
        path = folder/'records'/record['path']
        if path.name != 'environment_%05d.npz'%index or sha256(path) != record['sha256']:
            raise ValueError('评分记录路径或哈希不同。')
        with np.load(path) as f:
            a = f['metrics']; code = f['control_code']
            if str(f['environment_id']) != record['environment_id']:
                raise ValueError('数组内部的环境身份不符。')
        if a.shape != (17, len(methods), len(metrics)) or code.shape != (17, len(methods), 128):
            raise ValueError('方法/载频/指标覆盖不完整。')
        if not np.isfinite(a[..., :7]).all() or np.any(a[..., :7]<0):
            raise ValueError('接收错误计数或NMSE无效。')
        integer_fields = [0, 1, 3, 4, 5, 6]
        if not np.array_equal(a[..., integer_fields], np.rint(a[..., integer_fields])):
            raise ValueError('错误计数应为整数。')
        if not (np.all(a[..., 1]==496) and np.all(a[..., 4]==248) and np.all(a[..., 6]==8)):
            raise ValueError('评价比特、符号或帧数发生变化。')
        if (np.any(a[..., 0]>a[..., 1]) or np.any(a[..., 3]>a[..., 4])
            or np.any(a[..., 5]>a[..., 6]) or np.any(a[..., 0]<a[..., 3])
            or np.any(a[..., 0]>2*a[..., 3]) or np.any(a[..., 5]>a[..., 3])):
            raise ValueError('BER/SER/块错误计数关系不成立。')
        for mi, method in enumerate(methods):
            if method=='mrc':
                if not np.all(code[:, mi]==-1) or not np.isnan(a[:, mi, 7:10]).all():
                    raise ValueError('数字MRC被错误地当作光子硬件控制。')
                continue
            if not np.isfinite(a[:, mi, 7:10]).all() or np.any(a[:, mi, 7:10]<0) or np.any(a[:, mi, 8]<=0):
                raise ValueError('光子接收功率字段无效。')
            if (not np.array_equal(code[:, mi], np.rint(code[:, mi])) or np.any(code[:, mi]<0)
                    or np.any(code[:, mi]>LEVELS)):
                raise ValueError('非法器件控制档位。')
            if method!='teacher':
                calls = a[:, mi, metrics.index('feedback_calls')]
                if not np.isfinite(calls).all() or np.any(calls<0) or np.any(calls>64):
                    raise ValueError('反馈次数不符合预算。')
                if not np.array_equal(calls, np.rint(calls)):
                    raise ValueError('反馈次数必须是整数。')
        arrays.append(a); controls.append(code); ids.append(record['environment_id'])
    progress = json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {}
    complete = (progress.get('status')=='complete' and set(indices)==set(expected))
    if not partial and not complete:
        raise ValueError('该实验尚未完整完成：'+folder.name)
    if complete:
        stored = json.loads((folder/'records.json').read_text())
        if sorted(stored, key=lambda r:r['index']) != sorted(records, key=lambda r:r['index']):
            raise ValueError('完整记录索引与逐环境提交不同。')
    info = dict(status='complete' if complete else 'partial', count=len(records), expected=len(rows),
        folder=folder.name, methods=methods, metric_order=metrics, protocol_sha256=sha256(folder/'protocol.json'),
        record_sha256={r['path']:r['sha256'] for r in records})
    if not records:
        return info, None
    return info, dict(values=np.asarray(arrays), controls=np.asarray(controls), indices=indices,
        ids=ids, methods=methods, metric_order=metrics)


def statistics(a, method):
    """从原始计数与功率独立聚合，不调用原评测器的summarize。"""
    result = dict(ber=float(a[..., 0].sum()/a[..., 1].sum()),
        ser=float(a[..., 3].sum()/a[..., 4].sum()),
        block_error_rate=float(a[..., 5].sum()/a[..., 6].sum()),
        mean_nmse=float(a[..., 2].mean()))
    result['rms_evm_percent'] = 100*result['mean_nmse']**.5
    result['effective_snr_db'] = float(-10*np.log10(max(result['mean_nmse'], 1e-30)))
    if method!='mrc':
        result['paired_output_snr_db'] = float(10*np.log10(a[..., 7].sum()/a[..., 8].sum()))
        result['mean_optical_dc_w'] = float(a[..., 9].mean())
    return result


def check_summary(folder, arrays, rows):
    stored = json.loads((folder/'summary.json').read_text())
    if stored['status']!='complete':
        raise ValueError('汇总未完成。')
    lookup = {(r['method'], r['group'], r['value']):r for r in stored['records']}
    if len(lookup)!=len(stored['records']):
        raise ValueError('汇总存在重复方法或分组。')
    groups = [('all', None, np.ones(len(rows), bool), None)]
    groups += [('carrier_ghz', fc, np.ones(len(rows), bool), fc-4) for fc in range(4, 21)]
    for factor in ['rays', 'max_delay_ns', 'angular_std_deg', 'power_bin']:
        for value in sorted({r['factors'][factor] for r in rows}):
            groups.append((factor, value, np.asarray([r['factors'][factor]==value for r in rows]), None))
    expected_keys = set(); checked = 0; difference = 0.
    for mi, method in enumerate(arrays['methods']):
        for group, value, mask, ci in groups:
            key = (method, group, value); expected_keys.add(key)
            if key not in lookup:
                raise ValueError('缺少汇总分组。')
            a = arrays['values'][mask, :, mi]
            if ci is not None:
                a = a[:, ci:ci+1]
            recalculated = statistics(a, method); reference = lookup[key]
            if reference['environments']!=int(mask.sum()):
                raise ValueError('汇总环境数不同。')
            for field, current in recalculated.items():
                if field not in reference or not np.isclose(current, reference[field], rtol=1e-12, atol=1e-12):
                    raise ValueError('汇总数字不能从原始数据复现：'+str(key)+' '+field)
                difference = max(difference, abs(current-reference[field])); checked += 1
    if set(lookup)!=expected_keys:
        raise ValueError('存在未声明的汇总分组。')
    return dict(numbers_recomputed=checked, maximum_absolute_difference=difference,
        summary_sha256=sha256(folder/'summary.json'))


def cross_check(phases, left, right):
    lf, lm = left; rf, rm = right
    if lf not in phases or rf not in phases or phases[lf] is None or phases[rf] is None:
        return dict(left=left, right=right, status='waiting', environments=0)
    a, b = phases[lf], phases[rf]; ai = a['methods'].index(lm); bi = b['methods'].index(rm)
    common = sorted(set(a['indices']) & set(b['indices']))
    for index in common:
        ix = a['indices'].index(index); jx = b['indices'].index(index)
        if a['ids'][ix]!=b['ids'][jx]:
            raise ValueError('跨实验环境身份不一致。')
        if not np.array_equal(a['controls'][ix, :, ai], b['controls'][jx, :, bi]):
            raise ValueError('同一控制器跨流水线得到不同档位：'+lm+' / '+rm)
        if not np.allclose(a['values'][ix, :, ai, :10], b['values'][jx, :, bi, :10],
                           rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError('相同控制的接收结果跨流水线不同。')
    return dict(left=left, right=right, status='matched' if common else 'waiting',
        environments=len(common), checked_control_codes=len(common)*17*128,
        checked_quality_values=len(common)*17*10)


def run(data, study, output, partial):
    require_host(); manifest = check_data(data)
    rows = [r for r in manifest['environments'] if r['split']=='test']
    if [r['index'] for r in rows]!=list(range(len(rows))):
        raise ValueError('测试环境顺序不连续。')
    reports = {}; arrays = {}; digest = sha256(data/'manifest.json')
    for name in PHASES:
        info, values = read_phase(study/name, rows, digest, partial)
        if info['status']=='complete':
            info.update(check_summary(study/name, values, rows))
        reports[name] = info; arrays[name] = values
    crosses = [cross_check(arrays, left, right) for left, right in [
        (('classic','ttd_das'), ('learned_evaluation','public16')),
        (('learned_evaluation','covariance_response'), ('evaluation_components','covariance__base_2sweeps')),
        (('learned_evaluation','complex_response_cnn'), ('evaluation_components','cnn__base_2sweeps'))]]
    complete = all(v['status']=='complete' for v in reports.values()) and all(
        c['status']=='matched' and c['environments']==len(rows) for c in crosses)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(status='passed' if complete else 'partial_not_complete',
        data_sha256=digest, phases=reports, cross_pipeline_checks=crosses,
        all_required_phases_complete=complete, scientific_scope='old216 exploratory reception audit',
        model_training_and_online_latency_audits_separate=True,
        source_sha256=source_record(['study_full_baselines/audit_results.py']), at=now()))
    print(json.dumps(dict(status='passed' if complete else 'partial_not_complete',
        counts={k:v['count'] for k,v in reports.items()}, cross_checks=crosses)), flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'study', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true'); a = p.parse_args()
    run(a.data, a.study, a.output, a.allow_partial)
