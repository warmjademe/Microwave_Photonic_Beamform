"""独立核查最终接收记录并执行冻结的分层配对统计；旧案例仅作演练。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (SOURCE, LEVELS, ONLINE_CLASSIC, require_host,
    sha256, source_record, verify_sources, write_json, now, atomic_npz)
from study_full_baselines.confirmation_freeze import validate as validate_freeze
from study_full_baselines.confirmation_execute import rehearsal_identity
from study_full_baselines.confirmation_batch import committed_environment, fingerprint
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.confirmation_receiver import METRIC_ORDER
from study_full_baselines.paired_statistics import FIELDS, sufficient, metrics, bootstrap, self_check
from study_full_baselines.analyze_reception import planned_comparisons, clean_json, write_csv
from study_full_baselines.audit_results import statistics as direct_statistics

SOURCES = ['study_full_baselines/'+name for name in [
    'analyze_confirmation.py', 'CONFIRMATION_ANALYSIS_PROTOCOL.md', 'confirmation_freeze.py',
    'confirmation_execute.py', 'confirmation_batch.py', 'runtime_bundle.py', 'confirmation_receiver.py',
    'paired_statistics.py', 'analyze_reception.py', 'audit_results.py', 'common.py']]


def read(path):
    return json.loads(path.read_text())


def check_values(arrays, carrier, identity, public):
    """直接检查每个载频的数字、预算和查询记录，不靠汇总status判断通过。"""
    names = identity['methods']
    raw, code, quality = arrays['public_X'], arrays['control_code'], arrays['metrics']
    if raw.shape != (2513,) or not np.isfinite(raw).all() or raw[1984] != carrier:
        raise ValueError('公开输入维度、数值或载频错误。')
    if code.shape != (len(names), 128) or quality.shape != (len(names), len(METRIC_ORDER)):
        raise ValueError('控制或指标方法覆盖不完整。')
    if not np.isfinite(code).all() or not np.array_equal(code, np.rint(code)):
        raise ValueError('器件控制码必须是有限整数。')
    if np.any(code[:-1] < 0) or np.any(code[:-1] > LEVELS) or np.any(code[-1] != -1):
        raise ValueError('光子档位或MRC参考标记错误。')
    if not np.isfinite(quality[:, :7]).all() or np.any(quality[:, :7] < 0):
        raise ValueError('错误计数或NMSE错误。')
    integer = quality[:, [0, 1, 3, 4, 5, 6]]
    if not np.array_equal(integer, np.rint(integer)):
        raise ValueError('比特、符号和块计数应为整数。')
    np.testing.assert_array_equal(quality[:, [1, 4, 6]], np.tile([496, 248, 8], (len(names), 1)))
    if (np.any(quality[:, 0] > quality[:, 1]) or np.any(quality[:, 3] > quality[:, 4])
            or np.any(quality[:, 5] > quality[:, 6]) or np.any(quality[:, 0] < quality[:, 3])
            or np.any(quality[:, 0] > 2*quality[:, 3]) or np.any(quality[:, 5] > quality[:, 3])):
        raise ValueError('比特、符号与块错误计数不相容。')
    power = quality[:-1, 7:10]
    if not np.isfinite(power).all() or np.any(power < 0) or np.any(power[:, 1] <= 0):
        raise ValueError('光子接收功率字段错误。')
    if not np.isnan(quality[-1, 7:]).all() or not np.isnan(quality[-2, 10]):
        raise ValueError('额外信息参考被错误标成普通光子控制器。')
    if not np.isfinite(quality[:-1, 11:]).all() or np.any(quality[:-1, 11:] < 0):
        raise ValueError('接收运行的计算或反馈时间无效。')
    if quality[-2, 12] != 0:
        raise ValueError('教师参考不使用普通反馈回调。')
    trace_names = ONLINE_CLASSIC+['covariance_warm64', 'cnn_warm64']
    allowed = {'public_X', 'control_code', 'metrics', 'mrc_physical_reference_snr_db', 'teacher_objective_evaluations'}
    initial_codes = np.rint(public['probe_controls']*LEVELS).astype(np.int16)
    trace_rows = 0
    for mi, name in enumerate(identity['ordinary_methods']):
        probes = identity['measurement_budgets'][name]
        if quality[mi, 10] != probes:
            raise ValueError('实际测量预算不同：'+name)
        if name not in trace_names:
            continue
        ck, sk = name+'__trace_control_code', name+'__trace_scores'
        allowed.update([ck, sk])
        controls, scores = arrays[ck], arrays[sk]
        if controls.shape != (probes, 128) or scores.shape != (probes,):
            raise ValueError('反馈轨迹数量不符。')
        if (not np.isfinite(controls).all() or not np.isfinite(scores).all()
                or not np.array_equal(controls, np.rint(controls))
                or np.any(controls < 0) or np.any(controls > LEVELS)):
            raise ValueError('反馈轨迹存在非法数字。')
        np.testing.assert_array_equal(controls[:16], initial_codes)
        np.testing.assert_array_equal(scores[:16], raw[1985:2001])
        np.testing.assert_array_equal(code[mi], controls[int(np.argmax(scores))])
        trace_rows += probes
    if set(arrays) != allowed:
        raise ValueError('接收记录缺少字段或出现未声明字段。')
    teacher = arrays['teacher_objective_evaluations']
    mrc = arrays['mrc_physical_reference_snr_db']
    if teacher.shape != () or not np.isfinite(teacher) or teacher <= 0 or teacher != np.rint(teacher):
        raise ValueError('教师目标评价次数错误。')
    if mrc.shape != () or not np.isfinite(mrc):
        raise ValueError('数字参考SNR错误。')
    return trace_rows


def collect(project, folder, freeze, rehearse):
    if bool(freeze) == bool(rehearse):
        raise ValueError('只能选择正式冻结分析或旧案例演练。')
    identity = rehearsal_identity(project) if rehearse else validate_freeze(freeze)
    if Path(identity['project']).resolve() != project or read(folder/'protocol.json') != identity:
        raise ValueError('接收协议与当前冻结身份不一致。')
    if identity['metric_order'] != METRIC_ORDER or identity['methods'][-2:] != ['teacher', 'mrc']:
        raise ValueError('指标或参考方法顺序不同。')
    if not rehearse:
        claim = read(project/'dataset_simulation/ops/full_baselines_20260925/fresh_confirmation_claim.json')
        expected = dict(freeze_path=str(freeze), freeze_sha256=sha256(freeze),
            identity_sha256=fingerprint(identity), output=str(folder), fresh_plan_sha256=identity['fresh_plan_sha256'])
        if claim != expected:
            raise ValueError('新计划没有登记到当前冻结身份和接收目录。')
        if len(identity['rows']) != 864 or identity['carriers'] != list(range(4, 21)):
            raise ValueError('最终确认要求全部864环境、17载频。')
        if set(Counter(r['joint_stratum_id'] for r in identity['rows']).values()) != {4}:
            raise ValueError('新确认必须保持每联合分层4个环境。')
    verify_sources(identity['source_sha256'])
    for name, digest in identity['source_sha256'].items():
        if sha256(folder/'source_snapshot'/name) != digest:
            raise ValueError('接收源码快照改变。')
    package = verify_bundle(Path(identity['runtime_bundle']))
    if sha256(Path(identity['runtime_bundle'])/'manifest.json') != identity['runtime_manifest_sha256']:
        raise ValueError('接收权重包改变。')
    with np.load(Path(identity['runtime_bundle'])/'public.npz') as f:
        public = {k: f[k].copy() for k in f.files}
    progress, complete = read(folder/'progress.json'), read(folder/'complete.json')
    if progress['status'] != 'complete' or complete['status'] != 'complete_reception_records':
        raise ValueError('接收执行尚未完整完成。')
    if complete['records_sha256'] != sha256(folder/'records.json') or complete['identity_sha256'] != fingerprint(identity):
        raise ValueError('接收完成记录来源不同。')
    rows, carriers, names = identity['rows'], identity['carriers'], identity['methods']
    if (complete['environments'], complete['carriers'], complete['methods'], complete['cases']) != (
            len(rows), len(carriers), len(names), len(rows)*len(carriers)*len(names)):
        raise ValueError('接收完成计数不符。')
    if complete['final_confirmation'] != (not rehearse) or complete['scope'] != identity['scope']:
        raise ValueError('演练与独立确认范围混淆。')
    records, values, physical_snr, teacher_calls = [], [], [], []
    files, trace_count = {}, 0
    for row in rows:
        record = committed_environment(folder, row, identity)
        if record is None:
            raise ValueError('缺少已提交的完整环境。')
        current, reference, teacher = [], [], []
        records.append(record)
        for item in record['carriers']:
            fc = item['carrier_ghz']
            expected = 'records/environment_%05d/carrier_%02d.npz'%(row['index'], fc)
            if item['path'] != expected or item['methods'] != len(names):
                raise ValueError('逐载频文件路径或方法数不同。')
            path = folder/item['path']
            with np.load(path, allow_pickle=False) as f:
                arrays = {k: f[k].copy() for k in f.files}
            trace_count += check_values(arrays, fc, identity, public)
            current.append(arrays['metrics'])
            reference.append(float(arrays['mrc_physical_reference_snr_db']))
            teacher.append(int(arrays['teacher_objective_evaluations']))
            files[item['path']] = item['sha256']
        values.append(current); physical_snr.append(reference); teacher_calls.append(teacher)
    if records != read(folder/'records.json') or progress['completed'] != len(rows):
        raise ValueError('完整索引与各环境提交不一致。')
    audit = dict(status='passed_complete_reception_records', scope=identity['scope'], at=now(),
        environments=len(rows), carriers=len(carriers), methods=len(names), cases=complete['cases'],
        feedback_trace_rows_checked=trace_count, control_codes_checked=complete['cases']*128,
        protocol_sha256=sha256(folder/'protocol.json'), records_sha256=sha256(folder/'records.json'),
        complete_sha256=sha256(folder/'complete.json'), raw_file_sha256=files,
        runtime_manifest_sha256=identity['runtime_manifest_sha256'],
        training_environments=package['cohort']['train_environments'],
        measurement_budgets=identity['measurement_budgets'], final_confirmation=not rehearse,
        quality_only=True, reception_run_times_not_formal_online_latency=True)
    return identity, np.asarray(values), np.asarray(physical_snr), np.asarray(teacher_calls), audit


def grouped(values, identity):
    rows, names, carriers = identity['rows'], identity['methods'], identity['carriers']
    groups = [('all', None, np.ones(len(rows), bool), None)]
    groups += [('carrier_ghz', fc, np.ones(len(rows), bool), ci) for ci, fc in enumerate(carriers)]
    factors = ['rays', 'max_delay_ns', 'angular_std_deg', 'power_bin']
    for factor in factors:
        for value in sorted({r['factors'][factor] for r in rows}):
            groups.append((factor, value, np.asarray([r['factors'][factor] == value for r in rows]), None))
    result = []
    for group, value, mask, ci in groups:
        current = values[mask]
        if ci is not None:
            current = current[:, ci:ci+1]
        aggregate = metrics(sufficient(current).sum(0))
        for mi, name in enumerate(names):
            result.append(dict(method=name, group=group, value=value, environments=int(mask.sum()),
                carriers=current.shape[1], **{f:float(aggregate[mi, fi]) for fi, f in enumerate(FIELDS)}))
    return result


def run(project, folder, output, freeze=None, rehearse=False):
    require_host()
    project, folder, output = project.resolve(), folder.resolve(), output.resolve()
    if freeze is not None:
        freeze = freeze.resolve()
    if output.exists():
        raise FileExistsError('最终统计另建目录，不覆盖已有结果。')
    sources = source_record(SOURCES)
    identity, values, physical_snr, teacher_calls, audit = collect(project, folder, freeze, rehearse)
    names, rows = identity['methods'], identity['rows']
    s = sufficient(values)
    point = metrics(s.sum(0))
    # 与不使用充分统计量的原始计数/功率聚合交叉核对。
    checked = 0
    for mi, name in enumerate(names):
        direct = direct_statistics(values[:, :, mi], name)
        for fi, field in enumerate(FIELDS):
            if field in direct:
                np.testing.assert_allclose(point[mi, fi], direct[field], rtol=1e-12, atol=0)
                checked += 1
    audit['aggregate_numbers_independently_recomputed'] = checked
    statistical_checks = self_check()
    if rehearse:
        plan, comparisons, draws = None, [], None
    else:
        plan = identity['statistics']
        if not plan['stratified'] or plan['statistical_unit'] != 'independent propagation environment, keeping all17 carriers together':
            raise ValueError('最终统计单位或分层约定不同。')
        draws = bootstrap(s, plan['bootstrap_repetitions'], plan['bootstrap_seed'],
                          strata=[r['joint_stratum_id'] for r in rows])
        comparisons = planned_comparisons(plan, names, s, draws)
        if len(comparisons) != len(plan['comparisons'])*len(plan['metrics']):
            raise ValueError('未完整执行冻结的比较清单。')
    overall = []
    for mi, name in enumerate(names):
        record = dict(method=name, independent_environments=len(rows), carriers_per_environment=len(identity['carriers']),
            privileged=name in ['teacher', 'mrc'], photonic_hardware=name != 'mrc',
            probes=identity['measurement_budgets'].get(name),
            training_environments=None if name in ONLINE_CLASSIC+['ridge_response', 'teacher', 'mrc'] else audit['training_environments'],
            **{field:float(point[mi, fi]) for fi, field in enumerate(FIELDS)})
        for fi, field in enumerate(FIELDS):
            column = draws[:, mi, fi] if draws is not None else None
            record[field+'_ci95'] = np.quantile(column, [.025, .975]).tolist() if column is not None and np.isfinite(column).all() else None
        overall.append(record)
    output.mkdir(parents=True)
    for name in sources:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    write_json(output/'audit.json', audit)
    write_json(output/'statistical_self_check.json', statistical_checks)
    write_json(output/'plan.json', plan if plan is not None else dict(scope='single_old_environment_rehearsal', inference=False))
    groups = grouped(values, identity)
    for filename, entries in [('overall', overall), ('groups', groups), ('comparisons', comparisons)]:
        write_json(output/(filename+'.json'), clean_json(dict(scope=identity['scope'], records=entries,
            final_confirmation=not rehearse, formal_online_latency=False)))
        # 演练没有比较条目，仍写入一个明确为空的CSV，不伪造区间。
        write_csv(output/(filename+'.csv'), entries)
    arrays = dict(sufficient=s, environment_metrics=metrics(s), methods=np.asarray(names), fields=np.asarray(FIELDS),
        environment_ids=np.asarray([r['environment_id'] for r in rows]),
        mrc_physical_reference_snr_db=physical_snr, teacher_objective_evaluations=teacher_calls)
    if draws is not None:
        arrays['bootstrap_metrics'] = draws
        arrays['strata'] = np.asarray([r['joint_stratum_id'] for r in rows])
    atomic_npz(output/'statistics.npz', **arrays)
    verify_sources(sources)
    files = ['audit.json', 'statistical_self_check.json', 'plan.json', 'statistics.npz']
    files += [n+ext for n in ['overall', 'groups', 'comparisons'] for ext in ['.json', '.csv']]
    result = dict(status='complete_old_record_rehearsal' if rehearse else 'complete_frozen_confirmation_statistics',
        at=now(), scope=identity['scope'], final_confirmation=not rehearse,
        methods=len(names), independent_environments=len(rows), carriers=len(identity['carriers']),
        paired_metric_comparisons=len(comparisons), input_protocol_sha256=sha256(folder/'protocol.json'),
        freeze_sha256=sha256(freeze) if freeze else None, source_sha256=sources,
        file_sha256={name:sha256(output/name) for name in files},
        uncertainty='stratified paired environment bootstrap; fixed trained weights; no retraining variance' if not rehearse else 'No interval from a single old environment',
        full_research_complete=False)
    write_json(output/'complete.json', result)
    print(json.dumps({k:result[k] for k in ['status', 'methods', 'independent_environments', 'paired_metric_comparisons']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'input', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', type=Path)
    mode.add_argument('--rehearse-old', action='store_true')
    args = parser.parse_args()
    run(args.project, args.input, args.output, args.freeze, args.rehearse_old)
