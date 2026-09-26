"""64次反馈组合的独立轨迹审计和配对统计；保留冻结的六阶段分析不变。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.audit_results import read_phase, check_summary
from study_full_baselines.audit_reception_v2 import check_budget, strict_cross
from study_full_baselines.analyze_reception import compute, clean_json, write_csv
from study_full_baselines.paired_statistics import self_check
from study_full_baselines.export_examples_when_ready import alive

HERE = Path(__file__).resolve().parent
SOURCES = ['our_method_feedback_candidates/'+n for n in
           ['analyze_full.py', 'WARM_COMPARISONS.json', 'ANALYSIS.md']]
SOURCES += ['study_full_baselines/'+n for n in ['common.py', 'audit_results.py',
    'audit_reception_v2.py', 'analyze_reception.py', 'paired_statistics.py', 'export_examples_when_ready.py']]
PHASES = ['classic', 'learned_evaluation', 'evaluation_feedback_warm']


def trace_check(trace, scores, chosen, selected_code, unique, x, public):
    """核对2种估计器×64次测量，首16项为共同公开输入，返回已测最优档位。"""
    trace, scores = np.asarray(trace), np.asarray(scores)
    chosen, unique = np.asarray(chosen), np.asarray(unique)
    if (trace.shape != (2, 64, 128) or scores.shape != (2, 64)
            or chosen.shape != (2,) or unique.shape != (2,) or np.asarray(selected_code).shape != (2, 128)):
        raise ValueError('反馈轨迹覆盖不完整。')
    if (not np.isfinite(scores).all() or not np.array_equal(trace, np.rint(trace))
            or np.any(trace < 0) or np.any(trace > LEVELS)):
        raise ValueError('反馈分数或控制档位无效。')
    prefix = np.rint(public['probe_controls']*LEVELS).astype(np.int16)
    if (not np.array_equal(trace[:, :16], np.broadcast_to(prefix, (2, 16, 128)))
            or not np.array_equal(scores[:, :16], np.broadcast_to(x[1985:2001], (2, 16)))):
        raise ValueError('没有使用共同16套公开测量。')
    if not np.array_equal(chosen, scores.argmax(axis=1)):
        raise ValueError('选中编号不是已测分数最大值。')
    if not np.array_equal(selected_code, trace[np.arange(2), chosen.astype(int)]):
        raise ValueError('最终控制与选中查询不符。')
    if not np.isin(unique, [0, 1]).all():
        raise ValueError('单候选方案最多测一次模型候选。')
    source = []
    for m in range(2):
        if unique[m] == 1 and any(np.array_equal(trace[m, 16], c) for c in prefix):
            raise ValueError('模型候选重复了已测的初始控制。')
        if len({tuple(c) for c in trace[m]}) != 64:
            raise ValueError('额外反馈重复测量了已有控制。')
        source.append('initial' if chosen[m] < 16 else
                      'response' if chosen[m] == 16 and unique[m] == 1 else 'geometric')
    return source


def preflight(project, output):
    """直接用已完成训练诊断的真实轨迹，另做三项破坏性副本反例。"""
    require_host()
    if output.exists():
        raise FileExistsError('前置核验不覆盖。')
    base = project/'dataset_simulation'; data = base/'outputs/quality_rank_hybrid_20260925'
    public = public_data(data); checks = []; negative_checks = []
    for index in [0, 2]:
        path = base/'diagnostics/20260926_feedback_candidates_train/records'/('%03d.npz' % index)
        meta = json.loads(path.with_suffix('.json').read_text())
        if sha256(path) != meta['sha256']:
            raise ValueError('训练反馈轨迹来源改变。')
        cache = base/'diagnostics/20260925_two_stage_train/records'/path.name
        cache_meta = json.loads(cache.with_suffix('.json').read_text())
        if sha256(cache) != cache_meta['sha256']:
            raise ValueError('公开输入缓存改变。')
        with np.load(cache) as f:
            x = f['public_x'].copy()
        with np.load(path) as f:
            trace = f['trace_control_code'][[1, 3]]; scores = f['trace_scores'][[1, 3]]
            code = f['control_code'][[1, 3]]
        chosen = scores.argmax(1)
        unique = np.asarray([meta['candidate_details'][i]['unique_response_candidates'] for i in [1, 3]])
        origins = trace_check(trace, scores, chosen, code, unique, x, public)
        checks.append(dict(environment=meta['environment_id'], carrier=meta['carrier_ghz'],
                           selected_sources=origins, all64queries_checked=True))
        if index == 0:
            for fault in ['changed_public_score', 'wrong_selected_code', 'budget_exceeded']:
                t, s, u = trace.copy(), scores.copy(), code.copy()
                if fault == 'changed_public_score': s[0, 0] += 1
                elif fault == 'wrong_selected_code': u[0, 0] = (u[0, 0]+1) % 77
                else: t = np.concatenate([t, t[:, :1]], axis=1)
                rejected = False
                try: trace_check(t, s, chosen, u, unique, x, public)
                except ValueError: rejected = True
                if not rejected: raise ValueError('未拒绝被篡改的轨迹：'+fault)
                negative_checks.append(fault)
    plan = json.loads((HERE/'WARM_COMPARISONS.json').read_text())
    known = set(ONLINE_CLASSIC+['covariance_response', 'complex_response_cnn', 'covariance_warm64', 'cnn_warm64'])
    if len(plan['comparisons']) != 10:
        raise ValueError('计划比较数改变。')
    for spec in plan['comparisons']:
        if set(spec['terms'])-known or sum(spec['terms'].values()) != 0:
            raise ValueError('比较方法或系数错误。')
    write_json(output, dict(status='passed', checks=checks, negative_checks=negative_checks,
        statistical_checks=self_check(), planned_comparisons=10, planned_tests=50,
        source_sha256=source_record(SOURCES), at=now()))
    print(json.dumps(dict(status='passed', positive_trajectories=4, negative_checks=3, planned_tests=50)), flush=True)


def collect(project):
    base = project/'dataset_simulation'; data = base/'outputs/quality_rank_hybrid_20260925'
    study = base/'baseline_results/20260925_full_baselines'
    rows = [r for r in check_data(data)['environments'] if r['split'] == 'test']
    if len(rows) != 216 or [r['index'] for r in rows] != list(range(216)):
        raise ValueError('仅允许旧216探索测试。')
    reports = {}; arrays = {}; digest = sha256(data/'manifest.json')
    for phase in PHASES:
        info, value = read_phase(study/phase, rows, digest, False)
        info.update(check_summary(study/phase, value, rows)); info.update(check_budget(study/phase, value))
        reports[phase] = info; arrays[phase] = value
    ref = strict_cross(arrays, ('classic', 'codebook'), ('evaluation_feedback_warm', 'codebook64'))
    if ref['status'] != 'matched' or ref['environments'] != 216:
        raise ValueError('码本参考未逐环境一致重放。')
    public = public_data(data); origin_records = []; trace_values = 0
    for row in rows:
        name = 'environment_%05d.npz' % row['index']
        path = study/'evaluation_feedback_warm/records'/name
        meta = json.loads(path.with_suffix('.json').read_text())
        original = json.loads((study/'classic/records'/name).with_suffix('.json').read_text())
        if not meta['reference_replay'] or meta['reference_record_sha256'] != original['sha256']:
            raise ValueError('码本重放没有绑定正确的原始记录。')
        with np.load(data/row['path']/'data.npz') as f: x = f['X']
        with np.load(path) as f:
            trace, scores, chosen = f['trace_control_code'], f['trace_scores'], f['selected_query']
            unique, code, sim = f['unique_model_queries'], f['control_code'], f['feedback_simulator_seconds']
            if (trace.shape != (17, 2, 64, 128) or scores.shape != (17, 2, 64)
                    or chosen.shape != (17, 2) or unique.shape != (17, 2)
                    or sim.shape != (17, 2) or not np.isfinite(sim).all() or np.any(sim < 0)):
                raise ValueError('环境内轨迹尺寸或反馈模拟器时间无效。')
            for ci, fc in enumerate(range(4, 21)):
                origins = trace_check(trace[ci], scores[ci], chosen[ci], code[ci, 1:], unique[ci], x[ci], public)
                for m, name in enumerate(['covariance_warm64', 'cnn_warm64']):
                    origin_records.append(dict(environment_id=row['environment_id'], carrier_ghz=fc,
                        method=name, selected_query=int(chosen[ci, m]), selected_source=origins[m],
                        unique_response_candidates=int(unique[ci, m]), geometric_queries=int(48-unique[ci, m])))
                trace_values += int(trace[ci].size)
    names = ONLINE_CLASSIC+['covariance_response', 'complex_response_cnn', 'covariance_warm64', 'cnn_warm64']
    values = []; catalog = []
    for name in names:
        phase = 'classic' if name in ONLINE_CLASSIC else 'evaluation_feedback_warm' if name.endswith('warm64') else 'learned_evaluation'
        current = arrays[phase]; mi = current['methods'].index(name)
        if current['ids'] != [r['environment_id'] for r in rows]:
            raise ValueError('方法间环境顺序不同。')
        values.append(current['values'][:, :, mi, :10])
        calls = current['values'][:, :, mi, current['metric_order'].index('feedback_calls')]
        catalog.append(dict(id=name, sources=[dict(phase=phase, method=name)], feedback_calls=float(calls.mean()),
            privileged=False, photonic_hardware=True, training_environments=None if name in ONLINE_CLASSIC else 864,
            timing='Separate actual whole-controller online timing required; cached forward not counted here'))
    audit = dict(status='passed', scope='three-phase old216 warm extension only', phases=reports,
        reference=ref, checked_trace_control_values=trace_values, trajectory_cases=len(origin_records),
        trace_checks='all initial16 codes/scores; exact64 budget; unique queries; measured argmax returned',
        all_quality_summaries_recomputed=True, data_manifest_sha256=digest)
    return rows, names, np.stack(values, axis=2), catalog, audit, origin_records


def run(project, output, wait_pid=None):
    require_host(); output.mkdir(parents=True, exist_ok=False)
    sources = source_record(SOURCES)
    write_json(output/'protocol.json', dict(source_sha256=sources, wait_pid=wait_pid,
        phases=PHASES, scope='old216 exploratory warm extension; not sealed confirmation', at=now()))
    for name in SOURCES:
        dst = output/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dst)
    study = project/'dataset_simulation/baseline_results/20260925_full_baselines'
    while True:
        pending = []
        for phase in PHASES:
            path = study/phase/'progress.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            if list((study/phase).glob('failure*.json')): raise RuntimeError('前置评分失败：'+phase)
            if state.get('status') != 'complete': pending.append(phase)
        if not pending: break
        if wait_pid is None or not alive(wait_pid):
            raise RuntimeError('前置评分不完整且等待的反馈评价进程不在运行。')
        verify_sources(sources)
        write_json(output/'progress.json', dict(status='waiting', pending=pending, pid=os.getpid(), at=now()))
        time.sleep(30)
    verify_sources(sources); rows, names, values, catalog, audit, origins = collect(project)
    plan = json.loads((HERE/'WARM_COMPARISONS.json').read_text())
    overall, comparisons, arrays = compute(values, names, catalog, rows, plan)
    write_json(output/'audit.json', audit); write_json(output/'plan.json', plan)
    atomic_npz(output/'statistics.npz', **arrays)
    write_json(output/'overall.json', clean_json(dict(status='complete', scope=plan['scope'], records=overall)))
    write_json(output/'comparisons.json', clean_json(dict(status='complete', scope=plan['scope'], records=comparisons)))
    write_json(output/'selection_origins.json', dict(scope='descriptive provenance, not causal contribution', records=origins))
    write_csv(output/'overall.csv', overall); write_csv(output/'comparisons.csv', comparisons)
    write_csv(output/'selection_origins.csv', origins); verify_sources(sources)
    files = ['audit.json', 'plan.json', 'statistics.npz', 'overall.json', 'comparisons.json',
             'selection_origins.json', 'overall.csv', 'comparisons.csv', 'selection_origins.csv']
    write_json(output/'complete.json', dict(status='complete', scope=plan['scope'], at=now(),
        methods=len(names), comparisons=len(plan['comparisons']), metric_tests=len(comparisons),
        file_sha256={name: sha256(output/name) for name in files}, source_sha256=sources))
    write_json(output/'progress.json', dict(status='complete', at=now()))
    print(json.dumps(dict(status='complete', methods=len(names), metric_tests=len(comparisons))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-pid', type=int); p.add_argument('--preflight', action='store_true'); a = p.parse_args()
    try:
        if a.preflight: preflight(a.project, a.output)
        else: run(a.project, a.output, a.wait_pid)
    except BaseException:
        if not a.preflight and a.output.exists():
            write_json(a.output/('failure_%d.json' % os.getpid()), dict(traceback=traceback.format_exc(), at=now()))
        raise
