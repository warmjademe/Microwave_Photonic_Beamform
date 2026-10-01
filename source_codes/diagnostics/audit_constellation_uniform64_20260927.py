"""只读审计 RQ1 星座案例：控制来源、独立指标、完整重放及全测试控制重合率。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host, LEVELS, rng_for, sha256, now
from study_uniform64.run import initialize, STATE, load_original
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_receiver import public_observation, scoring_engines
from study_full_baselines.export_signal_examples import photonic_trace
from paper_results_20260927.build_uniform64 import METHODS, LABELS


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype='<i2').tobytes()).hexdigest()


def check_close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-28)


def main(project, output):
    require_host()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / 'workers').mkdir()
    root = project / 'dataset_simulation/baseline_results/20260927_uniform64_all13'
    figures = project / 'dataset_simulation/diagnostics/20260927_uniform64_all13_figures'
    protocol = read(root / 'protocol.json')
    old_root = Path(protocol['source_output'])
    old_protocol = read(old_root / 'protocol.json')
    manifest = read(figures / 'manifest.json')
    audit = read(root / 'analysis/audit.json')
    name_pdf = 'rq1_constellation_all13_12_64.pdf'
    assert sha256(figures / name_pdf) == manifest['figures'][name_pdf]
    assert sha256(root / 'protocol.json') == manifest['source_protocol_sha256']
    assert sha256(project / 'source_codes/paper_results_20260927/build_uniform64.py') == manifest['builder_sha256']
    row = protocol['rows'][0]
    env, old, old_sha = load_original(row, 12, protocol)
    path = root / 'records/environment_00000/carrier_12.npz'
    assert sha256(path) == audit['raw_file_sha256'][str(path.relative_to(root))]
    new = arrays(path)
    plotted = arrays(figures / 'constellation_all13_12.npz')
    assert plotted['methods'].tolist() == METHODS
    codes, metrics, details, all_scores, all_controls = [], [], [], [], []
    for i, name in enumerate(METHODS):
        src, order = (new, protocol['methods']) if name in protocol['methods'] else (old, old_protocol['methods'])
        j = order.index(name)
        code, stored = src['control_code'][j], src['metrics'][j, :10]
        trace, scores = src[name + '__trace_control_code'], src[name + '__trace_scores']
        assert trace.shape == (64, 128) and scores.shape == (64,)
        np.testing.assert_array_equal(code, trace[scores.argmax()])
        check_close(plotted['metrics'][i], stored)
        # 不调用原指标函数；从图中每个复数点重新判决 I/Q 符号并计算误差向量。
        got, sent = plotted['received_qpsk'][i], plotted['sent_qpsk']
        bad_i, bad_q = np.signbit(got.real) != np.signbit(sent.real), np.signbit(got.imag) != np.signbit(sent.imag)
        nmse = np.sum(abs(got - sent[None]) ** 2) / (got.shape[0] * np.sum(abs(sent) ** 2))
        independent = [int(bad_i.sum() + bad_q.sum()), 2 * got.size, float(nmse),
                       int((bad_i | bad_q).sum()), got.size, int((bad_i | bad_q).any(axis=1).sum()), got.shape[0]]
        check_close(independent, stored[:7])
        proposal_key = name + '__proposal_control_code'
        detail = dict(method=name, label=LABELS[i], selected_control_sha256=digest(code),
                      selected_probe_1based=int(scores.argmax()) + 1,
                      initial_probe_matches_1based=(np.flatnonzero(np.all(trace[:16] == code, axis=1)) + 1).tolist(),
                      best_initial_score=float(scores[:16].max()), best_extra_score=float(scores[16:].max()),
                      unique_probed_controls=len(np.unique(trace, axis=0)), bit_errors=independent[0],
                      bits_tested=independent[1], ber_percent=100 * independent[0] / independent[1],
                      evm_percent=100 * float(np.sqrt(nmse)), symbol_errors=independent[3],
                      paired_snr_db=10 * float(np.log10(stored[7] / stored[8])))
        if proposal_key in src:
            proposed = src[proposal_key]
            positions = np.flatnonzero(np.all(trace == proposed, axis=1))
            detail.update(proposal_sha256=digest(proposed), proposal_probe_1based=(positions + 1).tolist(),
                          proposal_score=float(scores[positions[0]]),
                          proposal_selected=bool(np.array_equal(code, proposed)))
        codes.append(code); metrics.append(stored); details.append(detail)
        all_scores.append(scores); all_controls.append(trace)
    codes, metrics = np.array(codes), np.array(metrics)
    control_groups = {}
    for detail in details:
        control_groups.setdefault(detail['selected_control_sha256'], []).append(detail['method'])
    for a in range(len(METHODS)):
        for b in range(a):
            if np.array_equal(codes[a], codes[b]):
                np.testing.assert_array_equal(plotted['received_qpsk'][a], plotted['received_qpsk'][b])
    coordinate = METHODS.index('coordinate')
    changed = np.flatnonzero(codes[coordinate] != codes[0])
    result = dict(status='running', started_at=now(), environment_id=row['environment_id'],
                  environment_index=0, carrier_ghz=12, power_dbm=env['power_dbm'],
                  paths=len(env['delays_s']), measured_draws=8, qpsk_symbols_per_draw=31,
                  source_hashes=dict(new_record=sha256(path), old_record=old_sha,
                                     figure_pdf=sha256(figures / name_pdf),
                                     constellation_arrays=sha256(figures / 'constellation_all13_12.npz'),
                                     script=sha256(__file__)),
                  controls_grouped=control_groups, methods=details,
                  coordinate_changes=[dict(array_index_0based=int(k), kind='delay' if k < 64 else 'attenuation',
                                           antenna_1based=int(k % 64) + 1, common_code=int(codes[0, k]),
                                           coordinate_code=int(codes[coordinate, k])) for k in changed],
                  independent_metrics_passed=True)
    save(output / 'case_audit.json', result)
    np.savez_compressed(output / 'case_evidence.npz', methods=METHODS, control_code=codes,
                        trace_control_code=np.array(all_controls), trace_scores=np.array(all_scores),
                        received_qpsk=plotted['received_qpsk'], sent_qpsk=plotted['sent_qpsk'], metrics=metrics)
    print('Independent bit decisions/EVM passed; unique final controls:', len(control_groups), flush=True)

    # 从冻结模型重新生成候选，并重新测量全部反馈，不能仅复用已存最终控制。
    initialize(output, protocol)
    public = STATE['public']
    models = dict(STATE['models'])
    models.update({n: OnlineController(Path(protocol['runtime_bundle']), n, public) for n in protocol['reused_methods']})
    raw, observed = public_observation(env, 12, public, protocol['backend'])
    np.testing.assert_array_equal(raw, old['public_X'])
    np.testing.assert_array_equal(raw, new['public_X'])
    cache, calls = {}, 0
    def measure(u, call):
        nonlocal calls
        calls += 1
        key = (call, tuple(observed.codes(u)))
        if key not in cache:
            cache[key] = observed.measure_detailed(u, rng_for(env['seed'], 12, 620, call))['score']
        return cache[key]
    for i, name in enumerate(METHODS):
        got = models[name].decide(raw.copy(), rng_for(0, env['seed'], 12, 630), measure)
        np.testing.assert_array_equal(got['control_code'], codes[i])
        np.testing.assert_array_equal(got['trace_control_code'], all_controls[i])
        check_close(got['trace_scores'], all_scores[i])
        if name + '__proposal_control_code' in new:
            np.testing.assert_array_equal(got['proposal_control_code'], new[name + '__proposal_control_code'])
        print('Controller replay passed:', name, flush=True)
    assert calls == len(METHODS) * 48
    # 用真实时域 I/Q 解调路径重建图形点，不从保存的 metrics 构造输出。
    engine, clean, payload = scoring_engines(env, 12, public, protocol['backend'])
    np.testing.assert_array_equal(payload, plotted['sent_qpsk'])
    max_delta = 0.
    replayed = []
    for i, name in enumerate(METHODS):
        trace = photonic_trace(engine, clean, codes[i], payload, env['seed'], 12)
        check_close(trace['metrics'], metrics[i])
        check_close(trace['received_qpsk'], plotted['received_qpsk'][i])
        max_delta = max(max_delta, float(abs(trace['received_qpsk'] - plotted['received_qpsk'][i]).max()))
        replayed.append(trace['received_qpsk'].copy())
    # 倒序再检查，排除前一次控制留下可变状态影响后续方法。
    for i in range(len(METHODS) - 1, -1, -1):
        check_close(photonic_trace(engine, clean, codes[i], payload, env['seed'], 12)['received_qpsk'], replayed[i])
    result.update(controller_replay_passed=True, replayed_feedback_calls=calls,
                  replayed_logical_probes=14 * 64, unique_extra_simulator_queries=len(cache),
                  scoring_replay_passed=True, scoring_reverse_order_passed=True, max_received_iq_difference=max_delta)
    save(output / 'case_audit.json', result)
    print('All 14 controller, feedback and independent time-domain I/Q replays passed', flush=True)

    # 全量原始控制的只读扫描：确认单案例重合不是全数据集被写成常数。
    pair_counts = np.zeros((len(METHODS), len(METHODS)), np.int64)
    group_sizes, own_groups, selected_code_sets = Counter(), Counter(), [set() for _ in METHODS]
    same13, total = 0, 0
    selected_methods = [i for i, n in enumerate(METHODS) if n != 'coordinate']
    for row in protocol['rows']:
        folder = 'records/environment_%05d' % row['index']
        for fc in protocol['carriers']:
            file = 'carrier_%02d.npz' % fc
            with np.load(root / folder / file) as n, np.load(old_root / folder / file) as o:
                nc, oc = n['control_code'], o['control_code']
                cc = np.stack([nc[protocol['methods'].index(m)] if m in protocol['methods']
                               else oc[old_protocol['methods'].index(m)] for m in METHODS])
            equal = np.all(cc[:, None, :] == cc[None, :, :], axis=-1)
            pair_counts += equal
            group_sizes[int(equal.sum(axis=1).max())] += 1
            own_groups[int(equal[-1].sum())] += 1
            same13 += int(np.all(equal[np.ix_(selected_methods, selected_methods)]))
            for i, c in enumerate(cc):
                selected_code_sets[i].add(c.astype('<i2').tobytes())
            total += 1
        if (row['index'] + 1) % 108 == 0:
            print('Full control scan:', row['index'] + 1, '/864 environments', flush=True)
    assert total == 864 * 17
    full = dict(total_cases=total, methods=METHODS, pair_same_control_count=pair_counts.tolist(),
                maximum_same_control_group_size_counts=dict(sorted(group_sizes.items())),
                our_method_same_control_group_size_counts=dict(sorted(own_groups.items())),
                same_13_methods_as_figure_count=same13,
                distinct_selected_controls={m: len(s) for m, s in zip(METHODS, selected_code_sets)})
    save(output / 'full_test_control_overlap.json', full)
    with (output / 'case_methods.csv').open('w', newline='') as f:
        fields = ['method', 'selected_probe_1based', 'bit_errors', 'bits_tested', 'ber_percent',
                  'evm_percent', 'paired_snr_db', 'best_initial_score', 'best_extra_score', 'selected_control_sha256']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(details)
    result.update(status='passed', completed_at=now(), full_test_control_overlap_file='full_test_control_overlap.json',
                  original_records_modified=False, paper_or_figures_modified=False)
    save(output / 'case_audit.json', result)
    print(json.dumps(dict(status='passed', same_13_count=same13, cases=total,
                          our_vs_initial_same=int(pair_counts[-1, 0]), max_iq_delta=max_delta)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.output)
