"""核对批量并行、两次接续及405个旧案例；保留全部测试和提交证据。"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host, sha256, write_json, now, LEVELS, ONLINE_CLASSIC
from study_full_baselines.confirmation_batch import (fingerprint, committed_environment,
    check_carrier, verify_artifacts)


def check_preserved(root, name):
    snapshot = json.loads((root/name).read_text())
    for path, old in snapshot['files'].items():
        f = root/path
        if sha256(f) != old['sha256'] or f.stat().st_mtime_ns != old['mtime_ns'] or f.stat().st_size != old['size']:
            raise ValueError('续跑重写了已经提交的结果：'+path)
    return snapshot


def run(project, folder):
    require_host(); dest = folder/'integrity_verification.json'
    if dest.exists(): raise FileExistsError('不覆盖核查结果。')
    protocol = json.loads((folder/'protocol.json').read_text())
    if protocol['scope'] != 'old_test_rehearsal' or protocol['final_confirmation']:
        raise ValueError('此核查只用于旧环境演练。')
    fp = fingerprint(protocol); methods = protocol['methods']; ordinary = protocol['ordinary_methods']
    if len(methods) != 45 or protocol['carriers'] != [4, 12, 20] or len(protocol['rows']) != 3:
        raise ValueError('批量演练范围改变。')
    for name, digest in protocol['source_sha256'].items():
        if sha256(folder/'source_snapshot'/name) != digest or sha256(project/'source_codes'/name) != digest:
            raise ValueError('代码来源改变。')
    verify_artifacts(protocol['model_artifacts'])
    first = check_preserved(folder, 'checkpoint_before_resume.json')
    last = check_preserved(folder, 'checkpoint_before_noop.json')
    if len(set(first['worker_pids'])) != 2: raise ValueError('首次未由两个worker并行执行。')
    lock = json.loads((folder/'live_lock_verification.json').read_text())
    if not lock['owner_live'] or not lock['second_writer_rejected']: raise ValueError('并发写锁未验证。')
    attempts = [json.loads(p.read_text()) for p in sorted((folder/'attempts').glob('*.json'))]
    if [(r['reused_environments'], r['newly_computed_environments']) for r in attempts] != [(0, 2), (2, 1), (3, 0)]:
        raise ValueError('未按2环境→续跑1环境→完整复用的顺序完成。')
    workers = [p.name for p in (folder/'workers').glob('*.json') if not p.name.endswith('_progress.json')]
    if sorted(workers) != sorted(last['worker_files']): raise ValueError('无工作续跑仍创建了数值worker。')
    prior = project/'dataset_simulation/diagnostics/20260926_confirmation_receiver_preflight'
    prior_summary = json.loads((prior/'summary.json').read_text())
    prior_quality_gate = {Path(p).name: digest for p, digest in prior_summary['result_and_reference_sha256'].items()
                          if Path(p).parent == prior/'records'}
    cases = 0; matched = 0; feedback_rows = 0; committed_hashes = {}
    data = Path(protocol['data_path'])
    for row in protocol['rows']:
        committed = committed_environment(folder, row, protocol)
        if committed is None: raise ValueError('环境未完整提交。')
        for name, digest in protocol['input_sha256'][row['environment_id']].items():
            if sha256(data/row['path']/name) != digest: raise ValueError('原输入文件变化。')
        with np.load(data/row['path']/'data.npz') as f: original_x = f['X'].copy()
        for record in committed['carriers']:
            fc = record['carrier_ghz']; path = folder/record['path']; committed_hashes[record['path']] = sha256(path)
            with np.load(path) as f: arrays = {k: f[k].copy() for k in f.files}
            np.testing.assert_array_equal(arrays['public_X'], original_x[fc-4])
            codes = arrays['control_code']; values = arrays['metrics']
            if codes.shape != (45, 128) or values.shape != (45, 13): raise ValueError('形状错误。')
            if np.any(codes[:-1] < 0) or np.any(codes[:-1] > LEVELS) or np.any(codes[:-1] != np.rint(codes[:-1])):
                raise ValueError('光子控制码非法。')
            np.testing.assert_array_equal(codes[-1], np.full(128, -1))
            if methods[-1] != 'mrc' or not np.isnan(values[-1, 7:]).all(): raise ValueError('MRC字段错误。')
            np.testing.assert_array_equal(values[:, [1, 4, 6]], np.tile([496, 248, 8], (45, 1)))
            if not np.isfinite(values[:, :7]).all(): raise ValueError('接收计数缺失。')
            for e, total in [(0, 1), (3, 4), (5, 6)]:
                if np.any(values[:, e] < 0) or np.any(values[:, e] > values[:, total]): raise ValueError('错误计数范围非法。')
            for mi, name in enumerate(ordinary):
                expected = 0 if name.endswith('_prior') else 64 if name in ONLINE_CLASSIC[1:] or name.endswith('_warm64') else 16
                if values[mi, 10] != expected: raise ValueError('预算错误。')
                trace_name = name+'__trace_control_code'
                if trace_name in arrays:
                    trace = arrays[trace_name]; score = arrays[name+'__trace_scores']
                    if trace.shape != (expected, 128) or score.shape != (expected,): raise ValueError('反馈轨迹形状不同。')
                    if not np.isfinite(score).all() or np.any(trace < 0) or np.any(trace > LEVELS): raise ValueError('反馈轨迹非法。')
                    feedback_rows += expected
            if row['index'] == 0:
                reference = prior/'records'/('carrier_%02d.npz' % fc)
                if sha256(reference) != prior_quality_gate[reference.name]: raise ValueError('旧接收核查参考改变。')
                with np.load(reference) as f:
                    np.testing.assert_array_equal(codes, f['control_code'])
                    np.testing.assert_allclose(values[:, :10], f['metrics_gpu'][:, :10], rtol=1e-12, atol=0., equal_nan=True)
                matched += len(methods)
            cases += len(methods)
    negative = []
    wrong = copy.deepcopy(protocol); wrong['backend'] = 'changed-backend-for-negative-check'
    try: committed_environment(folder, protocol['rows'][0], wrong)
    except ValueError: negative.append('reject_changed_identity')
    else: raise ValueError('身份改变未拒绝。')
    fault = folder/'negative_uncommitted'; fault.mkdir(exist_ok=False)
    (fault/'carrier_04.npz').write_bytes(b'intentionally uncommitted negative fixture')
    try: check_carrier(fault, 4, protocol['rows'][0], protocol)
    except ValueError: negative.append('reject_uncommitted_file_without_overwrite')
    else: raise ValueError('未提交文件没有被拒绝。')
    report = dict(status='passed_parallel_resume_and_records_audit', cases=cases, environments=3, methods=45,
        matched_prior_receiver_cases=matched, feedback_rows_checked=feedback_rows,
        preserved_first_checkpoint_files=len(first['files']), preserved_noop_checkpoint_files=len(last['files']),
        parallel_worker_pids=first['worker_pids'], attempts=[{k: r[k] for k in
            ['reused_environments', 'newly_computed_environments', 'status']} for r in attempts],
        negative_checks=negative, second_writer_rejected=True, no_worker_created_on_complete_resume=True,
        source_identity_sha256=fp, committed_file_sha256=committed_hashes,
        final_confirmation=False, new_environment_signals_generated=0,
        auditor_sha256=sha256(Path(__file__)), at=now())
    write_json(dest, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'committed_file_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); run(args.project.resolve(), args.input.resolve())
