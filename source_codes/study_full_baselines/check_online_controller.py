"""实际权重单条执行前置核对：公开输入边界、反馈轨迹、合法控制和完整接收。"""
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
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.online_controller import OnlineController, method_names
from study_full_baselines.analyze_reception import canonical
from our_method_response_control.train import precision


def references(study, row):
    result = {}; files = {}
    for phase in ['classic', 'learned_evaluation', 'evaluation_components',
                  'evaluation_real_response_cnn', 'evaluation_feedback_warm']:
        folder = study/phase; p = folder/'records'/('environment_%05d.npz' % row['index'])
        marker = p.with_suffix('.json')
        if not marker.exists(): continue
        meta = json.loads(marker.read_text()); protocol = json.loads((folder/'protocol.json').read_text())
        if meta['environment_id'] != row['environment_id'] or sha256(p) != meta['sha256']:
            raise ValueError('参考记录身份改变。')
        names = protocol.get('methods') or [m['name'] for m in protocol['bundle']['methods']]
        files[str(p)] = meta['sha256']
        with np.load(p) as f:
            for i, name in enumerate(names):
                key = {'public16': 'ttd_das', 'codebook64': 'codebook'}.get(name, canonical(name))
                record = dict(control=f['control_code'][:, i].copy(), quality=f['metrics'][:, i, :10].copy(),
                              path=str(p), original_method=name)
                if key in result:
                    old = result[key]
                    if not np.array_equal(old['control'], record['control']): raise ValueError('重复参考控制不一致。')
                if name.endswith('_warm64'):
                    record.update(trace=f['trace_control_code'][:, i-1].copy(),
                                  scores=f['trace_scores'][:, i-1].copy())
                result[key] = record
    return result, files


def run(project, output):
    require_host(); precision()
    if output.exists(): raise FileExistsError('在线检查不覆盖原结果。')
    root = project/'dataset_simulation'; data = root/'outputs/quality_rank_hybrid_20260925'
    study = root/'baseline_results/20260925_full_baselines'
    manifest = check_data(data); row = next(r for r in manifest['environments'] if r['split'] == 'test' and r['index'] == 0)
    index = {r['environment_id']: r for r in json.loads((data/'records.json').read_text())}
    for name, digest in index[row['environment_id']]['file_sha256'].items():
        if sha256(data/row['path']/name) != digest: raise ValueError('固定前置案例数据改变。')
    with np.load(data/'public.npz') as f: public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    with np.load(data/row['path']/'data.npz') as f: x = f['X'].copy()
    ref, ref_hashes = references(study, row)
    names = method_names([216, 432])
    if set(names)-set(ref): raise ValueError('固定案例尚缺参考记录：'+str(set(names)-set(ref)))
    source_names = ['study_full_baselines/'+n for n in ['online_controller.py', 'check_online_controller.py',
        'ONLINE_PROTOCOL.md', 'common.py', 'analyze_reception.py']]
    folders = ['deep_common', 'baseline_common', 'baseline_mlp', 'our_method_quality_rank',
               'our_method_response_control', 'baseline_response_realcnn', 'baseline_joint_response_linear',
               'our_method_two_stage', 'our_method_feedback_candidates']
    folders += ['baseline_'+n for n in ONLINE_CLASSIC+DEEP_METHODS]
    source_names += [str(f.relative_to(SOURCE)) for name in folders for f in (SOURCE/name).glob('*.py')]
    source_names = sorted(set(source_names)); sources = source_record(source_names)
    output.mkdir(parents=True); (output/'records').mkdir()
    for name in source_names:
        dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE/name, dest)
    write_json(output/'protocol.json', dict(methods=names, test_index=0, carriers=[4, 12, 20],
        environment_id=row['environment_id'], source_sha256=sources, reference_files=ref_hashes,
        data_manifest_sha256=sha256(data/'manifest.json'), batch_size=1,
        selection='fixed old-test case, no quality selection', final_fair_timing=False,
        load_average=list(os.getloadavg()), at=now()))
    # 仅由评价端持有传播环境；控制器只能调用返回质量标量的测量接口。
    env = json.loads((data/row['path']/'environment.json').read_text()); engines = {}
    for fc in [4, 12, 20]:
        observed, _ = frame_engine(env, fc, public['pilot_qpsk'], 0)
        scored, payload = frame_engine(env, fc, public['pilot_qpsk'], 5)
        clean = clean_frame_engine(env, fc, public['pilot_qpsk'], payload)
        engines[fc] = observed, scored, payload, clean
    original_load = np.load; loaded = []
    def guarded_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            path = Path(path)
            if path.name in ['data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy']:
                raise RuntimeError('在线入口试图读取数据或缓存预测：'+str(path))
            loaded.append(str(path))
        return original_load(path, *args, **kwargs)
    details = []; model_records = []; negative = []
    with patch('numpy.load', guarded_load):
        for forbidden in ['data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy']:
            rejected = False
            try: np.load(forbidden)
            except RuntimeError: rejected = True
            if not rejected: raise ValueError('读取边界哨兵无效。')
            negative.append('reject_'+forbidden)
    for ni, name in enumerate(names):
        with patch('numpy.load', guarded_load): model = OnlineController(project, name, public)
        results = []; extra = name in ONLINE_CLASSIC[1:] or name.endswith('_warm64')
        for fc in [4, 12, 20]:
            observed, scored, payload, clean = engines[fc]
            def measure(u, call):
                return observed.measure_detailed(u, rng_for(row['seed'], fc, 620, call))['score']
            old_precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            with patch('numpy.load', guarded_load):
                got = model.decide(x[fc-4], rng_for(0, row['seed'], fc, 630), measure if extra else None)
                if fc == 4:
                    repeated = model.decide(x[fc-4], rng_for(0, row['seed'], fc, 630), measure if extra else None)
                    if not np.array_equal(got['control_code'], repeated['control_code']):
                        raise ValueError('同输入同随机流未复现设置：'+name)
                    if 'trace_scores' in got and (not np.array_equal(got['trace_scores'], repeated['trace_scores']) or
                            not np.array_equal(got['trace_control_code'], repeated['trace_control_code'])):
                        raise ValueError('重复执行的反馈轨迹不同。')
            if old_precision != (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32):
                raise ValueError('在线控制器未恢复计算精度选项。')
            code = got['control_code']; reference = ref[name]['control'][fc-4]
            if code.shape != (128,) or np.any(code < 0) or np.any(code > LEVELS): raise ValueError('控制码无效。')
            pair = np.asarray([reference, code])/LEVELS
            quality, replayed_code = reception_metrics(scored, clean, pair, payload, row['seed'], fc)
            if not np.array_equal(replayed_code, np.stack([reference, code])):
                raise ValueError('接收器的档位投影不一致。')
            if not np.allclose(quality[0], ref[name]['quality'][fc-4], rtol=1e-12, atol=0):
                raise ValueError('原控制的接收重放未复现原始结果：'+name)
            if 'trace_scores' in got:
                trace = got['trace_control_code']; scores = got['trace_scores']
                if not np.array_equal(trace[:16], np.rint(public['probe_controls']*LEVELS)):
                    raise ValueError('没有共用初始16套设置。')
                if not np.array_equal(scores[:16], x[fc-4, 1985:2001]): raise ValueError('初始测量被改变。')
                if name.endswith('_warm64') and not np.array_equal(code, trace[scores.argmax()]):
                    raise ValueError('反馈确认没有选择实测最佳设置。')
            trace_match = None
            if 'trace' in ref[name]:
                trace_match = bool(np.array_equal(got['trace_control_code'], ref[name]['trace'][fc-4]) and
                                   np.array_equal(got['trace_scores'], ref[name]['scores'][fc-4]))
            path = output/'records'/('%s_fc%02d.npz' % (name, fc))
            arrays = dict(control_code=np.stack([reference, code]), quality=quality)
            for key in ['trace_control_code', 'trace_scores', 'estimated_response']:
                if key in got: arrays[key] = got[key]
            atomic_npz(path, **arrays)
            record = dict(method=name, carrier_ghz=fc, feedback_calls=got['feedback_calls'],
                changed_control_codes=int(np.sum(code != reference)),
                maximum_code_difference=int(np.max(abs(code-reference))),
                delta_bit_errors=int(quality[1, 0]-quality[0, 0]), delta_nmse=float(quality[1, 2]-quality[0, 2]),
                full_receiver_reference_matched=True, full_feedback_trace_matches_batch_reference=trace_match,
                software_seconds_diagnostic=got['software_seconds'], feedback_simulator_seconds=got['feedback_simulator_seconds'],
                result_file=str(path.relative_to(output)), result_sha256=sha256(path))
            results.append(record); details.append(record)
        if name == 'ttd_das':
            for fault in ['shape', 'nonfinite', 'carrier']:
                bad = x[0].copy()
                if fault == 'shape': bad = bad[:-1]
                elif fault == 'nonfinite': bad[0] = np.nan
                else: bad[1984] = 3
                rejected = False
                try: model.decide(bad, rng_for(0))
                except ValueError: rejected = True
                if not rejected: raise ValueError('未拒绝非法输入。')
                negative.append('reject_'+fault)
        if extra:
            rejected = False
            try: model.decide(x[0], rng_for(0))
            except ValueError: rejected = True
            if not rejected: raise ValueError('64次方法可在缺失反馈时运行。')
        model.verify()
        model_records.append(dict(method=name, artifacts=model.artifacts, cuda=model.uses_cuda,
            changed_cases=sum(r['changed_control_codes'] > 0 for r in results),
            repeated_first_case_equal=True))
        write_json(output/'progress.json', dict(status='running', completed_methods=ni+1, total_methods=len(names),
            method=name, pid=os.getpid(), at=now()))
        print(json.dumps(model_records[-1] | {'artifacts': len(model.artifacts)}), flush=True)
        del model; torch.cuda.empty_cache()
    verify_sources(sources)
    write_json(output/'summary.json', dict(status='passed_online_interface_preflight', models=model_records,
        cases=details, negative_checks=negative, numerical_files_read_by_online_entry=sorted(set(loaded)),
        methods=len(names), cases_count=len(details), source_sha256=sources,
        all_original_receiver_replays_matched=True, all_single_controls_match_batch=all(r['changed_control_codes'] == 0 for r in details),
        final_fair_timing=False, full_deployment_audit_complete=False, at=now()))
    write_json(output/'progress.json', dict(status='complete', methods=len(names), cases=len(details), at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try: run(a.project.resolve(), a.output.resolve())
    except BaseException:
        if a.output.exists(): write_json(a.output/('failure_%d.json' % time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
