"""旧864共同成员模型包核查：实际权重执行、严格反馈重放及输入边界。"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import (SOURCE, require_host, check_data, sha256,
    write_json, now, rng_for, source_record, verify_sources)
from study_full_baselines.runtime_bundle import (build, verify, load_models, read,
    check_training_protocol)
from study_full_baselines.benchmark_online import TraceReplay
from our_method_response_control.train import precision


def run(project, view, output):
    require_host(); precision()
    if output.exists(): raise FileExistsError('不覆盖在线模型包前置证据。')
    root = project/'dataset_simulation'; base = root/'outputs/quality_rank_hybrid_20260925'
    study = root/'baseline_results/20260925_full_baselines'
    manifest = check_data(view); original = check_data(base)
    if manifest['environments'] != original['environments']:
        raise ValueError('本前置检查只使用已完成的原864/216环境。')
    target = root/'diagnostics/20260925_response_control_targets'
    folders = dict(direct=study/'learned', quality=root/'baseline_results/20260925_quality_rank_hybrid',
        response=root/'baseline_results/20260925_response_control', real=study/'real_response_cnn',
        linear=study/'joint_linear')
    spec = {name:dict(path=str(folder), training_data=str(base)) for name, folder in folders.items()}
    spec['direct']['control_labels'] = str(root/'outputs/unified_teacher_labels_20260925')
    for name in ['response', 'real', 'linear']: spec[name]['targets'] = str(target)
    output.mkdir(parents=True); write_json(output/'spec.json', spec)
    document = build(view, spec, output/'bundle')
    reference = root/'diagnostics/20260926_online_controller_preflight'
    reference_summary = read(reference/'summary.json')
    ref = {(r['method'], r['carrier_ghz']):r for r in reference_summary['cases']}
    row = next(r for r in original['environments'] if r['split']=='test' and r['index']==0)
    records = {r['environment_id']:r for r in read(base/'records.json')}
    for name, digest in records[row['environment_id']]['file_sha256'].items():
        if sha256(base/row['path']/name) != digest: raise ValueError('固定旧案例改变。')
    with np.load(base/row['path']/'data.npz') as f: raw = f['X'].copy()
    with np.load(base/'public.npz') as f:
        public = {k:f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    arrays = {}; reference_hashes = {}
    for method in document['methods']:
        for fc in [4, 12, 20]:
            item = ref[method, fc]; path = reference/item['result_file']
            if sha256(path) != item['result_sha256']: raise ValueError('原在线输出参考改变。')
            reference_hashes[str(path)] = item['result_sha256']
            with np.load(path) as f: arrays[method, fc] = {k:f[k].copy() for k in f.files}
    original_load = np.load; loaded = []
    def guard(path, *args, **kwargs):
        if not isinstance(path, (str, Path)): raise ValueError('在线数组必须来自显式模型包文件。')
        p = Path(path).resolve()
        if not p.is_relative_to((output/'bundle').resolve()): raise ValueError('在线模型试图从包外加载数组。')
        if p.name in ['data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy', 'train_Y.npy']:
            raise ValueError('在线模型试图读取预测或标签。')
        loaded.append(str(p.relative_to(output/'bundle')))
        return original_load(path, *args, **kwargs)
    negative = []
    with patch('numpy.load', guard):
        models = load_models(output/'bundle', public)
        for name in ['data.npz', 'predictions.npy', 'train_response.npy']:
            try: np.load(base/name)
            except ValueError: negative.append('reject_online_read_'+name)
            else: raise ValueError('在线读取边界没有拒绝。')
    results = []; queries = 0
    for method, model in models.items():
        for fc in [4, 12, 20]:
            saved = arrays[method, fc]; callback = None
            if ref[method, fc]['feedback_calls'] == 64:
                callback = TraceReplay(saved['trace_control_code'][16:], saved['trace_scores'][16:])
            with patch('numpy.load', guard):
                got = model.decide(raw[fc-4], rng_for(0, row['seed'], fc, 630), callback)
            np.testing.assert_array_equal(got['control_code'], saved['control_code'][1])
            if callback is not None:
                callback.finish(); queries += callback.position
            if 'trace_control_code' in saved:
                for key in ['trace_control_code', 'trace_scores']:
                    np.testing.assert_array_equal(got[key], saved[key])
            if 'estimated_response' in saved:
                np.testing.assert_array_equal(got['estimated_response'], saved['estimated_response'])
            if got['feedback_calls'] != ref[method, fc]['feedback_calls']: raise ValueError('实际预算改变。')
            results.append(dict(method=method, carrier_ghz=fc, control_codes_exact=True,
                feedback_calls=got['feedback_calls'], feedback_trace_exact='trace_control_code' in saved,
                reference_sha256=ref[method, fc]['result_sha256']))
        model.verify()
    rows = [r for r in original['environments'] if r['split']=='train']
    protocol = read(folders['direct']/'protocol.json'); digest = sha256(base/'manifest.json')
    for name in ['member_order', 'train_count', 'data_hash', 'epochs', 'seed']:
        bad = copy.deepcopy(protocol)
        if name=='member_order': bad['train_ids'][:2] = reversed(bad['train_ids'][:2])
        elif name=='train_count': bad['train_samples'] += 17
        elif name=='data_hash': bad['data_manifest_sha256'] = '0'*64
        elif name=='epochs': bad['epochs'] = 39
        else: bad['seed'] = 1
        try: check_training_protocol('direct', bad, rows, digest)
        except ValueError: negative.append('reject_'+name)
        else: raise ValueError('训练来源错误未拒绝：'+name)
    bad = {**read(folders['response']/'protocol.json'), 'schedule':'equal_updates'}
    try: check_training_protocol('response', bad, rows, digest)
    except ValueError: negative.append('reject_equal_updates_as_main_comparison')
    else: raise ValueError('等更新对照被混入主比较。')
    stray = output/'bundle/predictions.npy'; stray.write_bytes(b'boundary-test')
    try:
        try: verify(output/'bundle')
        except ValueError: negative.append('reject_unlisted_bundle_file')
        else: raise ValueError('模型包非白名单文件没有拒绝。')
    finally: stray.unlink()
    verify(output/'bundle', public)
    sources = source_record(['study_full_baselines/check_runtime_bundle.py',
        'study_full_baselines/benchmark_online.py'])
    for name in sources:
        dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    verify_sources(sources)
    result = dict(status='passed_runtime_bundle_preflight', at=now(), methods=len(models), cases=len(results),
        control_values_checked=len(results)*128, replayed_additional_feedback_queries=queries,
        train_environments=864, all_actual_single_controls_exact=True,
        all_existing_estimated_responses_exact=True, all_feedback_traces_exact=True,
        common_training_sources_checked=True, negative_checks=negative,
        new_signal_simulations=0, new_models_trained=0, fresh_holdout_revealed=False,
        final_fair_timing=False, source_sha256=sources, bundle_manifest_sha256=sha256(output/'bundle/manifest.json'),
        reference_summary_sha256=sha256(reference/'summary.json'), reference_files=reference_hashes,
        numerical_files_read=sorted(set(loaded)), results=results)
    write_json(output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['results','reference_files','numerical_files_read']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'view', 'output']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); run(args.project.resolve(), args.view.resolve(), args.output.resolve())
