"""验证共享数据视图的成员、加载顺序、标签、预处理与硬链接，不重新训练模型。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.train import verify_labels
from study_full_baselines.build_fair_view import run as build_view
from our_method_quality_rank.train import TrainingArray
from deep_common.preprocessing import fit, apply
from baseline_common.data import features


def run(project, view, output):
    require_host()
    if output.exists(): raise FileExistsError('不覆盖原数据视图核查。')
    base = project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    manifest = check_data(view); original = check_data(base)
    identity = json.loads((view/'view_identity.json').read_text())
    if identity['train_source'] != str(base) or identity['test_source'] != str(base):
        raise ValueError('此前置核查只允许已完成864/216数据。')
    if manifest['environments'] != original['environments']:
        raise ValueError('视图改变了成员、分组或排列顺序。')
    source_hashes = {name: sha256(base/name) for name in ['manifest.json', 'records.json', 'public.npz']}
    report = json.loads((view/'view_report.json').read_text()); links = report['links']
    for link in links:
        target = view/link['view']; source = Path(link['source'])
        if not os.path.samefile(target, source) or sha256(target) != link['sha256']:
            raise ValueError('样本不是相同原文件。')
    if len(links) != 2160 or report['copied_sample_array_bytes'] != 0: raise ValueError('视图复制或缺少样本。')
    started = time.perf_counter()
    train, single, robust, train_rows = load_split(view, 'train', labels=True)
    old_train, old_single, old_robust, old_train_rows = load_split(base, 'train', labels=True)
    for a, b in [(train, old_train), (single, old_single), (robust, old_robust)]:
        np.testing.assert_array_equal(a, b)
    if train_rows != old_train_rows: raise ValueError('训练行顺序不一致。')
    test, _, _, test_rows = load_split(view, 'test', labels=False)
    old_test, _, _, old_test_rows = load_split(base, 'test', labels=False)
    np.testing.assert_array_equal(test, old_test)
    if test_rows != old_test_rows: raise ValueError('测试行顺序不一致。')
    labels = project/'dataset_simulation/outputs/unified_teacher_labels_20260925'
    label_protocol = verify_labels(view, labels)
    if label_protocol['train_ids'] != [r['environment_id'] for r in train_rows]: raise ValueError('控制标签顺序不同。')
    normalization = project/'dataset_simulation/baseline_results/20260925_full_baselines/learned/normalization.npz'
    with np.load(normalization) as f: saved_stats = {k: f[k].copy() for k in f.files}
    calculated = fit(TrainingArray(train))
    if set(calculated) != set(saved_stats): raise ValueError('预处理统计字段改变。')
    for key in calculated: np.testing.assert_array_equal(calculated[key], saved_stats[key])
    for left, right in [(train, old_train), (test, old_test)]:
        for start in range(0, len(left), 256):
            np.testing.assert_array_equal(apply(left[start:start+256], calculated), apply(right[start:start+256], saved_stats))
    public = public_data(view); feature_cases = 0
    for left, right in [(train, old_train), (test, old_test)]:
        for index in [0, 17, len(left)-1]:
            np.testing.assert_array_equal(features(observation(left[index], public)), features(observation(right[index], public)))
            feature_cases += 1
    output.mkdir(parents=True)
    negative = []
    try: load_split(view, 'test', labels=True)
    except ValueError: negative.append('reject_test_supervision_read')
    else: raise ValueError('测试标签读取没有拒绝。')
    try: build_view(base, view, output/'wrong_test_source')
    except ValueError: negative.append('reject_non_original_test_source')
    else: raise ValueError('非原测试源没有拒绝。')
    if (output/'wrong_test_source').exists(): raise ValueError('拒绝前创建了错误视图。')
    incomplete = output/'incomplete_source_negative'; incomplete.mkdir()
    write_json(incomplete/'manifest.json', dict(status='running'))
    try: build_view(incomplete, base, output/'incomplete_view')
    except ValueError: negative.append('reject_incomplete_training_source')
    else: raise ValueError('未完整训练源没有拒绝。')
    if (output/'incomplete_view').exists(): raise ValueError('拒绝前创建了不完整视图。')
    for name, digest in source_hashes.items():
        if sha256(base/name) != digest: raise ValueError('前置核查改变了源数据。')
    result = dict(status='passed_shared_cohort_view_preflight', train_environments=len(train_rows), test_environments=len(test_rows),
        train_samples=len(train), test_samples=len(test), all_x_and_train_quality_labels_bitwise_equal=True,
        original_control_label_members_and_hashes_verified=True, all_normalization_fields_bitwise_equal=True,
        all_preprocessed_inputs_bitwise_equal=True, mlp_feature_cases_equal=feature_cases,
        hardlinked_sample_files_checked=len(links), copied_sample_array_bytes=0, negative_checks=negative,
        network_training_started=False, new_signals_generated=0, source_data_unchanged=True,
        view_manifest_sha256=sha256(view/'manifest.json'), original_manifest_sha256=sha256(base/'manifest.json'),
        normalization_sha256=sha256(normalization), source_sha256=source_record([
            'study_full_baselines/build_fair_view.py', 'study_full_baselines/check_fair_view.py',
            'study_full_baselines/FAIR_COHORT_PROTOCOL.md', 'study_full_baselines/train.py',
            'our_method_quality_rank/train.py', 'deep_common/preprocessing.py']), seconds=time.perf_counter()-started, at=now())
    write_json(output/'summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'view', 'output']: parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(); run(args.project.resolve(), args.view.resolve(), args.output.resolve())
