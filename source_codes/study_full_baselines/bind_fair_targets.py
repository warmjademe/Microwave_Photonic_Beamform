"""为共享数据视图绑定相同训练成员的响应标签；不重新计算或改写标签数组。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, check_data, sha256, write_json, now, source_record, verify_sources
from study_full_baselines.build_fair_view import make_link
from our_method_response_control.train import verify_targets


def run(source_data, source_targets, view, output):
    require_host()
    source_manifest = check_data(source_data); view_manifest = check_data(view)
    verify_targets(source_data, source_targets)
    identity = json.loads((view/'view_identity.json').read_text())
    if identity['train_source'] != str(source_data.resolve()) or identity['train_manifest_sha256'] != sha256(source_data/'manifest.json'):
        raise ValueError('标签来源不是共享视图的训练源。')
    source_train = [r for r in source_manifest['environments'] if r['split'] == 'train']
    view_train = [r for r in view_manifest['environments'] if r['split'] == 'train']
    if source_train != view_train: raise ValueError('训练成员、顺序或传播配置不同。')
    old_protocol = json.loads((source_targets/'protocol.json').read_text())
    old_complete = json.loads((source_targets/'complete.json').read_text())
    binding = dict(schema='same-training-response-target-view-v1', source_data=str(source_data),
        source_data_manifest_sha256=sha256(source_data/'manifest.json'), source_targets=str(source_targets),
        source_target_protocol_sha256=sha256(source_targets/'protocol.json'),
        source_target_complete_sha256=sha256(source_targets/'complete.json'),
        view_manifest_sha256=sha256(view/'manifest.json'), training_rows_equal=True,
        train_environment_ids=[r['environment_id'] for r in view_train],
        source_sha256=source_record(['study_full_baselines/bind_fair_targets.py',
            'study_full_baselines/build_fair_view.py', 'our_method_response_control/train.py']))
    output.mkdir(parents=True, exist_ok=True)
    if (output/'binding.json').exists():
        if json.loads((output/'binding.json').read_text()) != binding: raise ValueError('目标引用来源改变。')
    else: write_json(output/'binding.json', binding)
    for name, key in [('train_response.npy', 'label_sha256'), ('training_covariance.npy', 'covariance_sha256')]:
        make_link(source_targets/name, output/name, old_complete[key])
    # 旧校验器要求整个输入目录计划一致；记录新清单的同时明确绑定原训练目标证据。
    protocol = {**old_protocol, 'source_environment_plan_sha256': hashlib.sha256(
        json.dumps(view_manifest['environments'], sort_keys=True).encode()).hexdigest(),
        'data_manifest_sha256': sha256(view/'manifest.json'), 'target_view_binding_sha256': sha256(output/'binding.json')}
    write_json(output/'protocol.json', protocol)
    complete = dict(status='complete', samples=len(view_train)*17,
        label_sha256=old_complete['label_sha256'], covariance_sha256=old_complete['covariance_sha256'],
        target_arrays_recomputed=False, source_target_complete_sha256=sha256(source_targets/'complete.json'),
        target_view_binding_sha256=sha256(output/'binding.json'), binding_created_at=now())
    write_json(output/'complete.json', complete)
    verify_targets(view, output); verify_targets(source_data, source_targets); verify_sources(binding['source_sha256'])
    print(json.dumps(dict(status='passed_identical_training_target_binding', train_environments=len(view_train),
        labels_sha256=old_complete['label_sha256'], covariance_sha256=old_complete['covariance_sha256'],
        recomputed_target_arrays=0, copied_target_array_bytes=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-data', 'source-targets', 'view', 'output']: parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(); run(args.source_data.resolve(), args.source_targets.resolve(), args.view.resolve(), args.output.resolve())
