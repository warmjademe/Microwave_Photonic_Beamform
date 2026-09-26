"""把同一训练群组的真实权重接入原在线算法；不复制测试预测。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (SOURCE, DEEP_METHODS, ONLINE_CLASSIC,
    check_data, load_split, require_host, sha256, source_record, verify_sources, write_json, now)
from study_full_baselines.fair_training_common import cohort_metadata
from study_full_baselines.build_fair_view import verified_source
from study_full_baselines.train import verify_labels
from study_full_baselines.online_controller import OnlineController, method_names
from our_method_quality_rank.common import METHODS as QUALITY_METHODS
from our_method_quality_rank.train import TrainingArray
from our_method_response_control.train import verify_targets
from deep_common.preprocessing import fit as fit_normalization
from baseline_frequency_prior.method import fit as fit_prior, predict_indices

GROUPS = ('direct', 'quality', 'response', 'real', 'linear')
LINEAR = ('per_tone_relative', 'joint_relative', 'joint_absolute')
LAYOUT = dict(direct='baseline_results/20260925_full_baselines/learned',
    quality='baseline_results/20260925_quality_rank_hybrid',
    response='baseline_results/20260925_response_control',
    real='baseline_results/20260925_full_baselines/real_response_cnn',
    linear='baseline_results/20260925_full_baselines/joint_linear')


def read(path): return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_training_protocol(group, protocol, expected_rows, data_digest):
    """只允许相同成员的最终权重，不把训练步数对照混入方法主比较。"""
    key = 'data_sha256' if group == 'linear' else 'data_manifest_sha256'
    if protocol.get(key) != data_digest: raise ValueError('训练协议与声明数据源不同。')
    ids = [r['environment_id'] for r in expected_rows]
    if 'train_ids' in protocol and protocol['train_ids'] != ids: raise ValueError('模型训练成员顺序不同。')
    if 'cohort' in protocol and protocol['cohort']['train_ids'] != ids: raise ValueError('模型群组成员不同。')
    if 'train_samples' in protocol and protocol['train_samples'] != len(ids)*17:
        raise ValueError('模型训练样本规模不同。')
    if 'train_environments' in protocol and protocol['train_environments'] != len(ids):
        raise ValueError('模型训练环境规模不同。')
    if group != 'linear' and (protocol.get('epochs') != 40 or protocol.get('seed') != 0):
        raise ValueError('神经网络必须使用固定40轮seed0。')
    if group == 'response' and protocol.get('schedule', 'fixed_epochs') != 'fixed_epochs':
        raise ValueError('不能把等更新步数模型充当固定40轮主比较。')
    if protocol.get('validation', False) or protocol.get('test_tuning', False):
        raise ValueError('不允许验证/测试选模。')


def runtime_sources():
    folders = ['deep_common', 'baseline_common', 'baseline_mlp', 'our_method_quality_rank',
        'our_method_response_control', 'baseline_response_realcnn', 'baseline_joint_response_linear',
        'our_method_two_stage', 'our_method_feedback_candidates', 'baseline_frequency_prior']
    folders += ['baseline_'+n for n in ONLINE_CLASSIC+DEEP_METHODS]
    names = [str(p.relative_to(SOURCE)) for folder in folders for p in (SOURCE/folder).glob('*.py')]
    names += ['study_full_baselines/'+n for n in ['runtime_bundle.py', 'RUNTIME_BUNDLE_PROTOCOL.md',
        'online_controller.py', 'common.py', 'fair_training_common.py', 'build_fair_view.py', 'train.py']]
    names += ['compact_dataset.py', 'native_sim/config.py', 'native_sim/control_engine.py']
    return source_record(names)


def build(data, spec, output):
    require_host(); data = Path(data).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError('模型包不覆盖原产物。')
    if set(spec) != set(GROUPS): raise ValueError('必须明确指定全部五组训练产物。')
    cohort = cohort_metadata(data); manifest = check_data(data)
    rows = [r for r in manifest['environments'] if r['split'] == 'train']
    _, expected_records = verified_source(data, rows)
    public_digest = sha256(data/'public.npz')
    # 先收集并验证全部链接计划；任何一组未完成，都不创建半成品模型包。
    links = {}; provenance = {}; checked_data = {str(data): manifest}; sources = runtime_sources()
    def add(source, relative, expected=None):
        source = Path(source).resolve(); digest = sha256(source)
        if expected is not None and expected != digest: raise ValueError('训练产物哈希不同：'+str(source))
        if relative in links: raise ValueError('模型包路径重复。')
        links[relative] = dict(source=str(source), sha256=digest, bytes=source.stat().st_size)
    for group in GROUPS:
        item = spec[group]; folder = Path(item['path']).resolve(); training = Path(item['training_data']).resolve()
        if str(training) not in checked_data:
            candidate = check_data(training)
            if [r for r in candidate['environments'] if r['split'] == 'train'] != rows:
                raise ValueError('混用了不同训练成员或顺序：'+group)
            _, actual_records = verified_source(training, rows)
            if actual_records != expected_records: raise ValueError('成员对应的实际样本改变：'+group)
            if sha256(training/'public.npz') != public_digest: raise ValueError('公开配置不同。')
            checked_data[str(training)] = candidate
        protocol = read(folder/'protocol.json')
        check_training_protocol(group, protocol, rows, sha256(training/'manifest.json'))
        verify_sources(protocol['source_sha256'])
        provenance[group] = dict(path=str(folder), training_data=str(training),
            training_manifest_sha256=sha256(training/'manifest.json'),
            training_records_sha256=sha256(training/'records.json'), protocol_sha256=sha256(folder/'protocol.json'))
        prefix = 'dataset_simulation/'+LAYOUT[group]
        add(folder/'protocol.json', prefix+'/protocol.json')
        if group in ['direct', 'quality']:
            if read(folder/'progress.json')['status'] != 'complete': raise ValueError('训练没有完成。')
            names = ['mlp']+DEEP_METHODS if group == 'direct' else list(QUALITY_METHODS)
            if protocol['methods'] != names: raise ValueError('模型集合不完整。')
            add(folder/'normalization.npz', prefix+'/normalization.npz')
            if group == 'direct':
                labels = Path(item['control_labels']).resolve(); label_protocol = verify_labels(training, labels)
                if label_protocol['data_manifest_sha256'] != sha256(training/'manifest.json'):
                    raise ValueError('控制标签没有绑定实际训练数据。')
                if protocol['labels_sha256'] != sha256(labels/'complete.json'): raise ValueError('控制标签来源改变。')
                provenance[group]['control_labels_complete_sha256'] = sha256(labels/'complete.json')
            for name in names:
                meta = read(folder/name/'complete.json')
                if (meta['epochs'], meta['seed']) != (40, 0): raise ValueError('模型轮数或seed改变。')
                if group == 'direct' and meta['fingerprint'] != fingerprint(protocol): raise ValueError('模型未绑定训练协议。')
                add(folder/name/'complete.json', prefix+'/'+name+'/complete.json')
                weight = 'model/weights.npz' if name == 'mlp' else 'weights.pt'
                add(folder/name/weight, prefix+'/'+name+'/'+weight, meta['weights_sha256'])
                if name == 'mlp':
                    cp = read(folder/name/'model/checkpoint.json')
                    if cp['metadata']['fingerprint'] != fingerprint(protocol): raise ValueError('MLP来源不同。')
                    if cp['settings']['training_rows'] != len(rows)*17: raise ValueError('MLP训练规模不同。')
                    add(folder/name/'model/checkpoint.json', prefix+'/'+name+'/model/checkpoint.json')
                    add(folder/name/'model/training_loss.json', prefix+'/'+name+'/model/training_loss.json', cp['training_loss_sha256'])
                elif 'history_sha256' in meta:
                    add(folder/name/'history.json', prefix+'/'+name+'/history.json', meta['history_sha256'])
        else:
            targets = Path(item['targets']).resolve(); verify_targets(training, targets)
            target_key = dict(response='targets_complete_sha256', real='target_sha256', linear='targets_sha256')[group]
            if protocol[target_key] != sha256(targets/'complete.json'): raise ValueError('响应监督来源改变。')
            provenance[group]['targets_complete_sha256'] = sha256(targets/'complete.json')
            meta = read(folder/'complete.json')
            if meta['status'] != 'complete': raise ValueError('模型尚未完成。')
            if 'fingerprint' in meta and meta['fingerprint'] != fingerprint(protocol): raise ValueError('训练协议指纹改变。')
            add(folder/'complete.json', prefix+'/complete.json')
            if group == 'linear':
                if [m['method'] for m in meta['records']] != list(LINEAR): raise ValueError('缺少线性基线。')
                for name, record in zip(LINEAR, meta['records']):
                    if read(folder/name/'complete.json') != record: raise ValueError('线性子模型来源不同。')
                    add(folder/name/'complete.json', prefix+'/'+name+'/complete.json')
                    for fc in range(4, 21):
                        weight = 'weights_%02d.npy' % fc
                        add(folder/name/weight, prefix+'/'+name+'/'+weight, record['weights_sha256'][str(fc)])
            else:
                if (meta['epochs'], meta['seed']) != (40, 0): raise ValueError('响应模型轮数或seed改变。')
                add(folder/'weights.pt', prefix+'/weights.pt', meta['weights_sha256'])
                if group == 'response':
                    target_prefix = 'dataset_simulation/diagnostics/20260925_response_control_targets/'
                    add(targets/'complete.json', target_prefix+'complete.json')
                    add(targets/'training_covariance.npy', target_prefix+'training_covariance.npy', read(targets/'complete.json')['covariance_sha256'])
    train_x, _, robust, train_rows = load_split(data, 'train', labels=True)
    if train_rows != rows: raise ValueError('训练数组顺序改变。')
    stats = fit_normalization(TrainingArray(train_x))
    for group in ['direct', 'quality']:
        with np.load(Path(spec[group]['path'])/'normalization.npz') as old:
            if set(old.files) != set(stats): raise ValueError('标准化字段不同。')
            for name in stats: np.testing.assert_array_equal(old[name], stats[name])
    prior = fit_prior(robust.reshape(len(rows), 17, 64))
    with np.load(data/'public.npz') as f: catalog = f['catalog_controls'].copy()
    indices = predict_indices(prior, np.arange(4, 21))
    fixed = [dict(name=name, kind='fixed', probes=0, controls=catalog[indices[:, i]].tolist())
        for i, name in enumerate(['global_prior', 'frequency_prior'])]
    add(data/'public.npz', 'public.npz', public_digest)
    output.mkdir(parents=True)
    for relative, record in links.items():
        source = Path(record['source']); dest = output/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        if sha256(source) != record['sha256']: raise ValueError('链接前来源改变。')
        os.link(source, dest)
    prior_path = 'dataset_simulation/baseline_results/20260925_full_baselines/learned_evaluation/protocol.json'
    (output/prior_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(output/prior_path, dict(bundle=dict(methods=fixed, prior_fit_training_only=True,
        train_ids=cohort['train_ids']), provenance='recomputed from common training robust candidate quality'))
    generated = {prior_path: sha256(output/prior_path)}
    for name, digest in sources.items():
        relative = 'source_snapshot/'+name; dest = output/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest); generated[relative] = digest
    verify_sources(sources)
    document = dict(schema='common-cohort-online-runtime-bundle-v1', status='complete', at=now(),
        cohort=cohort, common_data=str(data), common_manifest_sha256=sha256(data/'manifest.json'),
        public_sha256=public_digest, training_provenance=provenance, source_sha256=sources,
        methods=method_names(), linked_files=links, generated_files=generated,
        normalized_stats_recomputed=True, arrays_are_read_only_hardlinks=True,
        contains_cached_predictions=False, contains_training_labels=False, contains_test_environments=False,
        independent_confirmation_started=False)
    write_json(output/'manifest.json', document)
    verify(output)
    print(json.dumps(dict(status='runtime_bundle_complete', train_environments=len(rows),
        methods=len(document['methods']), linked_files=len(links), output=str(output))), flush=True)
    return document


def verify(output, public=None):
    """正式入口加载前验证整个包，拒绝夹带预测数组或来源发生变化。"""
    require_host(); output = Path(output).resolve(); document = read(output/'manifest.json')
    if document['schema'] != 'common-cohort-online-runtime-bundle-v1' or document['status'] != 'complete':
        raise ValueError('模型包未完成或格式错误。')
    if document['methods'] != method_names(): raise ValueError('在线方法集合改变。')
    expected = {'manifest.json'} | set(document['linked_files']) | set(document['generated_files'])
    found = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if found != expected: raise ValueError('模型包包含缺失或非白名单文件。')
    for relative, record in document['linked_files'].items():
        if sha256(output/relative) != record['sha256']: raise ValueError('模型包文件改变。')
        if not os.path.samefile(output/relative, record['source']): raise ValueError('模型包链接来源改变。')
    for relative, digest in document['generated_files'].items():
        if sha256(output/relative) != digest: raise ValueError('模型包元数据或源码快照改变。')
    verify_sources(document['source_sha256'])
    if public is not None:
        with np.load(output/'public.npz') as f:
            for name in ['pilot_qpsk', 'probe_controls', 'catalog_controls']:
                np.testing.assert_array_equal(f[name], public[name])
    return document


def load_models(output, public):
    """只改变产物根目录；每次控制仍执行原算法，不读取预测缓存。"""
    document = verify(output, public)
    models = {name: OnlineController(output, name, public) for name in document['methods']}
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'spec', 'output']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); build(args.data, read(args.spec), args.output)
