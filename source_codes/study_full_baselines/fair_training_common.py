"""共同训练规模检查；仅核对已有数据，不生成新环境或测试监督标签。"""
from collections import Counter
import json
from pathlib import Path
from study_full_baselines.common import SOURCE, check_data, require_host, sha256

ALLOWED_COUNTS = (216, 432, 864, 1728, 3456)


def check_members(manifest, identity, original):
    """每个载频随所属传播环境整体划分；测试成员固定为已揭晓的旧216个。"""
    rows = manifest['environments']
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    count = len(train)
    if count not in ALLOWED_COUNTS or len(test) != 216 or len(rows) != count+216:
        raise ValueError('共同训练规模或原探索测试规模不符。')
    if test != [r for r in original['environments'] if r['split'] == 'test']:
        raise ValueError('禁止替换已揭晓的探索测试环境。')
    for group, key in [(train, 'train_ids'), (test, 'test_ids')]:
        if [r['index'] for r in group] != list(range(len(group))):
            raise ValueError('环境顺序改变。')
        if [r['environment_id'] for r in group] != identity[key]:
            raise ValueError('成员与共享视图身份不一致。')
    if len({r['environment_id'] for r in rows}) != len(rows):
        raise ValueError('环境ID重复。')
    if len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('种子重复或训练测试交叠。')
    strata = Counter(r['joint_stratum_id'] for r in train)
    if len(strata) != 216 or set(strata.values()) != {count//216}:
        raise ValueError('216个联合分层不平衡。')
    if manifest['train_samples'] != count*17 or manifest['test_samples'] != 216*17:
        raise ValueError('样本数量没有覆盖每环境全部17个载频。')
    return train, test


def cohort_metadata(data):
    require_host()
    data = Path(data).resolve(); manifest = check_data(data)
    if manifest.get('schema') != 'quality-rank-fair-comparison-view-v1':
        raise ValueError('可变规模训练必须使用已核验的共同数据视图。')
    if sha256(data/'view_identity.json') != manifest['view_identity_sha256']:
        raise ValueError('共同数据视图身份改变。')
    identity = json.loads((data/'view_identity.json').read_text())
    base = SOURCE.parent/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    if Path(identity['test_source']).resolve() != base.resolve():
        raise ValueError('测试源必须是原216环境。')
    if sha256(base/'manifest.json') != identity['test_manifest_sha256']:
        raise ValueError('原探索测试源改变。')
    original = check_data(base)
    train, test = check_members(manifest, identity, original)
    source = Path(identity['train_source'])
    if sha256(source/'manifest.json') != identity['train_manifest_sha256']:
        raise ValueError('训练源改变。')
    source_manifest = check_data(source)
    if train != [r for r in source_manifest['environments'] if r['split'] == 'train']:
        raise ValueError('训练视图成员与来源不同。')
    return dict(train_environments=len(train), train_samples=len(train)*17,
        test_environments=len(test), test_samples=len(test)*17,
        train_ids=[r['environment_id'] for r in train],
        test_ids=[r['environment_id'] for r in test],
        view_identity_sha256=manifest['view_identity_sha256'],
        training_source_manifest_sha256=identity['train_manifest_sha256'],
        exploratory_test_source_manifest_sha256=identity['test_manifest_sha256'],
        validation=False, test_scope='unchanged old216 exploratory; no fresh holdout')
