"""把完整训练成员和原探索测试集组成只读使用的数据视图；硬链接避免复制数组。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (require_host, check_data, sha256, source_record,
    verify_sources, write_json, now, SOURCE)


def verified_source(root, rows):
    manifest = check_data(root)
    records = {r['environment_id']: r for r in json.loads((root/'records.json').read_text())}
    result = {}
    for row in rows:
        record = records[row['environment_id']]
        if record['path'] != row['path'] or record['samples'] != 17:
            raise ValueError('源记录与成员清单不一致。')
        if set(record['file_sha256']) != {'data.npz', 'environment.json'}:
            raise ValueError('源环境文件清单改变。')
        for name, digest in record['file_sha256'].items():
            if sha256(root/row['path']/name) != digest: raise ValueError('源环境文件改变。')
        result[row['environment_id']] = record
    return manifest, result


def make_link(source, destination, digest):
    if sha256(source) != digest: raise ValueError('硬链接前来源文件改变。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination): raise ValueError('视图中存在不同来源的文件。')
    else: os.link(source, destination)
    if sha256(destination) != digest or not os.path.samefile(source, destination):
        raise ValueError('视图没有引用原文件。')


def run(train_root, test_root, output):
    require_host(); train_root = train_root.resolve(); test_root = test_root.resolve(); output = output.resolve()
    # 必须先确认生成完成；不能把部分扩展目录标成完整训练数据。
    train_manifest = check_data(train_root); test_manifest = check_data(test_root)
    train_rows = [r for r in train_manifest['environments'] if r['split'] == 'train']
    test_rows = [r for r in test_manifest['environments'] if r['split'] == 'test']
    if len(train_rows) not in [216, 432, 864, 1728, 3456] or len(test_rows) != 216:
        raise ValueError('本视图限已规划训练规模与原216环境探索集。')
    original_root = SOURCE.parent/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    if test_root != original_root.resolve(): raise ValueError('测试源必须是已揭晓的原216环境，禁止接入新留出。')
    for rows in [train_rows, test_rows]:
        if [r['index'] for r in rows] != list(range(len(rows))): raise ValueError('成员索引不连续。')
    train_seeds = {r['seed'] for r in train_rows}; test_seeds = {r['seed'] for r in test_rows}
    if len(train_seeds) != len(train_rows) or train_seeds & test_seeds: raise ValueError('重复环境或训练测试重叠。')
    counts = Counter(r['joint_stratum_id'] for r in train_rows)
    if len(counts) != 216 or set(counts.values()) != {len(train_rows)//216}:
        raise ValueError('训练联合分层不平衡。')
    public_digest = sha256(train_root/'public.npz')
    if public_digest != sha256(test_root/'public.npz') or train_manifest['core_source_sha256'] != test_manifest['core_source_sha256']:
        raise ValueError('训练/测试公开设置或器件核不同。')
    _, train_records = verified_source(train_root, train_rows)
    _, test_records = verified_source(test_root, test_rows)
    sources = dict(train_manifest['source_sha256'])
    for name, digest in test_manifest['source_sha256'].items():
        if name in sources and sources[name] != digest: raise ValueError('训练/测试生成源码版本冲突。')
        sources[name] = digest
    sources.update(source_record(['study_full_baselines/build_fair_view.py', 'study_full_baselines/FAIR_COHORT_PROTOCOL.md']))
    verify_sources(sources)
    identity = dict(train_source=str(train_root), test_source=str(test_root),
        train_manifest_sha256=sha256(train_root/'manifest.json'), test_manifest_sha256=sha256(test_root/'manifest.json'),
        train_ids=[r['environment_id'] for r in train_rows], test_ids=[r['environment_id'] for r in test_rows],
        public_sha256=public_digest, source_sha256=sources, linkage='hardlink, consumers read only')
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output/'view_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity: raise ValueError('视图来源改变。')
    else: write_json(identity_path, identity)
    make_link(train_root/'public.npz', output/'public.npz', public_digest)
    records = []; linked_bytes = 0; links = []
    for source, rows, index in [(train_root, train_rows, train_records), (test_root, test_rows, test_records)]:
        for row in rows:
            record = index[row['environment_id']]
            for name, digest in record['file_sha256'].items():
                original = source/row['path']/name; target = output/row['path']/name
                make_link(original, target, digest); linked_bytes += original.stat().st_size
                links.append(dict(view=str(target.relative_to(output)), source=str(original), sha256=digest))
            records.append(record)
    for name, digest in sources.items():
        dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256(dest) != digest: raise ValueError('视图已有不同源码快照。')
        else: shutil.copyfile(SOURCE/name, dest)
    write_json(output/'records.json', sorted(records, key=lambda r: r['environment_id']))
    manifest = {**train_manifest, 'schema': 'quality-rank-fair-comparison-view-v1', 'status': 'complete',
        'created_at_utc': now(), 'environments': train_rows+test_rows,
        'train_samples': len(train_rows)*17, 'test_samples': len(test_rows)*17,
        'source_sha256': sources, 'public_sha256': public_digest, 'records_sha256': sha256(output/'records.json'),
        'view_identity_sha256': sha256(identity_path), 'validation': False,
        'test_scope': 'unchanged old216 exploratory observations; no fresh holdout',
        'note': 'Read-only use of hard-linked existing samples. No new propagation, labels, train/test reassignment or array copies.'}
    write_json(output/'manifest.json', manifest)
    verify_sources(sources); check_data(output)
    report = dict(status='complete_read_only_dataset_view', train_environments=len(train_rows), test_environments=len(test_rows),
        train_samples=len(train_rows)*17, test_samples=len(test_rows)*17, sample_files_linked=len(links),
        linked_existing_bytes=linked_bytes, copied_sample_array_bytes=0, new_signals_generated=0,
        train_test_seed_overlap=0, identity_sha256=sha256(identity_path),
        manifest_sha256=sha256(output/'manifest.json'), links=links, at=now())
    write_json(output/'view_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'links'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--old-test-data', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.train_data, args.old_test_data, args.output)
