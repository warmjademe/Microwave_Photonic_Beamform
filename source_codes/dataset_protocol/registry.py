"""将既有数据明确登记为训练/验证/测试三组；不修改原始清单或重建大数组。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, write_json, now
from study_full_baselines.check_confirmation_plan import run as check_plan

SOURCES = {
    'train': ('outputs/scaling_train_3456_20260925/manifest.json', 'train', 3456,
              '8fbeb2c651ff1f03b356b25e63d9f0b93b488391423073f11f1d53d9afd7e9f6'),
    'validation': ('outputs/quality_rank_hybrid_20260925/manifest.json', 'test', 216,
                   '2d103743842132c0813c8cb565b80260bb5c9e35ca289f7e4a6e700c42b3f910'),
    'test': ('ops/full_baselines_20260925/fresh_confirmation_environment_plan.json', 'test', 864,
             '06c44a38ad6ac36a213671181574d415814cee6c7f26a5cb5885a94e3c3b8378'),
}


def build(project, output):
    require_host()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    root = project/'dataset_simulation'
    check_plan(project, output/'seed_exclusion_audit.json')
    groups = {}; sets = {}; files_checked = 0
    for role, (relative, old_split, count, digest) in SOURCES.items():
        path = root/relative
        if sha256(path) != digest: raise ValueError('来源身份改变：'+role)
        original = json.loads(path.read_text())
        rows = [r for r in original['environments'] if r['split'] == old_split]
        if len(rows) != count or len({r['seed'] for r in rows}) != count:
            raise ValueError('环境数量或种子不符：'+role)
        counts = Counter(r['joint_stratum_id'] for r in rows)
        if len(counts) != 216 or set(counts.values()) != {count//216}:
            raise ValueError('分层采样不符：'+role)
        converted = []
        records = {} if role == 'test' else {
            r['environment_id']: r for r in json.loads((path.parent/'records.json').read_text())}
        if role != 'test' and original['status'] != 'complete': raise ValueError('来源未完成')
        for row in rows:
            item = dict(row, split=role, source_split=old_split)
            if role != 'test':
                record = records[row['environment_id']]
                if record['samples'] != 17: raise ValueError('载频记录数改变')
                for name, expected in record['file_sha256'].items():
                    if sha256(path.parent/row['path']/name) != expected:
                        raise ValueError('原始数据改变：'+row['environment_id'])
                    files_checked += 1
                item.update(source_directory=str(path.parent/row['path']),
                            file_sha256=record['file_sha256'])
            converted.append(item)
        sets[role] = {r['seed'] for r in rows}
        groups[role] = dict(environments=count, records=count*17, members_per_stratum=count//216,
            source_manifest=str(path), source_sha256=digest, source_split=old_split,
            status='planned_not_generated' if role == 'test' else 'complete', rows=converted)
    overlap = {a+'_'+b: len(sets[a] & sets[b]) for a,b in
               [('train','validation'),('train','test'),('validation','test')]}
    if any(overlap.values()): raise ValueError('划分存在重叠')
    result = dict(schema='mwp-dataset-split-3456-216-864-v1', at=now(),
        carriers_ghz=list(range(4,21)), split_unit='independent propagation environment',
        joint_strata=216, pairwise_seed_overlap=overlap, verified_data_files=files_checked,
        roles={'train':'fit weights and training statistics',
               'validation':'method design and selection; historically named test',
               'test':'frozen final comparison; never used for model selection'}, splits=groups)
    write_json(output/'registry.json', result)
    write_json(output/'complete.json', dict(status='passed', at=now(),
        registry_sha256=sha256(output/'registry.json'), verified_data_files=files_checked,
        pairwise_seed_overlap=overlap, counts={k:v['environments'] for k,v in groups.items()}))
    print(json.dumps({'status':'passed', 'counts':{k:len(v) for k,v in sets.items()}, 'overlap':overlap}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); build(a.project.resolve(),a.output.resolve())
