"""重建留出环境清单并核对已用/计划训练种子；只操作元数据，不生成传播或信号。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, write_json, now, source_record
from generate_dataset import plan_environments


def run(project, output):
    require_host()
    if output.exists(): raise FileExistsError('不覆盖原留出计划核查。')
    root = project/'dataset_simulation'
    path = root/'ops/full_baselines_20260925/fresh_confirmation_environment_plan.json'
    expected_sha = '06c44a38ad6ac36a213671181574d415814cee6c7f26a5cb5885a94e3c3b8378'
    if sha256(path) != expected_sha: raise ValueError('原留出计划身份改变。')
    saved = json.loads(path.read_text())
    if saved['status'] != 'plan_only_signals_not_generated' or saved['master_seed'] != 2026092509:
        raise ValueError('不是原预设未解封计划。')
    rebuilt, designs = plan_environments(216, 864, 2026092509, 'joint_stratified')
    rebuilt = [r for r in rebuilt if r['split'] == 'test']
    if saved['environments'] != rebuilt or saved['sampling_design'] != designs['test']:
        raise ValueError('原种子及采样器未重现完整留出清单。')
    rows = saved['environments']; seeds = {r['seed'] for r in rows}
    if len(rows) != 864 or len(seeds) != 864 or saved['samples'] != 864*17:
        raise ValueError('新测试数量或独立种子数量不符。')
    if saved['carriers_ghz'] != list(range(4, 21)): raise ValueError('载频范围不符。')
    counts = Counter(r['joint_stratum_id'] for r in rows)
    if len(counts) != 216 or set(counts.values()) != {4}: raise ValueError('联合分层未保持每层4个成员。')
    for stratum in counts:
        if sorted(r['stratum_repeat_index'] for r in rows if r['joint_stratum_id'] == stratum) != list(range(4)):
            raise ValueError('层内重复编号不符。')
    identities = {}; groups = {}
    for split in ['train', 'test']:
        legacy = root/('dataset_'+split)/'environments.json'
        identities[str(legacy)] = sha256(legacy)
        groups['legacy_'+split] = {int(r['environment_id'].split('-')[-1]) for r in json.loads(legacy.read_text())}
    base = root/'outputs/quality_rank_hybrid_20260925/manifest.json'
    original = json.loads(base.read_text()); identities[str(base)] = sha256(base)
    for split in ['train', 'test']:
        groups['current_'+split] = {r['seed'] for r in original['environments'] if r['split'] == split}
    # 同时排除尚在生成或尚未启动的训练扩展成员，不能只看当前完成的目录。
    for count, master in [(864, 2026092507), (1728, 2026092508)]:
        planned, _ = plan_environments(count, 216, master, 'joint_stratified')
        groups['additional_train_'+str(master)] = {r['seed'] for r in planned if r['split'] == 'train'}
    excluded = set().union(*groups.values())
    if len(excluded) != saved['excluded_known_seed_count'] or seeds & excluded:
        raise ValueError('留出种子与已用/计划训练成员重叠或排除清单改变。')
    actual_expansions = []
    for count in [1728, 3456]:
        manifest = root/('outputs/scaling_train_%d_20260925/manifest.json' % count)
        if not manifest.exists():
            actual_expansions.append(dict(train_count=count, status='not_created_yet', prospective_seeds_checked=True))
            continue
        actual = json.loads(manifest.read_text()); actual_seeds = {r['seed'] for r in actual['environments']}
        wanted = groups['current_train'] | groups['additional_train_2026092507']
        if count == 3456: wanted |= groups['additional_train_2026092508']
        if actual_seeds != wanted or len(actual['environments']) != count or any(r['split'] != 'train' for r in actual['environments']):
            raise ValueError('真实扩展清单与已排除的训练计划不同。')
        actual_expansions.append(dict(train_count=count, status=actual['status'], planned_members_equal=True,
            manifest_path=str(manifest), manifest_sha256_at_check=sha256(manifest)))
    report = dict(status='passed_plan_and_seed_exclusion_audit', independent_test_environments=864,
        joint_strata=216, members_per_stratum=4, carriers=17, total_records=14688,
        plan_exactly_rebuilt=True, fresh_plan_sha256=expected_sha, excluded_unique_seeds=len(excluded),
        overlap=0, exclusion_group_counts={k: len(v) for k, v in groups.items()},
        metadata_sha256=identities, expansion_manifest_checks=actual_expansions,
        new_environment_signals_generated=0, final_method_frozen=False, final_confirmation_complete=False,
        source_sha256=source_record(['study_full_baselines/check_confirmation_plan.py', 'generate_dataset.py',
            'baseline_common/channel.py']), at=now())
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in
        ['source_sha256', 'metadata_sha256', 'expansion_manifest_checks']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); run(args.project.resolve(), args.output.resolve())
