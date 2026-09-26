"""验证在线计时确实加载声明的共同训练权重，再重算原始计时统计。"""
import argparse
import json
import os
from pathlib import Path
import sys
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE, require_host, sha256, verify_sources, write_json, now
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.confirmation_freeze import validate as validate_freeze
from study_full_baselines.audit_online_timing import audit as audit_original


def read(path):
    return json.loads(path.read_text())


def audit(project, output):
    require_host()
    project, output = project.resolve(), output.resolve()
    if (output/'integrity_verification.json').exists():
        raise FileExistsError('不覆盖已有共同模型包计时审计。')
    binding = read(output/'binding.json')
    if binding['schema'] != 'common-cohort-online-timing-v1' or Path(binding['project']) != project:
        raise ValueError('计时绑定格式或项目不同。')
    verify_sources(binding['source_sha256'])
    for name, digest in binding['source_sha256'].items():
        if sha256(output/'source_snapshot'/name) != digest:
            raise ValueError('计时连接源码快照改变。')
    bundle = Path(binding['runtime_bundle']).resolve()
    package = verify_bundle(bundle)
    if sha256(bundle/'manifest.json') != binding['runtime_manifest_sha256']:
        raise ValueError('权重包与计时绑定不同。')
    if package['cohort'] != binding['common_training_cohort'] or package['public_sha256'] != binding['public_sha256']:
        raise ValueError('共同训练成员或公开设置不同。')
    timing = output/'timing'
    protocol, summary = read(timing/'protocol.json'), read(timing/'summary.json')
    execution = read(output/'execution.json')
    if execution['status'] != 'complete_actual_timing' or execution['binding_sha256'] != sha256(output/'binding.json'):
        raise ValueError('实际执行没有绑定当前模型包。')
    if execution['timing_summary_sha256'] != sha256(timing/'summary.json'):
        raise ValueError('计时结果与执行完成记录不一致。')
    if execution['constructed_methods'] != binding['methods'] or protocol['methods'] != binding['methods']:
        raise ValueError('实际方法覆盖与声明不同。')
    if protocol['public_sha256'] != binding['public_sha256']:
        raise ValueError('计时输入配置与权重包不一致。')
    for key in ['preflight', 'final_fair_timing']:
        if protocol[key] != binding[key]:
            raise ValueError('计时范围标记不一致。')
    if binding['preflight']:
        if binding['final_fair_timing'] or binding['freeze_path'] is not None or package['cohort']['train_environments'] != 864:
            raise ValueError('旧案例演练被误标为正式计时。')
    else:
        freeze = Path(binding['freeze_path'])
        frozen = validate_freeze(freeze)
        if (not binding['final_fair_timing'] or sha256(freeze) != binding['freeze_sha256']
                or frozen['runtime_manifest_sha256'] != binding['runtime_manifest_sha256']
                or frozen['ordinary_methods'] != binding['methods']):
            raise ValueError('正式计时没有使用最终冻结的方法和权重。')
    allowed = {name: item['sha256'] for name, item in package['linked_files'].items()}
    allowed.update(package['generated_files'])
    checked = 0
    for model in summary['methods']:
        for filename, digest in model['model_artifacts'].items():
            path = Path(filename).resolve()
            if not path.is_relative_to(bundle):
                raise ValueError('模型从共同权重包之外加载了产物。')
            relative = str(path.relative_to(bundle))
            if allowed.get(relative) != digest or sha256(path) != digest:
                raise ValueError('实际权重或统计参数不在共同模型包内。')
            checked += 1
    for filename in summary['numerical_artifacts_loaded']:
        path = Path(filename).resolve()
        if not path.is_relative_to(bundle) or str(path.relative_to(bundle)) not in allowed:
            raise ValueError('在线阶段读取了模型包外数值文件。')
    audit_original(project, timing)
    original = read(timing/'integrity_verification.json')
    write_json(output/'integrity_verification.json', dict(status='passed_common_bundle_timing_audit',
        at=now(), binding_sha256=sha256(output/'binding.json'),
        execution_sha256=sha256(output/'execution.json'),
        original_audit_sha256=sha256(timing/'integrity_verification.json'),
        runtime_manifest_sha256=binding['runtime_manifest_sha256'],
        train_environments=package['cohort']['train_environments'],
        methods=len(binding['methods']), artifact_bindings_checked=checked,
        cases=original['cases'], timing_repetitions_checked=original['timing_repetitions_checked'],
        additional_query_rows_checked=original['additional_query_rows_checked'],
        final_fair_timing=binding['final_fair_timing'], auditor_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(status='passed_common_bundle_timing_audit', cases=original['cases'],
        methods=len(binding['methods']), artifact_bindings_checked=checked,
        final_fair_timing=binding['final_fair_timing'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    audit(args.project, args.input)
