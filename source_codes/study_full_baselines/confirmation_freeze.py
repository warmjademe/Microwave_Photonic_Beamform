"""最终确认启动前冻结：只读旧证据和训练产物，不生成新传播环境。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE, ONLINE_CLASSIC, require_host, sha256, source_record, verify_sources, write_json, now
from study_full_baselines.runtime_bundle import verify as verify_bundle, method_names, fingerprint
from study_full_baselines.fair_training_common import cohort_metadata
from study_full_baselines.check_confirmation_plan import run as check_plan
from study_full_baselines.confirmation_receiver import METRIC_ORDER

FRESH_SHA = '06c44a38ad6ac36a213671181574d415814cee6c7f26a5cb5885a94e3c3b8378'
METRICS = ['ber', 'ser', 'block_error_rate', 'rms_evm_percent', 'paired_output_snr_db']


def read(path): return json.loads(Path(path).read_text())


def budget(name):
    if name in ['global_prior', 'frequency_prior']: return 0
    if name in ONLINE_CLASSIC[1:] or name.endswith('_warm64'): return 64
    return 16


def statistics_plan(selection):
    if set(selection) != {'primary16', 'primary64', 'rationale'} or not str(selection['rationale']).strip():
        raise ValueError('必须明确最终两种预算的方法及基于旧探索结果的选择依据。')
    if selection['primary16'] not in ['complex_response_cnn', 'cnn__multi_mmse', 'cnn__single_10sweeps']:
        raise ValueError('最终16次方法必须是既定复响应路线的已实现变体。')
    if selection['primary64'] != 'cnn_warm64': raise ValueError('64次主方法限定已实现的CNN反馈控制。')
    comparisons = []
    def add(name, family, terms, meaning):
        if set(terms)-set(method_names()) or sum(terms.values()) != 0: raise ValueError('比较的方法或系数错误。')
        comparisons.append(dict(id=name, family=family, terms=terms, meaning=meaning))
    for probes, key in [(16, 'primary16'), (64, 'primary64')]:
        own = selection[key]
        for other in method_names():
            if other != own and budget(other) == probes:
                add('primary%d_vs_%s'%(probes,other), 'RQ1_budget_%d'%probes,
                    {own:1,other:-1}, 'Selected method minus comparator with the same measurement budget')
    for other in ['global_prior','frequency_prior']:
        add('primary16_vs_'+other, 'RQ1_fixed_prior', {selection['primary16']:1,other:-1},
            '16-probe controller minus zero-probe training prior; different measurement budgets')
    a0b0, a1b0, a0b1, a1b1 = 'covariance_response', 'complex_response_cnn', 'covariance_warm64', 'cnn_warm64'
    for label, terms in [
        ('estimator_at16',{a1b0:1,a0b0:-1}), ('estimator_at64',{a1b1:1,a0b1:-1}),
        ('feedback_with_covariance',{a0b1:1,a0b0:-1}), ('feedback_with_cnn',{a1b1:1,a1b0:-1}),
        ('interaction',{a1b1:1,a0b1:-1,a1b0:-1,a0b0:1})]:
        add(label,'RQ2_two_stages',terms,'Covariance/CNN estimator × base/64-probe feedback; feedback changes measurement budget')
    add('complex_vs_real','RQ2_estimator',{a1b0:1,'real_response_cnn':-1},'Parameter-matched real and complex response models')
    for base, alias in [('covariance_response','covariance'),('complex_response_cnn','cnn')]:
        for variant in ['multi_mmse','single_10sweeps']:
            add(alias+'_'+variant,'RQ2_controller',{alias+'__'+variant:1,base:-1},'Same 16 probes, changed internal search')
    return dict(schema='fresh864-confirmation-statistics-v1', primary_metric='ber', metrics=METRICS,
        bootstrap_repetitions=10000, bootstrap_seed=20260926, stratified=True, confidence_level=.95,
        statistical_unit='independent propagation environment, keeping all17 carriers together',
        multiplicity='BH-FDR within each predeclared family across all its metrics',
        sign_test_estimand='environment win/loss probability, not mean effect',
        training_seed=0, selection=selection, comparisons=comparisons,
        reference_methods=['teacher','mrc'], references_descriptive_only=True,
        groups=['carrier_ghz','rays','max_delay_ns','angular_std_deg','power_bin'],
        fixed_signal_example_index=0, signal_example_carriers=list(range(4,21)))


def checked_file(path, files):
    path = Path(path)
    if not path.is_file(): raise ValueError('最终确认前置文件尚未完成：'+str(path))
    files[str(path)] = sha256(path)
    return read(path)


def prerequisites(project):
    root = project/'dataset_simulation'; study = root/'baseline_results/20260925_full_baselines'; files = {}
    for phase in ['classic','learned_evaluation','evaluation_components','evaluation_real_response_cnn',
                  'evaluation_scaling_1728','evaluation_scaling_3456','evaluation_feedback_warm']:
        value = checked_file(study/phase/'progress.json', files)
        if value.get('status') != 'complete' or value.get('completed') != 216:
            raise ValueError('原216环境的完整探索评分尚未结束：'+phase)
    for name in ['analysis','analysis_feedback_warm']:
        folder = study/name; complete = checked_file(folder/'complete.json', files)
        if complete['status'] != 'complete': raise ValueError('探索统计尚未完成。')
        verify_sources(complete['source_sha256'])
        for filename, digest in complete['file_sha256'].items():
            path = folder/filename
            if sha256(path) != digest: raise ValueError('探索统计原始产物改变。')
            files[str(path)] = digest
    receiver = root/'diagnostics/20260926_confirmation_receiver_preflight'
    proof = checked_file(receiver/'summary.json', files); audit = checked_file(receiver/'integrity_verification.json', files)
    if (proof['cases_count'] != 135 or not proof['all_controls_equal'] or not proof['all_inputs_bitwise_equal']
            or not proof['cpu_gpu_scoring_passed'] or audit['summary_sha256'] != sha256(receiver/'summary.json')):
        raise ValueError('完整接收器前置核查未通过。')
    verify_sources(proof['source_sha256'])
    for path, digest in proof['result_and_reference_sha256'].items():
        if sha256(path) != digest: raise ValueError('接收器核查来源改变。')
        files[path] = digest
    replay = root/'diagnostics/20260926_confirmation_batch_rehearsal'
    r = checked_file(replay/'integrity_verification.json', files)
    if r['status'] != 'passed_parallel_resume_and_records_audit' or r['cases'] != 405:
        raise ValueError('并行与续跑演练未通过。')
    sources = dict(proof['source_sha256'])
    sources.update(source_record(['study_full_baselines/'+n for n in ['confirmation_freeze.py',
        'confirmation_execute.py','FINAL_CONFIRMATION_PROTOCOL.md','runtime_bundle.py',
        'check_confirmation_plan.py','paired_statistics.py','confirmation_batch.py']]))
    return files, sources


def prepare(project, bundle, selection, output):
    require_host(); project = project.resolve(); bundle = bundle.resolve(); output = output.resolve()
    if output.exists(): raise FileExistsError('冻结文件不覆盖既有版本。')
    # 首先拒绝前序尚未完成的状态；此处不会调用任何传播或波形生成函数。
    evidence, sources = prerequisites(project)
    package = verify_bundle(bundle); cohort = cohort_metadata(Path(package['common_data']))
    if cohort != package['cohort']: raise ValueError('运行包与当前训练成员不同。')
    if sha256(Path(package['common_data'])/'manifest.json') != package['common_manifest_sha256']:
        raise ValueError('运行包共同数据来源改变。')
    sources.update(package['source_sha256']); plan = statistics_plan(selection)
    fresh = project/'dataset_simulation/ops/full_baselines_20260925/fresh_confirmation_environment_plan.json'
    if sha256(fresh) != FRESH_SHA: raise ValueError('预设新计划改变。')
    saved = read(fresh); rows = saved['environments']
    if (len(rows) != 864 or saved['carriers_ghz'] != list(range(4,21))
            or len(Counter(r['joint_stratum_id'] for r in rows)) != 216):
        raise ValueError('新测试范围错误。')
    if len({r['seed'] for r in rows}) != 864 or set(Counter(r['joint_stratum_id'] for r in rows).values()) != {4}:
        raise ValueError('新测试种子或分层不同。')
    output.mkdir(parents=True)
    check_plan(project, output/'seed_exclusion.json')
    for name in sources:
        dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE/name,dest)
    identity = dict(schema='frozen-fresh864-confirmation-v1', scope='fresh_confirmation', frozen_at=now(),
        project=str(project), runtime_bundle=str(bundle), runtime_manifest_sha256=sha256(bundle/'manifest.json'),
        common_training_cohort=cohort, common_data_manifest_sha256=package['common_manifest_sha256'],
        fresh_plan_path=str(fresh), fresh_plan_sha256=FRESH_SHA, rows=rows, carriers=list(range(4,21)),
        ordinary_methods=method_names(), methods=method_names()+['teacher','mrc'],
        metric_order=METRIC_ORDER,
        measurement_budgets={name:budget(name) for name in method_names()}, statistics=plan,
        source_sha256=sources, prerequisite_files=evidence, seed_exclusion_sha256=sha256(output/'seed_exclusion.json'),
        backend='cuda_fft_cpu_rk4_v1', final_confirmation=True,
        new_environment_signals_generated_at_freeze=0)
    verify_sources(sources); write_json(output/'freeze.json', identity)
    print(json.dumps(dict(status='frozen_before_new_signals', freeze_sha256=sha256(output/'freeze.json'),
        train_environments=cohort['train_environments'], test_environments=864, methods=len(identity['methods']),
        comparisons=len(plan['comparisons']))), flush=True)


def validate(freeze):
    require_host(); freeze = Path(freeze).resolve(); identity = read(freeze)
    if identity.get('schema') != 'frozen-fresh864-confirmation-v1' or identity.get('scope') != 'fresh_confirmation':
        raise ValueError('不是正式冻结版本。')
    if identity['ordinary_methods'] != method_names() or identity['methods'] != method_names()+['teacher','mrc']:
        raise ValueError('方法集合或顺序改变。')
    if identity['metric_order'] != METRIC_ORDER: raise ValueError('接收指标定义改变。')
    if identity['statistics'] != statistics_plan(identity['statistics']['selection']): raise ValueError('统计清单改变。')
    if identity['measurement_budgets'] != {name:budget(name) for name in method_names()}:
        raise ValueError('测量预算改变。')
    if identity['backend'] != 'cuda_fft_cpu_rk4_v1' or not identity['final_confirmation']:
        raise ValueError('正式接收后端或范围改变。')
    expected_evidence, expected_sources = prerequisites(Path(identity['project']))
    if expected_evidence != identity['prerequisite_files']:
        raise ValueError('冻结清单没有完整绑定前置证据。')
    if any(identity['source_sha256'].get(k) != v for k,v in expected_sources.items()):
        raise ValueError('冻结清单缺少必要的接收源码。')
    fresh = Path(identity['fresh_plan_path'])
    if sha256(fresh) != FRESH_SHA or identity['fresh_plan_sha256'] != FRESH_SHA:
        raise ValueError('新计划身份不同。')
    if identity['rows'] != read(fresh)['environments'] or identity['carriers'] != list(range(4,21)):
        raise ValueError('测试成员或频点改变。')
    for path, digest in identity['prerequisite_files'].items():
        if sha256(path) != digest: raise ValueError('已冻结前置证据改变。')
    if sha256(freeze.parent/'seed_exclusion.json') != identity['seed_exclusion_sha256']:
        raise ValueError('种子隔离证据改变。')
    verify_sources(identity['source_sha256'])
    for name, digest in identity['source_sha256'].items():
        if sha256(freeze.parent/'source_snapshot'/name) != digest: raise ValueError('源码快照改变。')
    bundle = Path(identity['runtime_bundle']); package = verify_bundle(bundle)
    if sha256(bundle/'manifest.json') != identity['runtime_manifest_sha256'] or package['cohort'] != identity['common_training_cohort']:
        raise ValueError('最终运行权重或训练成员改变。')
    return identity


def register(freeze, output):
    """同一预设新计划只登记一个冻结身份；登记先于第一次信号生成。"""
    identity = validate(freeze)
    path = Path(identity['project'])/'dataset_simulation/ops/full_baselines_20260925/fresh_confirmation_claim.json'
    claim = dict(freeze_path=str(Path(freeze).resolve()), freeze_sha256=sha256(freeze),
        identity_sha256=fingerprint(identity), output=str(Path(output).resolve()), fresh_plan_sha256=FRESH_SHA)
    if path.exists():
        if read(path) != claim: raise ValueError('该新计划已登记其他冻结版本或输出目录，不能重新挑选。')
    else:
        with path.open('x') as stream: stream.write(json.dumps(claim,sort_keys=True,indent=2)+'\n')
    return identity


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project','bundle','selection','output']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); prepare(a.project,a.bundle,read(a.selection),a.output)
