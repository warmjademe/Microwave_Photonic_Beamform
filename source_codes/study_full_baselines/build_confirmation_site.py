"""绑定冻结方案、完整接收统计、实际计时和信号回放，构建独立确认网站。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE, require_host, sha256, verify_sources, source_record, write_json, now
from study_full_baselines.confirmation_execute import rehearsal_identity
from study_full_baselines.confirmation_freeze import validate as validate_freeze
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.build_research_site import label
from study_full_baselines.analyze_reception import write_csv


def read(path):
    return json.loads(path.read_text())


def check(path, digest):
    if sha256(path) != digest:
        raise ValueError('网站来源哈希不一致：'+str(path))


def check_files(folder, entries):
    for name, digest in entries.items():
        path = folder/name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('公开或来源文件越出声明目录。')
        check(path, digest)


def check_sources(folder, entries):
    verify_sources(entries)
    check_files(folder/'source_snapshot', entries)


def checked_timing(project, folder, identity, freeze):
    binding, execution, audit = [read(folder/n) for n in ['binding.json', 'execution.json', 'integrity_verification.json']]
    inner = folder/'timing'
    summary, original, protocol = [read(inner/n) for n in ['summary.json', 'integrity_verification.json', 'protocol.json']]
    formal = identity['final_confirmation']
    if (audit['status'] != 'passed_common_bundle_timing_audit' or execution['status'] != 'complete_actual_timing'
            or original['status'] != 'passed_artifact_and_reference_audit' or summary['status'] != 'complete'):
        raise ValueError('实际在线计时未完成核查。')
    for item in [binding, execution, audit, summary, original, protocol]:
        if item['final_fair_timing'] != formal:
            raise ValueError('正式计时和旧案例演练范围混淆。')
    if Path(binding['project']).resolve() != project or binding['runtime_manifest_sha256'] != identity['runtime_manifest_sha256']:
        raise ValueError('计时没有使用本次共同权重。')
    if audit['runtime_manifest_sha256'] != identity['runtime_manifest_sha256']:
        raise ValueError('计时核查没有绑定本次共同权重。')
    for obj, key, path in [(audit, 'binding_sha256', folder/'binding.json'),
            (audit, 'execution_sha256', folder/'execution.json'),
            (audit, 'original_audit_sha256', inner/'integrity_verification.json'),
            (execution, 'binding_sha256', folder/'binding.json'),
            (execution, 'timing_summary_sha256', inner/'summary.json'),
            (original, 'summary_sha256', inner/'summary.json'),
            (summary, 'protocol_sha256', inner/'protocol.json')]:
        check(path, obj[key])
    check(SOURCE/'study_full_baselines/audit_bundle_online.py', audit['auditor_sha256'])
    check(SOURCE/'study_full_baselines/audit_online_timing.py', original['auditor_sha256'])
    check_sources(folder, binding['source_sha256'])
    check_sources(inner, protocol['source_sha256'])
    if (execution['constructed_methods'] != binding['methods'] or protocol['methods'] != binding['methods']
            or [r['method'] for r in summary['methods']] != binding['methods']):
        raise ValueError('计时方法列表不一致。')
    if formal:
        if binding['methods'] != identity['ordinary_methods'] or binding['freeze_sha256'] != sha256(freeze):
            raise ValueError('正式计时未覆盖全部冻结方法。')
        if any(r['contention_reasons'] for r in summary['methods']):
            raise ValueError('正式计时存在资源竞争。')
    for r in summary['cases']:
        check(inner/r['file'], r['sha256'])
    # 网站仅保留读者需要的数字，不公开模型路径和主机进程信息。
    keys = ['method', 'measurement_calls', 'mean_software_ms', 'median_software_ms',
            'p95_software_ms', 'estimated_total_mean_ms', 'estimated_measurement_switch_ms',
            'cases', 'time_samples', 'cuda']
    return [{k:r[k] for k in keys} for r in summary['methods']]


def build(project, records, analysis, signals, timing, figures, exploration, output, freeze=None, rehearse=False):
    require_host()
    if bool(freeze) == bool(rehearse):
        raise ValueError('只选择旧案例展示演练或最终冻结方案。')
    if output.exists():
        raise FileExistsError('网站构建另建目录，不覆盖已有结果。')
    formal = not rehearse
    identity = rehearsal_identity(project) if rehearse else validate_freeze(freeze)
    if read(records/'protocol.json') != identity or Path(identity['project']).resolve() != project:
        raise ValueError('接收与当前冻结身份不同。')
    package = verify_bundle(Path(identity['runtime_bundle']))
    check(Path(identity['runtime_bundle'])/'manifest.json', identity['runtime_manifest_sha256'])
    checked = read(analysis/'complete.json')
    expected = 'complete_frozen_confirmation_statistics' if formal else 'complete_old_record_rehearsal'
    if checked['status'] != expected or checked['final_confirmation'] != formal:
        raise ValueError('独立确认统计未完成或范围不同。')
    if checked['freeze_sha256'] != (sha256(freeze) if freeze else None):
        raise ValueError('分析没有绑定本次冻结方案。')
    check(records/'protocol.json', checked['input_protocol_sha256'])
    check_sources(analysis, checked['source_sha256'])
    check_files(analysis, checked['file_sha256'])
    audit = read(analysis/'audit.json')
    for n in ['protocol', 'records', 'complete']:
        check(records/(n+'.json'), audit[n+'_sha256'])
    check_files(records, audit['raw_file_sha256'])
    if (audit['runtime_manifest_sha256'] != identity['runtime_manifest_sha256']
            or audit['training_environments'] != package['cohort']['train_environments']):
        raise ValueError('完整分析没有使用共同训练权重。')
    if formal:
        claim = read(project/'dataset_simulation/ops/full_baselines_20260925/fresh_confirmation_claim.json')
        if claim['freeze_sha256'] != sha256(freeze) or Path(claim['output']).resolve() != records:
            raise ValueError('最终接收没有登记到当前冻结方案。')
    overall = read(analysis/'overall.json')['records']
    groups = read(analysis/'groups.json')['records']
    comparisons = read(analysis/'comparisons.json')['records']
    if [r['method'] for r in overall] != identity['methods']:
        raise ValueError('质量汇总的方法不完整。')
    if formal and read(analysis/'plan.json') != identity['statistics']:
        raise ValueError('统计比较清单与冻结方案不同。')
    if formal and len(comparisons) != len(identity['statistics']['comparisons'])*len(identity['statistics']['metrics']):
        raise ValueError('未覆盖全部预设比较。')
    if rehearse and (comparisons or any(r['ber_ci95'] is not None for r in overall)):
        raise ValueError('单旧案例不能展示推断区间。')
    signal_protocol, signal_summary, signal_audit = [read(signals/n) for n in ['protocol.json', 'summary.json', 'integrity_verification.json']]
    for obj in [signal_protocol, signal_summary, signal_audit]:
        if obj['final_confirmation'] != formal:
            raise ValueError('信号案例范围不同。')
    for obj, field, path in [(signal_protocol, 'record_protocol_sha256', records/'protocol.json'),
            (signal_protocol, 'analysis_complete_sha256', analysis/'complete.json'),
            (signal_summary, 'protocol_sha256', signals/'protocol.json'),
            (signal_audit, 'protocol_sha256', signals/'protocol.json'),
            (signal_audit, 'summary_sha256', signals/'summary.json')]:
        check(path, obj[field])
    if (not signal_summary['all_quality_replays_passed'] or signal_audit['status'] != 'passed_signal_arrays_json_csv_audit'
            or signal_protocol['methods'] != identity['methods'] or signal_protocol['environment_index'] != 0
            or signal_protocol['carriers_ghz'] != identity['carriers']
            or [r['carrier_ghz'] for r in signal_summary['entries']] != identity['carriers']):
        raise ValueError('固定信号没有覆盖全部方法或载频。')
    check_sources(signals, signal_protocol['source_sha256'])
    check_sources(signals, signal_audit['source_sha256'])
    for entry in signal_summary['entries']:
        check_files(signals, entry['files'])
    plot = read(figures/'build.json'); review = read(figures/'visual_review.json')
    if plot['final_confirmation'] != formal or review['final_confirmation'] != formal or review['status'] != 'visually_reviewed':
        raise ValueError('本次信号图像尚未完成实际查看。')
    check(figures/'build.json', review['build_sha256'])
    check(signals/'summary.json', plot['input_summary_sha256'])
    check_files(signals, plot['input_files']); check_files(figures, plot['files'])
    check_sources(figures, plot['source_sha256'])
    if set(review['viewed_png_files']) != {n for n in plot['files'] if n.endswith('.png')}:
        raise ValueError('信号图像查看覆盖不完整。')
    times = checked_timing(project, timing, identity, freeze)
    old = read(exploration/'build.json')
    check_files(exploration, old['public_files'])
    verify_sources(old['source_sha256'])
    if formal and (old['status'] != 'exploratory_built' or len(old['complete_phases']) != 7):
        raise ValueError('完整探索过程尚未构建。')
    output.mkdir(parents=True)
    assets = []
    def copy(folder, name, destination=None):
        dest = destination or name
        target = output/dest; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder/name, target); assets.append(dest)
    for name in ['index.html', 'app.js', 'app.css']:
        copy(SOURCE/'study_full_baselines/site', name)
    for name in old['public_files']:
        copy(exploration, name, 'exploration/'+name)
    for name in plot['files']:
        copy(figures, name, 'signal_figures/'+name)
    methods = [dict(id=r['method'], label=label(r['method']), overall=r, phase='confirmation',
        probes=r['probes'], training_environments=r['training_environments'], privileged=r['privileged'],
        photonic_hardware=r['photonic_hardware'], full_online_timing_available=formal and not r['privileged']) for r in overall]
    examples = []; example_labels = {}
    scope = '新864环境独立确认' if formal else '旧单环境展示演练；不是独立测试结果'
    for entry in signal_summary['entries']:
        fc = entry['carrier_ghz']; name = 'carrier_%02d.json'%fc; control = 'carrier_%02d_controls.csv'%fc
        copy(signals, name, 'examples/'+name); copy(signals, control, 'examples/'+control)
        for method in read(signals/name)['methods']:
            example_labels[method['method']] = label(method['method'].split('/')[-1])
        examples.append(dict(carrier=fc, url='examples/'+name, controls_csv='examples/'+control,
                             scope=signal_protocol['scope'], scope_label=scope))
    phases = [dict(phase=n, complete=True, completed=len(identity['rows']), total=len(identity['rows']))
              for n in ['统一接收评测', '逐记录核查与统计', '固定案例回放']]
    result = dict(schema='mwp-confirmation-dashboard-v1', generated_at=now(), preview=rehearse, scope=scope,
        confirmation_complete=formal, test_environments=len(identity['rows']), carriers_per_environment=len(identity['carriers']),
        default_training_environments=package['cohort']['train_environments'], training_seed=0, apd_draws=8,
        methods=methods, groups={n:[r for r in groups if r['method'] == n] for n in identity['methods']},
        phases=phases, figures=[], examples=examples, example_labels=example_labels,
        exploration_url='exploration/index.html', comparisons=comparisons, comparisons_csv='comparisons.csv',
        comparison_scope='方法、权重和比较清单冻结后的新864环境，分层配对统计。' if formal else '单个旧案例仅检查展示流程，不计算置信区间或p值。',
        quality_scope_note='同一训练成员、固定权重、同一批新测试环境；黑线为95%环境区间。' if formal else '旧单环境演练均值，不用于总体方法排名。',
        timing_records=times,
        timing_status='全部37项普通方法使用最终共同权重的实际软件计时；固定旧输入、单条推理，模拟测量等待单独估算。' if formal else '仅6方法的计时流程演练；测量时后台任务正在运行，数字不用于正式速度排名。',
        scientific_notes=['仅含期望信号的多径、衰减、衰落和正常噪声；不加入独立干扰源或温漂。',
            '所有拟合方法使用相同训练环境；0、16、64次测量预算分开比较。',
            '教师使用额外信息；MRC采用不同数字接收硬件，其光子功率字段不适用。',
            '同一环境的载频和8次噪声抽样不当成新的独立环境。',
            '固定信号索引0；星座显示抽样0，指标汇总8次抽样。'],
        provenance=dict(analysis_sha256=sha256(analysis/'complete.json'), runtime_manifest_sha256=identity['runtime_manifest_sha256'],
            signal_audit_sha256=sha256(signals/'integrity_verification.json'), timing_audit_sha256=sha256(timing/'integrity_verification.json')))
    write_json(output/'results.json', result); assets.append('results.json')
    write_csv(output/'results.csv', [dict(m['overall'], label=m['label']) for m in methods]); assets.append('results.csv')
    copy(analysis, 'comparisons.csv')
    write_csv(output/'timing.csv', times); assets.append('timing.csv')
    # 私有来源记录只供本地复现；公开路径完全由assets白名单指定。
    write_json(output/'audit.json', dict(final_confirmation=formal, paths={k:str(v) for k,v in dict(
        records=records, analysis=analysis, signals=signals, timing=timing, figures=figures, exploration=exploration).items()},
        freeze_path=str(freeze) if freeze else None, provenance=result['provenance']))
    for name in assets:
        if Path(name).suffix in ['.json', '.csv', '.js', '.html']:
            text = (output/name).read_text()
            if '/home/qyb/' in text or '/Users/qyb/' in text:
                raise ValueError('公开文件包含私有绝对路径：'+name)
    sources = source_record(['study_full_baselines/'+n for n in ['build_confirmation_site.py',
        'build_research_site.py', 'CONFIRMATION_SITE_PROTOCOL.md', 'site/app.js', 'site/app.css', 'site/index.html']])
    for name in sources:
        dest=output/'source_snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
    write_json(output/'build.json', dict(status='confirmation_preview_built' if rehearse else 'confirmation_built',
        at=now(), public_files={name:sha256(output/name) for name in assets}, source_sha256=sources,
        displayed_methods=len(methods), comparison_rows=len(comparisons), final_confirmation=formal,
        online_timing_formal=formal, deployed=False, full_research_complete=False))
    print(json.dumps(dict(status='built', methods=len(methods), formal_confirmation=formal, public_files=len(assets))), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'records', 'analysis', 'signals', 'timing', 'figures', 'exploration', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', type=Path); mode.add_argument('--rehearse-old', action='store_true')
    a=parser.parse_args()
    build(*[getattr(a,n).resolve() for n in ['project','records','analysis','signals','timing','figures','exploration','output']],
          a.freeze.resolve() if a.freeze else None, a.rehearse_old)
