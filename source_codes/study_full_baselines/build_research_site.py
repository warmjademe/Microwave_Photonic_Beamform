"""从核验后的完整评分阶段构建网站；预览模式也不发布半个测试集的排名。"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.audit_results import read_phase, check_summary
from study_full_baselines.audit_reception_v2 import check_budget
from study_full_baselines.analyze_reception import canonical, clean_json, write_csv
from study_full_baselines.export_signal_examples import PHASES

LABELS = {'mlp': 'MLP · 直接控制', 'dnn': 'DNN · 直接控制', 'cnn': 'CNN · 直接控制',
    'rescnn': 'ResCNN · 直接控制', 'transformer': 'Transformer · 直接控制',
    'complex_cnn': '复数 CNN · 直接控制', 'jct': 'JCT · 监督适配',
    'ttd_das': '初始16套实测选择', 'public16': '初始16套实测选择',
    'codebook': '几何码本反馈', 'coordinate': '坐标搜索', 'spsa': 'SPSA',
    'done': 'DONE', 'de': '差分进化', 'teacher': '教师参考（额外信息）',
    'mrc': '理想数字 MRC（不同硬件）', 'ridge_response': '固定正则化响应估计',
    'covariance_response': '传统协方差估计＋逐路控制',
    'complex_response_cnn': '复响应 CNN＋逐路控制', 'real_response_cnn': '实响应 CNN＋逐路控制',
    'cnn_warm64': '复响应 CNN＋反馈确认', 'covariance_warm64': '传统估计＋反馈确认',
    'global_prior': '训练集全局固定设置', 'frequency_prior': '训练集逐载频固定设置',
    'per_tone_relative': '逐频相对线性估计', 'joint_relative': '联合频率相对线性估计',
    'joint_absolute': '联合频率普通线性估计',
    'quality_absolute_single': '单帧绝对控制监督', 'quality_absolute_robust': '多帧绝对控制监督',
    'quality_candidate_ce_robust': '候选分类监督', 'quality_candidate_soft_single': '单帧质量软标签',
    'quality_candidate_soft_robust': '多帧质量软标签'}


def label(name):
    if name in LABELS: return LABELS[name]
    for suffix, text in [('__multi_mmse', '＋多起点控制'), ('__single_10sweeps', '＋单起点10轮')]:
        if name.endswith(suffix):
            prefix = name[:-len(suffix)]
            return {'cnn': '复响应 CNN', 'covariance': '传统协方差'}.get(prefix, label(prefix))+text
    m = re.fullmatch(r'covariance_n(\d+)', name)
    if m: return '传统协方差 · %d环境' % int(m[1])
    m = re.fullmatch(r'response_n(\d+)_(fixed_epochs|equal_updates)', name)
    if m: return '响应 CNN · %d环境 · %s' % (int(m[1]), '40轮' if m[2] == 'fixed_epochs' else '等更新')
    return name


def key_for(name):
    return {'codebook64': 'codebook', 'public16': 'ttd_das'}.get(name, canonical(name))


def verified_manifest(folder, filename, field):
    meta = json.loads((folder/filename).read_text())
    for name, digest in meta[field].items():
        if sha256(folder/name) != digest: raise ValueError('待发布文件哈希改变：'+name)
    return meta


def build(project, output, preview):
    require_host()
    if output.exists(): raise FileExistsError('网站构建使用新目录，不覆盖既有发布。')
    base = project/'dataset_simulation'; data = base/'outputs/quality_rank_hybrid_20260925'
    study = base/'baseline_results/20260925_full_baselines'
    rows = [r for r in check_data(data)['environments'] if r['split'] == 'test']
    digest = sha256(data/'manifest.json'); methods = []; groups = {}; raw = {}; controls = {}
    states = []; audits = {}; source_records = {}
    for phase in PHASES:
        folder = study/phase; p = folder/'progress.json'
        status = json.loads(p.read_text()) if p.exists() else {}
        states.append(dict(phase=phase, complete=status.get('status') == 'complete',
                           completed=status.get('completed', 0), total=len(rows)))
        if status.get('status') != 'complete':
            if not preview: raise ValueError('完整探索结果构建缺少阶段：'+phase)
            continue
        info, values = read_phase(folder, rows, digest, False)
        info.update(check_summary(folder, values, rows)); info.update(check_budget(folder, values))
        audits[phase] = info
        summary = json.loads((folder/'summary.json').read_text())
        source_records[phase] = dict(summary_sha256=sha256(folder/'summary.json'),
                                    protocol_sha256=sha256(folder/'protocol.json'))
        for mi, name in enumerate(values['methods']):
            key = key_for(name); a = values['values'][:, :, mi, :10]; code = values['controls'][:, :, mi]
            if key in raw:
                if not np.array_equal(code, controls[key]) or not np.allclose(a, raw[key], rtol=1e-12, atol=0, equal_nan=True):
                    raise ValueError('网站合并了并不相同的控制器：'+key)
                continue
            raw[key], controls[key] = a, code
            records = [dict(r, method=key) for r in summary['records'] if r['method'] == name]
            overall = next(r for r in records if r['group'] == 'all')
            calls = values['values'][:, :, mi, values['metric_order'].index('feedback_calls')]
            count = re.search(r'_n(\d+)', key)
            training = None if key in ONLINE_CLASSIC+['teacher', 'mrc', 'ridge_response'] else int(count[1]) if count else 864
            methods.append(dict(id=key, label=label(key), phase=phase, overall=overall,
                probes=float(calls.mean()) if np.isfinite(calls).all() else None,
                training_environments=training, privileged=key in ['teacher', 'mrc'],
                photonic_hardware=key != 'mrc', full_online_timing_available=False))
            groups[key] = records
    if not methods: raise ValueError('没有完整且通过核验的阶段可供预览。')
    output.mkdir(parents=True); (output/'figures').mkdir(); (output/'examples').mkdir()
    assets = []
    for name in ['index.html', 'app.css', 'app.js']:
        shutil.copyfile(SOURCE/'study_full_baselines/site'/name, output/name); assets.append(name)
    figures = []
    scale = base/'diagnostics/20260926_scale_through_1728'
    scale_stems = ['scale_through_1728_curve', 'scale_through_1728_paired']
    if not (scale/'complete.json').exists():
        scale = base/'diagnostics/20260926_small_scale_figures'
        scale_stems = ['small_scale_curve', 'small_scale_paired']
    if (scale/'complete.json').exists():
        verified_manifest(scale, 'complete.json', 'files')
        for stem in scale_stems:
            for ext in ['png', 'pdf', 'svg']:
                name = stem+'.'+ext; shutil.copyfile(scale/name, output/'figures'/name); assets.append('figures/'+name)
            figures.append(dict(name=stem, section='scale', url='figures/'+stem+'.png', pdf='figures/'+stem+'.pdf'))
    components = base/'diagnostics/20260926_component_complete_figures'
    if (components/'complete.json').exists():
        verified_manifest(components, 'complete.json', 'files')
        for stem, title in [('component_four_combinations', '两阶段四组合的接收质量'),
                            ('component_paired_effects', '两个组件的环境配对差值和95%区间'),
                            ('component_all_carriers', '两阶段四组合在全部17档载频的结果')]:
            for ext in ['png', 'pdf', 'svg']:
                name = stem+'.'+ext
                shutil.copyfile(components/name, output/'figures'/name); assets.append('figures/'+name)
            figures.append(dict(name=stem, section='components', title=title,
                url='figures/'+stem+'.png', pdf='figures/'+stem+'.pdf'))
    for folder_name, section, entries in [
        ('20260926_completed_rq2','estimator',[
            ('matched_response_networks','参数量匹配的实数与复数响应网络'),
            ('estimator_paired_effects','估计阶段的配对差异与区间'),
            ('matched_networks_all_carriers','实数和复数网络的全部17载频结果')]),
        ('20260926_completed_feedback_figures','feedback',[
            ('feedback64_matched_budget','相同64次预算的完整质量比较'),
            ('feedback64_paired_effects','相同64次预算的配对差异与区间'),
            ('feedback64_selection_origins','最终采用控制设置的来源')])]:
        folder=base/'diagnostics'/folder_name
        if not (folder/'complete.json').exists(): continue
        verified_manifest(folder,'complete.json','files')
        for stem,title in entries:
            for ext in ['png','pdf','svg']:
                name=stem+'.'+ext;shutil.copyfile(folder/name,output/'figures'/name);assets.append('figures/'+name)
            figures.append(dict(name=stem,section=section,title=title,
                url='figures/'+stem+'.png',pdf='figures/'+stem+'.pdf'))
    comparisons = []
    for folder, prefix, checksum_field in [
        (base/'diagnostics/20260926_completed_rq2','components','files'),
        (study/'analysis_feedback_warm','feedback64','file_sha256')]:
        if not (folder/'complete.json').exists(): continue
        verified_manifest(folder,'complete.json',checksum_field)
        name=prefix+'_comparisons.csv';shutil.copyfile(folder/'comparisons.csv',output/name);assets.append(name)
        comparisons.extend(json.loads((folder/'comparisons.json').read_text())['records'])
    example_dir = base/'diagnostics/20260926_all_method_signal_examples'
    example_status = 'complete'
    if not (example_dir/'summary.json').exists():
        if not preview: raise ValueError('完整17载频信号案例尚未完成。')
        example_dir = base/'diagnostics/20260926_all_method_example_preflight'; example_status = 'preflight_subset'
    examples = []; example_labels = {}
    if (example_dir/'summary.json').exists():
        meta = json.loads((example_dir/'summary.json').read_text())
        if not meta['all_quality_replays_passed']: raise ValueError('案例质量核对未通过。')
        for entry in meta['entries']:
            for name, expected in entry['files'].items():
                if sha256(example_dir/name) != expected: raise ValueError('案例内容发生改变。')
            name = 'carrier_%02d.json' % entry['carrier_ghz']
            for method in json.loads((example_dir/name).read_text())['methods']:
                example_labels[method['method']] = label(key_for(method['method'].split('/')[-1]))
            shutil.copyfile(example_dir/name, output/'examples'/name); assets.append('examples/'+name)
            control = 'carrier_%02d_controls.csv' % entry['carrier_ghz']
            shutil.copyfile(example_dir/control, output/'examples'/control); assets.append('examples/'+control)
            examples.append(dict(carrier=entry['carrier_ghz'], url='examples/'+name,
                                 controls_csv='examples/'+control, scope=example_status))
    result = dict(schema='mwp-exploratory-dashboard-v1', generated_at=now(), preview=preview,
        scope='旧216环境探索性比较；新的864环境独立确认尚未开始',
        confirmation_complete=False, test_environments=len(rows), carriers_per_environment=17,
        default_training_environments=864, comparisons=clean_json(comparisons), comparisons_csv='comparisons.csv',
        comparison_scope='旧216环境探索性比较；每个预设比较族分别校正，未作为新环境确认结果。',
        quality_scope_note='当前排序仅适用于所选预算和训练规模；新环境确认与正式在线耗时待完成。',
        timing_records=[],
        training_seed=0, apd_draws=8, phases=states, methods=clean_json(methods), groups=clean_json(groups),
        figures=figures, examples=examples, example_scope=example_status, example_labels=example_labels,
        timing_status='最终在线耗时待实测；不以缓存读取或解码器片段代替完整推理耗时',
        scientific_notes=['仅含期望发射源的多径、衰减、衰落和正常噪声；无独立干扰源或新增温漂。',
            '共同输入为16套探测测量；控制输出为64路有效延时码和64路光衰减码。',
            '64次反馈方法另加48次测量。教师和数字MRC使用额外信息，MRC还采用不同接收硬件。',
            'EVM、NMSE和由NMSE换算的等效SNR相关；配对输出SNR采用无噪信号/噪声功率比。',
            '这里只展示完整测试阶段；待完成阶段不参与排序，也不补填零分。'])
    write_json(output/'results.json', result); assets.append('results.json')
    write_csv(output/'results.csv', [dict(m['overall'], label=m['label'], probes=m['probes'],
        training_environments=m['training_environments'], privileged=m['privileged']) for m in methods])
    assets.append('results.csv')
    write_csv(output/'comparisons.csv', comparisons); assets.append('comparisons.csv')
    # 核验记录留在构建目录内供复核，不自动加入NAS的公开路由。
    write_json(output/'audit.json', dict(phases=audits, input_manifest_sha256=digest, phase_sources=source_records))
    write_json(output/'build.json', dict(status='preview_built' if preview else 'exploratory_built',
        at=now(), public_files={name:sha256(output/name) for name in assets},
        source_sha256=source_record(['study_full_baselines/build_research_site.py']+
            ['study_full_baselines/site/'+name for name in ['index.html', 'app.css', 'app.js']]),
        scope=result['scope'], confirmation_complete=False, deployed=False,
        complete_phases=[s['phase'] for s in states if s['complete']], displayed_methods=len(methods)))
    print(json.dumps(dict(status='built', methods=len(methods), phases=len(audits), preview=preview)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preview', action='store_true'); a = p.parse_args(); build(a.project, a.output, a.preview)
