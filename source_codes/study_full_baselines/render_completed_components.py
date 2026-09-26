"""完整216环境的两阶段消融：配对区间与图表，不发布未完成统计族的p值。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.audit_results import read_phase, check_summary
from study_full_baselines.audit_reception_v2 import check_budget, strict_cross
from study_full_baselines.analyze_reception import canonical, clean_json, write_csv
from study_full_baselines.paired_statistics import FIELDS, sufficient, metrics, bootstrap, contrast

SOURCES = ['study_full_baselines/'+n for n in [
    'render_completed_components.py', 'common.py', 'audit_results.py',
    'audit_reception_v2.py', 'analyze_reception.py', 'paired_statistics.py',
    'EXPLORATORY_COMPARISONS.json', 'TWO_STAGE_ABLATION.md']]
FOUR = ['covariance_response', 'complex_response_cnn',
        'covariance__multi_mmse', 'cnn__multi_mmse']
LABELS = ['传统估计＋基础控制', 'CNN估计＋基础控制',
          '传统估计＋多起点控制', 'CNN估计＋多起点控制']


def run(project, output):
    require_host()
    if output.exists(): raise FileExistsError('消融图表与统计使用新目录，不覆盖。')
    base = project/'dataset_simulation'
    data = base/'outputs/quality_rank_hybrid_20260925'
    study = base/'baseline_results/20260925_full_baselines'
    rows = [r for r in check_data(data)['environments'] if r['split'] == 'test']
    if len(rows) != 216: raise ValueError('此入口限定旧216环境探索。')
    digest = sha256(data/'manifest.json'); arrays = {}; audit = {}
    sources = source_record(SOURCES)
    for phase in ['learned_evaluation', 'evaluation_components']:
        folder = study/phase
        info, value = read_phase(folder, rows, digest, False)
        info.update(check_summary(folder, value, rows)); info.update(check_budget(folder, value))
        if value['ids'] != [r['environment_id'] for r in rows]:
            raise ValueError('测试环境顺序不一致。')
        arrays[phase] = value; audit[phase] = info
    cross = [strict_cross(arrays, ('learned_evaluation', left), ('evaluation_components', right))
             for left, right in [('covariance_response', 'covariance__base_2sweeps'),
                                 ('complex_response_cnn', 'cnn__base_2sweeps')]]
    if any(c['status'] != 'matched' or c['environments'] != 216 for c in cross):
        raise ValueError('基础控制未跨流水线完整重放。')
    a = arrays['evaluation_components']; names = [canonical(n) for n in a['methods']]
    s = sufficient(a['values']); draws = bootstrap(s, 10000, 20260926)
    point = metrics(s.sum(0)); overall = []
    for mi, name in enumerate(names):
        row = dict(method=name, independent_environments=216, carriers_per_environment=17,
                   training_environments=864, probes=16)
        for fi, field in enumerate(FIELDS):
            row[field] = float(point[mi, fi])
            row[field+'_ci95'] = np.quantile(draws[:, mi, fi], [.025, .975]).tolist()
        overall.append(row)
    plan = json.loads((SOURCE/'study_full_baselines/EXPLORATORY_COMPARISONS.json').read_text())
    paired = []; pending = []
    for spec in plan['comparisons']:
        if not (spec['family'].startswith('RQ2_') or spec['id'] == 'core_vs_covariance_response'):
            continue
        if set(spec['terms'])-set(names):
            pending.append(spec['id']); continue
        c = np.asarray([spec['terms'].get(n, 0) for n in names], float)
        for field in plan['metrics']:
            result = contrast(s, draws, c, field)
            for key in ['p_value', 'estimand']: result.pop(key, None)
            paired.append(dict(comparison=spec['id'], family=spec['family'], terms=spec['terms'],
                scope='Descriptive paired interval; full prespecified multiplicity families pending', **result))
    output.mkdir(parents=True)
    for name in SOURCES:
        dst = output/'source_snapshot'/name
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE/name, dst)
    write_json(output/'audit.json', dict(phases=audit, cross_pipeline_checks=cross))
    write_json(output/'overall.json', dict(records=overall, scope='Old216 exploratory; fixed seed0'))
    write_json(output/'paired.json', clean_json(dict(records=paired, pending_comparisons=pending,
        multiplicity='No partial family adjusted p-values; full analysis remains queued')))
    write_csv(output/'overall.csv', overall); write_csv(output/'paired.csv', paired)
    atomic_npz(output/'bootstrap.npz', sufficient=s, bootstrap_metrics=draws,
        methods=np.asarray(names), fields=np.asarray(FIELDS), environment_ids=np.asarray(a['ids']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
    figures = []
    def save(fig, stem):
        for ext in ['png', 'pdf', 'svg']:
            path = output/(stem+'.'+ext); fig.savefig(path, dpi=200, bbox_inches='tight'); figures.append(path.name)
        plt.close(fig)
    fields = ['ber', 'rms_evm_percent', 'paired_output_snr_db']
    scales = [100, 1, 1]
    titles = ['BER（%）越低越好', 'RMS EVM（%）越低越好', '配对输出 SNR（dB）越高越好']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, field, scale, title in zip(axes, fields, scales, titles):
        for ids, color, label in [(FOUR[::2], '#777777', '传统训练协方差估计'),
                                  (FOUR[1::2], '#0072B2', '复响应 CNN估计')]:
            y = [next(r for r in overall if r['method'] == n)[field]*scale for n in ids]
            ax.plot([0, 1], y, marker='o', color=color, label=label)
            for i, value in enumerate(y): ax.annotate('%.3f' % value, (i, value), (0, 8),
                                                     textcoords='offset points', ha='center', fontsize=10)
        ax.set_xticks([0, 1], ['基础：单起点2轮', '扩展：5起点各2轮'])
        ax.set_title(title, fontsize=11); ax.grid(alpha=.18); ax.margins(y=.3, x=.15)
    axes[0].legend(fontsize=9, loc='best')
    fig.suptitle('两阶段消融：主要收益来自信号估计，控制搜索的变化另行检验', fontsize=14)
    fig.text(.5, .04, '相同864训练环境、216旧测试环境×17载频；均只用16次测量。', ha='center')
    fig.text(.5, .008, '这里只画总体均值；配对区间见下一图。多起点控制使用更多计算，不增加现场测量。', ha='center')
    fig.tight_layout(rect=(0, .10, 1, .92)); save(fig, 'component_four_combinations')

    chosen = ['core_vs_covariance_response', 'cnn__multi_mmse_minus_complex_response_cnn',
              'covariance__multi_mmse_minus_covariance_response',
              'cnn__multi_mmse_minus_cnn__single_10sweeps', 'two_stage_interaction']
    labels = ['换成CNN估计（基础控制不变）', 'CNN估计：多起点 − 基础',
              '传统估计：多起点 − 基础', 'CNN估计：多起点 − 单起点10轮', '两阶段交互差值']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, field, scale, title in zip(axes, fields, scales,
                                      ['BER差值（百分点）', 'EVM差值（百分点）', '输出SNR差值（dB）']):
        for i, key in enumerate(chosen):
            r = next(r for r in paired if r['comparison'] == key and r['metric'] == field)
            lo, hi = np.asarray(r['ci95'])*scale; value = r['estimate']*scale
            color = '#0072B2' if i == 0 else '#D55E00'
            ax.plot([lo, hi], [i, i], color=color, lw=2); ax.scatter(value, i, color=color, zorder=3)
        ax.axvline(0, color='#444444', ls='--', lw=1); ax.grid(axis='x', alpha=.18)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xlabel(title+'\n'+('负值更好' if field != 'paired_output_snr_db' else '正值更好'))
    axes[0].invert_yaxis()
    fig.suptitle('组件作用的配对比较：点为总体差值，横线为95%环境bootstrap区间', fontsize=14)
    fig.text(.5, .025, '10,000次环境块重采样；旧测试探索结果，固定训练seed0；完整预设检验族尚待实数CNN对照。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .92)); save(fig, 'component_paired_effects')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, field, scale, title in zip(axes, fields, scales, titles):
        fi = FIELDS.index(field)
        for name, label, color, style in zip(FOUR, LABELS,
                ['#777777', '#0072B2', '#777777', '#0072B2'], ['--', '--', '-', '-']):
            mi = names.index(name); ys = []
            for ci in range(17):
                one = sufficient(a['values'][:, ci:ci+1, mi:mi+1])
                ys.append(float(metrics(one.sum(0))[0, fi])*scale)
            ax.plot(range(4, 21), ys, style, color=color, label=label)
        ax.set_title(title, fontsize=11); ax.set_xlabel('载频（GHz）')
        ax.set_xticks([4, 8, 12, 16, 20]); ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle('全部17档载频：两阶段四组合的接收质量', fontsize=14)
    fig.text(.5, .025, '每个频率包含相同216个传播环境；相邻点不是独立测试环境，重合曲线不作偏移。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92)); save(fig, 'component_all_carriers')
    verify_sources(sources)
    files = ['audit.json', 'overall.json', 'paired.json', 'overall.csv', 'paired.csv', 'bootstrap.npz']+figures
    write_json(output/'complete.json', dict(status='complete_component_descriptive_analysis', at=now(),
        source_sha256=sources, data_manifest_sha256=digest,
        summary_sha256=sha256(study/'evaluation_components/summary.json'),
        files={name: sha256(output/name) for name in files}, visual_review='pending',
        full_experiment_complete=False))
    print(json.dumps(dict(status='complete_component_descriptive_analysis', methods=len(names),
                          paired_metric_intervals=len(paired), figures=3)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.project, a.output)
