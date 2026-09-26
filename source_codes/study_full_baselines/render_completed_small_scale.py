"""只展示已完整评测的216/432/864规模曲线；扩大规模不填零，不补造结果。"""
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
from study_full_baselines.audit_reception_v2 import check_budget
from study_full_baselines.paired_statistics import FIELDS, sufficient, metrics, bootstrap, contrast
from study_full_baselines.analyze_reception import clean_json, write_csv

CURVES = {
    'cnn_fixed40': ['response_n0216_fixed_epochs', 'response_n0432_fixed_epochs', 'complex_response_cnn'],
    'cnn_equal_updates': ['response_n0216_equal_updates', 'response_n0432_equal_updates', 'complex_response_cnn'],
    'covariance': ['covariance_n0216', 'covariance_n0432', 'covariance_response']}
SOURCES = ['study_full_baselines/'+n for n in ['render_completed_small_scale.py', 'paired_statistics.py',
    'common.py', 'audit_results.py', 'audit_reception_v2.py', 'analyze_reception.py', 'EXPLORATORY_COMPARISONS.json']]


def run(project, output):
    require_host()
    if output.exists(): raise FileExistsError('已有规模图及统计不得覆盖。')
    base = project/'dataset_simulation'; data = base/'outputs/quality_rank_hybrid_20260925'
    folder = base/'baseline_results/20260925_full_baselines/learned_evaluation'
    rows = [r for r in check_data(data)['environments'] if r['split'] == 'test']
    if len(rows) != 216: raise ValueError('本入口只用于已查看的旧216环境。')
    audit, a = read_phase(folder, rows, sha256(data/'manifest.json'), False)
    audit.update(check_summary(folder, a, rows)); audit.update(check_budget(folder, a))
    if a['ids'] != [r['environment_id'] for r in rows]: raise ValueError('环境顺序错误。')
    sources = source_record(SOURCES); names = a['methods']; s = sufficient(a['values'])
    b = bootstrap(s, 10000, 20260926); point = metrics(s.sum(0)); entries = []
    for kind, methods in CURVES.items():
        for count, name in zip([216, 432, 864], methods):
            mi = names.index(name)
            entry = dict(curve=kind, training_environments=count, method=name,
                         old_test_environments=216, nominal_probes=16)
            for fi, field in enumerate(FIELDS):
                entry[field] = float(point[mi, fi]); entry[field+'_ci95'] = np.quantile(b[:, mi, fi], [.025, .975]).tolist()
            entries.append(entry)
    plan = json.loads((SOURCE/'study_full_baselines/EXPLORATORY_COMPARISONS.json').read_text())
    paired = []; pending = []
    for spec in plan['comparisons']:
        if not spec['family'].startswith('RQ3_'): continue
        if set(spec['terms'])-set(names):
            pending.append(spec['id']); continue
        c = np.asarray([spec['terms'].get(name, 0) for name in names], float)
        for field in plan['metrics']:
            result = contrast(s, b, c, field)
            # 尚未完成整个预设RQ3校正族；这里只展示效应和区间，不公布部分校正排名。
            for key in ['p_value', 'estimand']: result.pop(key, None)
            paired.append(dict(comparison=spec['id'], family=spec['family'], terms=spec['terms'],
                inference_scope='descriptive paired interval; full planned multiplicity analysis pending', **result))
    output.mkdir(parents=True)
    for name in SOURCES:
        dst = output/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE/name, dst)
    write_json(output/'audit.json', audit)
    write_json(output/'curves.json', dict(status='complete_for_three_sizes', records=entries,
        pending_sizes=[1728, 3456], scope='Old216 exploration; one fixed training seed; no sealed confirmation'))
    write_json(output/'paired.json', clean_json(dict(records=paired, pending_planned_comparisons=pending,
        multiplicity='Full original comparison family not complete; no partial adjusted p-values reported')))
    write_csv(output/'curves.csv', entries); write_csv(output/'paired.csv', paired)
    atomic_npz(output/'bootstrap.npz', sufficient=s, bootstrap_metrics=b,
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
            path = output/(stem+'.'+ext); fig.savefig(path, dpi=200, bbox_inches='tight'); figures.append(path)
        plt.close(fig)
    settings = [('cnn_fixed40', '响应 CNN：固定40轮', '#0072B2', '-'),
                ('cnn_equal_updates', '响应 CNN：相同更新次数', '#D55E00', '--'),
                ('covariance', '传统训练协方差', '#777777', ':')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    for ax, field, scale, title in zip(axes, ['ber', 'rms_evm_percent', 'paired_output_snr_db'],
                                      [100, 1, 1], ['BER（%）越低越好', 'RMS EVM（%）越低越好', '配对输出 SNR（dB）越高越好']):
        for kind, label, color, style in settings:
            values = [r for r in entries if r['curve'] == kind]
            y = np.asarray([r[field] for r in values])*scale
            interval = np.asarray([r[field+'_ci95'] for r in values])*scale
            ax.plot(np.arange(3), y, style, marker='o', color=color, label=label)
            ax.fill_between(np.arange(3), interval[:, 0], interval[:, 1], alpha=.08, color=color)
        ax.set_xticks(np.arange(3), ['216', '432', '864']); ax.set_xlabel('独立训练环境数')
        ax.set_title(title, fontsize=11); ax.grid(alpha=.18)
    axes[0].legend(fontsize=9, loc='best')
    fig.suptitle('首段规模曲线：数据覆盖与训练更新次数分别比较', fontsize=15)
    fig.text(.5, .045, '同一216个旧测试环境×17载频；阴影为环境块bootstrap的95%区间，固定训练seed0。', ha='center')
    fig.text(.5, .012, '等更新日程匹配864环境的训练步数；1728和3456环境尚未完成，不绘制缺失成绩。', ha='center')
    fig.tight_layout(rect=(0, .10, 1, .92)); save(fig, 'small_scale_curve')
    selected = ['scale_fixed_epochs_432_minus_216', 'scale_fixed_epochs_864_minus_432',
                'scale_equal_updates_432_minus_216', 'scale_equal_updates_864_minus_432']
    labels = ['固定40轮：432 − 216', '固定40轮：864 − 432',
              '相同更新：432 − 216', '相同更新：864 − 432']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, field, scale, title in zip(axes, ['ber', 'rms_evm_percent'], [100, 1],
                                      ['BER差值（百分点）', 'EVM差值（百分点）']):
        for i, name in enumerate(selected):
            row = next(r for r in paired if r['comparison'] == name and r['metric'] == field)
            lo, hi = np.asarray(row['ci95'])*scale; value = row['estimate']*scale
            color = '#0072B2' if i < 2 else '#D55E00'
            ax.plot([lo, hi], [i, i], color=color, linewidth=2); ax.scatter(value, i, color=color, s=50, zorder=3)
        ax.axvline(0, color='#444444', linestyle='--', linewidth=1)
        ax.set_yticks(np.arange(4), labels); ax.set_xlabel(title+'；负值表示增加数据后更好')
        ax.grid(axis='x', alpha=.18)
    axes[0].invert_yaxis()
    fig.suptitle('按同一环境配对：增加一档训练规模的接收质量变化', fontsize=14)
    fig.text(.5, .028, '横线为10,000次环境块bootstrap的95%区间；旧测试探索性结果，完整规模比较仍在进行。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92)); save(fig, 'small_scale_paired')
    verify_sources(sources)
    files = ['audit.json', 'curves.json', 'paired.json', 'curves.csv', 'paired.csv', 'bootstrap.npz']
    write_json(output/'complete.json', dict(status='complete_for_three_sizes', at=now(),
        source_sha256=sources, learned_results_summary_sha256=sha256(folder/'summary.json'),
        files={name: sha256(output/name) for name in files+[p.name for p in figures]},
        visual_review='pending', full_scale_study_complete=False))
    print(json.dumps(dict(status='complete_for_three_sizes', figures=2, paired_metric_intervals=len(paired))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.project, a.output)
