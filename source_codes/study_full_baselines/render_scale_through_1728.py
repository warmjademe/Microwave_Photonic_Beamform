"""从完整接收记录绘制216/432/864/1728环境规模曲线，不改写前三档结果。"""
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
from study_full_baselines.analyze_reception import canonical, clean_json, write_csv

COUNTS = [216, 432, 864, 1728]
CURVES = {
    'cnn_fixed40': ['response_n0216_fixed_epochs', 'response_n0432_fixed_epochs',
                    'complex_response_cnn', 'response_n1728_fixed_epochs'],
    'cnn_equal_updates': ['response_n0216_equal_updates', 'response_n0432_equal_updates',
                          'complex_response_cnn', 'response_n1728_equal_updates'],
    'covariance': ['covariance_n0216', 'covariance_n0432', 'covariance_response', 'covariance_n1728']}
SOURCES = ['study_full_baselines/'+name for name in [
    'render_scale_through_1728.py', 'paired_statistics.py', 'common.py', 'audit_results.py',
    'audit_reception_v2.py', 'analyze_reception.py', 'EXPLORATORY_COMPARISONS.json']]


def run(project, output):
    require_host()
    if output.exists():
        raise FileExistsError('已有结果不得覆盖。')
    sources = source_record(SOURCES)
    base = project/'dataset_simulation'
    data = base/'outputs/quality_rank_hybrid_20260925'
    rows = [r for r in check_data(data)['environments'] if r['split'] == 'test']
    ids = [r['environment_id'] for r in rows]
    if len(ids) != 216 or len(set(ids)) != 216:
        raise ValueError('本入口固定使用旧216个测试环境。')
    names = list(dict.fromkeys(name for curve in CURVES.values() for name in curve))
    columns, audits = {}, {}
    for phase in ['learned_evaluation', 'evaluation_scaling_1728']:
        folder = base/'baseline_results/20260925_full_baselines'/phase
        audit, arrays = read_phase(folder, rows, sha256(data/'manifest.json'), False)
        audit.update(check_summary(folder, arrays, rows))
        audit.update(check_budget(folder, arrays))
        if arrays['ids'] != ids:
            raise ValueError('跨流水线的环境身份或顺序不同。')
        audits[phase] = audit
        for mi, original in enumerate(arrays['methods']):
            name = canonical(original)
            if name not in names:
                continue
            if name in columns:
                raise ValueError('规模方法重复。')
            calls = arrays['values'][:, :, mi, arrays['metric_order'].index('feedback_calls')]
            if not np.all(calls == 16):
                raise ValueError('规模比较必须都使用16次测量。')
            columns[name] = arrays['values'][:, :, mi, :10]
    if set(columns) != set(names):
        raise ValueError('某一规模尚无完整接收记录。')
    s = sufficient(np.stack([columns[name] for name in names], axis=2))
    plan = json.loads((SOURCE/'study_full_baselines/EXPLORATORY_COMPARISONS.json').read_text())
    draws = bootstrap(s, plan['bootstrap_repetitions'], plan['bootstrap_seed'])
    point = metrics(s.sum(0))
    entries = []
    for curve, methods in CURVES.items():
        for count, name in zip(COUNTS, methods):
            mi = names.index(name)
            row = dict(curve=curve, training_environments=count, method=name,
                       old_test_environments=216, nominal_probes=16)
            for fi, field in enumerate(FIELDS):
                row[field] = float(point[mi, fi])
                row[field+'_ci95'] = np.quantile(draws[:, mi, fi], [.025, .975]).tolist()
            entries.append(row)
    paired, pending = [], []
    for spec in plan['comparisons']:
        if not spec['family'].startswith('RQ3_'):
            continue
        if set(spec['terms'])-set(names):
            pending.append(spec['id'])
            continue
        coefficients = np.asarray([spec['terms'].get(name, 0) for name in names], float)
        for field in plan['metrics']:
            result = contrast(s, draws, coefficients, field)
            # 完整比较族还包含3456规模，此处只报告效应和区间。
            for key in ['p_value', 'estimand']:
                result.pop(key, None)
            paired.append(dict(comparison=spec['id'], family=spec['family'], terms=spec['terms'],
                inference_scope='descriptive paired interval; full multiplicity analysis pending', **result))
    output.mkdir(parents=True)
    for name in SOURCES:
        dst = output/'source_snapshot'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dst)
    write_json(output/'audit.json', audits)
    write_json(output/'curves.json', dict(status='complete_through_1728', records=entries,
        pending_sizes=[3456], scope='Old216 exploration; fixed seed0; not sealed confirmation'))
    write_json(output/'paired.json', clean_json(dict(records=paired,
        pending_planned_comparisons=pending, multiplicity='No partial adjusted p-values reported')))
    write_csv(output/'curves.csv', entries)
    write_csv(output/'paired.csv', paired)
    atomic_npz(output/'bootstrap.npz', sufficient=s, bootstrap_metrics=draws,
        methods=np.asarray(names), fields=np.asarray(FIELDS), environment_ids=np.asarray(ids))
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
            path = output/(stem+'.'+ext)
            fig.savefig(path, dpi=200, bbox_inches='tight')
            figures.append(path.name)
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
            ci = np.asarray([r[field+'_ci95'] for r in values])*scale
            ax.plot(range(4), y, style, marker='o', color=color, label=label)
            ax.fill_between(range(4), ci[:, 0], ci[:, 1], alpha=.08, color=color)
        ax.set_xticks(range(4), list(map(str, COUNTS)))
        ax.set_xlabel('独立训练环境数')
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=9)
    fig.suptitle('训练规模216→1728：数据量与训练更新次数分别比较', fontsize=15)
    fig.text(.5, .045, '同一216个旧测试环境×17载频；阴影为环境块bootstrap的95%区间，固定训练seed0。', ha='center')
    fig.text(.5, .012, '等更新日程匹配864环境的训练步数；3456环境尚未完成，不绘制缺失成绩。', ha='center')
    fig.tight_layout(rect=(0, .10, 1, .92))
    save(fig, 'scale_through_1728_curve')
    selected, labels = [], []
    for schedule, label in [('fixed_epochs', '固定40轮'), ('equal_updates', '相同更新')]:
        for left, right in zip(COUNTS[1:], COUNTS[:-1]):
            selected.append(f'scale_{schedule}_{left}_minus_{right}')
            labels.append(f'{label}：{left} − {right}')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, field, scale, title in zip(axes, ['ber', 'rms_evm_percent', 'paired_output_snr_db'],
            [100, 1, 1], ['BER差值（百分点）', 'EVM差值（百分点）', '输出 SNR差值（dB）']):
        for i, name in enumerate(selected):
            row = next(r for r in paired if r['comparison'] == name and r['metric'] == field)
            lo, hi = np.asarray(row['ci95'])*scale
            color = '#0072B2' if i < 3 else '#D55E00'
            ax.plot([lo, hi], [i, i], color=color, linewidth=2)
            ax.scatter(row['estimate']*scale, i, color=color, s=45, zorder=3)
        ax.axvline(0, color='#444444', linestyle='--', linewidth=1)
        ax.set_yticks(range(6), labels)
        ax.set_xlabel(title+'\n'+('正值表示改善' if scale == 1 and 'snr' in field else '负值表示改善'))
        ax.grid(axis='x', alpha=.18)
    axes[0].invert_yaxis()
    fig.suptitle('逐档增加训练数据：同一测试环境上的配对变化', fontsize=14)
    fig.text(.5, .025, '横线为10,000次环境块bootstrap的95%区间；旧测试探索性结果，完整3456规模比较仍待完成。', ha='center')
    fig.tight_layout(rect=(0, .08, 1, .92))
    save(fig, 'scale_through_1728_paired')
    verify_sources(sources)
    files = ['audit.json', 'curves.json', 'paired.json', 'curves.csv', 'paired.csv', 'bootstrap.npz']+figures
    write_json(output/'complete.json', dict(status='complete_through_1728', at=now(),
        source_sha256=sources, files={name: sha256(output/name) for name in files},
        visual_review='pending', full_scale_study_complete=False))
    print(json.dumps(dict(status='complete_through_1728', methods=len(names),
        curves=len(entries), paired_metric_intervals=len(paired))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.output)
