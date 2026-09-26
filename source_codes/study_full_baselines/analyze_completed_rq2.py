"""实数响应CNN完整评分后，完成预设RQ2全部比较族与参数量匹配图表。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.audit_results import read_phase, check_summary
from study_full_baselines.audit_reception_v2 import check_budget
from study_full_baselines.analyze_reception import canonical, clean_json, write_csv, planned_comparisons
from study_full_baselines.paired_statistics import FIELDS, sufficient, metrics, bootstrap


def run(project, output):
    require_host()
    if output.exists(): raise FileExistsError('RQ2结果不覆盖旧统计。')
    root = project/'dataset_simulation'; data = root/'outputs/quality_rank_hybrid_20260925'
    study = root/'baseline_results/20260925_full_baselines'
    rows = [r for r in check_data(data)['environments'] if r['split'] == 'test']
    if len(rows) != 216: raise ValueError('只允许旧216环境探索。')
    arrays = []; names = []; audit = {}
    for phase in ['evaluation_components', 'evaluation_real_response_cnn']:
        folder = study/phase; info, value = read_phase(folder, rows, sha256(data/'manifest.json'), False)
        info.update(check_summary(folder, value, rows)); info.update(check_budget(folder, value))
        if value['ids'] != [r['environment_id'] for r in rows]: raise ValueError('环境顺序不同。')
        arrays.append(value['values'][:, :, :, :10]); names.extend(canonical(n) for n in value['methods'])
        audit[phase] = info
    if len(names) != len(set(names)): raise ValueError('重复方法未单列核对。')
    plan_path = SOURCE/'study_full_baselines/EXPLORATORY_COMPARISONS.json'
    original_plan = json.loads(plan_path.read_text())
    plan = {**original_plan, 'comparisons': [r for r in original_plan['comparisons'] if r['family'].startswith('RQ2_')]}
    if len(plan['comparisons']) != 12 or any(set(r['terms'])-set(names) for r in plan['comparisons']):
        raise ValueError('RQ2原预设比较不完整。')
    sources = source_record(['study_full_baselines/'+n for n in ['analyze_completed_rq2.py',
        'audit_results.py', 'audit_reception_v2.py', 'analyze_reception.py', 'paired_statistics.py',
        'EXPLORATORY_COMPARISONS.json', 'ANALYSIS_PROTOCOL.md', 'common.py']])
    values = np.concatenate(arrays, axis=2); s = sufficient(values)
    draws = bootstrap(s, plan['bootstrap_repetitions'], plan['bootstrap_seed'])
    overall = []; point = metrics(s.sum(0))
    for mi, name in enumerate(names):
        record = dict(method=name, training_environments=864, test_environments=216, carriers=17, probes=16)
        for fi, field in enumerate(FIELDS):
            record[field] = float(point[mi, fi]); record[field+'_ci95'] = np.quantile(draws[:, mi, fi], [.025, .975]).tolist()
        overall.append(record)
    paired = planned_comparisons(plan, names, s, draws)
    if len(paired) != 60 or any(r['status'] != 'computed' for r in paired): raise ValueError('RQ2指标检验缺失。')
    families = {name: sum(r['family'] == name for r in paired) for name in {r['family'] for r in paired}}
    if families != {'RQ2_estimator': 20, 'RQ2_controller': 35, 'RQ2_interaction': 5}:
        raise ValueError('预设校正族大小改变。')
    model_sources = {}
    for name, folder, expected in [('real', study/'real_response_cnn', 17524),
        ('complex', root/'baseline_results/20260925_response_control', 17540)]:
        meta = json.loads((folder/'complete.json').read_text())
        if meta['parameters'] != expected or sha256(folder/'weights.pt') != meta['weights_sha256']:
            raise ValueError('参数量匹配模型身份错误。')
        model_sources[name] = dict(parameters=expected, weights_sha256=meta['weights_sha256'])
    output.mkdir(parents=True)
    for name in sources:
        dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE/name, dest)
    write_json(output/'audit.json', dict(phases=audit, model_sources=model_sources))
    write_json(output/'plan.json', plan)
    write_json(output/'overall.json', dict(records=overall, scope='Old216 exploratory; one training seed'))
    write_json(output/'comparisons.json', clean_json(dict(records=paired, complete_families=families,
        scope='All prespecified RQ2 comparisons; not fresh confirmation or completed RQ1/RQ3')))
    write_csv(output/'overall.csv', overall); write_csv(output/'comparisons.csv', paired)
    atomic_npz(output/'statistics.npz', sufficient=s, bootstrap_metrics=draws,
        methods=np.asarray(names), fields=np.asarray(FIELDS), environment_ids=np.asarray([r['environment_id'] for r in rows]))
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
    fields = ['ber', 'rms_evm_percent', 'paired_output_snr_db']; scales = [100, 1, 1]
    titles = ['BER（%）越低越好', 'RMS EVM（%）越低越好', '配对输出 SNR（dB）越高越好']
    selected = ['complex_response_cnn', 'cnn__multi_mmse', 'real_response_cnn', 'real_response_cnn__multi_mmse']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    for ax, field, scale, title in zip(axes, fields, scales, titles):
        for group, label, color in [(selected[:2], '复数CNN（17,540参数）', '#0072B2'),
                                    (selected[2:], '实数CNN（17,524参数）', '#D55E00')]:
            ys = [next(r for r in overall if r['method'] == n)[field]*scale for n in group]
            ax.plot([0, 1], ys, 'o-', label=label, color=color)
            for x, y in enumerate(ys): ax.annotate('%.3f' % y, (x, y), (0, 8), textcoords='offset points', ha='center', fontsize=10)
        ax.set_xticks([0, 1], ['基础：单起点2轮', '多起点：5起点各2轮']); ax.set_title(title, fontsize=11)
        ax.grid(alpha=.18); ax.margins(x=.15, y=.3)
    axes[0].legend(fontsize=9)
    fig.suptitle('参数量匹配对照：实数与复数响应CNN', fontsize=14)
    fig.text(.5, .035, '相同864训练环境、216旧测试环境×17载频；共同响应监督、40轮seed0；均使用16次测量。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92)); save(fig, 'matched_response_networks')
    chosen = ['estimation_vs_per_tone_relative', 'estimation_vs_joint_relative', 'estimation_vs_joint_absolute', 'estimation_vs_real_response_cnn']
    labels = ['相对损失逐频线性', '相对损失联合线性', '绝对损失联合线性', '参数量匹配实数CNN']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, field, scale, title in zip(axes, fields, scales, ['BER差值（百分点）', 'EVM差值（百分点）', '输出SNR差值（dB）']):
        for i, key in enumerate(chosen):
            r = next(r for r in paired if r['comparison'] == key and r['metric'] == field)
            lo, hi = np.asarray(r['ci95'])*scale
            ax.plot([lo, hi], [i, i], color='#0072B2', lw=2); ax.scatter(r['estimate']*scale, i, color='#0072B2')
        ax.axvline(0, ls='--', color='#555'); ax.set_yticks(range(4), labels); ax.grid(axis='x', alpha=.18)
        ax.set_xlabel(title+'\n'+('负值有利于复数CNN' if field != 'paired_output_snr_db' else '正值有利于复数CNN'))
    axes[0].invert_yaxis(); fig.suptitle('估计阶段比较：复数CNN减各参照，点与95%环境配对区间', fontsize=14)
    fig.text(.5, .025, '10,000次环境块bootstrap；全部预设RQ2检验族已按族做BH–FDR，完整p值见CSV。旧测试探索结果。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .92)); save(fig, 'estimator_paired_effects')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    for ax, field, scale, title in zip(axes, fields, scales, titles):
        for name, label, color, style in zip(selected,
            ['复数CNN＋基础', '复数CNN＋多起点', '实数CNN＋基础', '实数CNN＋多起点'],
            ['#0072B2', '#0072B2', '#D55E00', '#D55E00'], ['--', '-', '--', '-']):
            mi = names.index(name); ys = [metrics(sufficient(values[:, ci:ci+1, mi:mi+1]).sum(0))[0, FIELDS.index(field)]*scale for ci in range(17)]
            ax.plot(range(4, 21), ys, style, label=label, color=color)
        ax.set_title(title, fontsize=11); ax.set_xlabel('载频（GHz）'); ax.set_xticks([4, 8, 12, 16, 20]); ax.grid(alpha=.18)
    axes[0].legend(fontsize=8); fig.suptitle('全部17档载频：响应网络与控制阶段的组合', fontsize=14)
    fig.text(.5, .025, '同一环境的频点保持配对；重合曲线不作偏移；数字MRC不进入该同硬件对照。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92)); save(fig, 'matched_networks_all_carriers')
    verify_sources(sources)
    filenames = ['audit.json', 'plan.json', 'overall.json', 'comparisons.json', 'overall.csv', 'comparisons.csv', 'statistics.npz']+figures
    write_json(output/'complete.json', dict(status='complete_exploratory_rq2', methods=len(names), tests=len(paired), families=families,
        source_sha256=sources, original_plan_sha256=sha256(plan_path), files={n: sha256(output/n) for n in filenames},
        visual_review='pending', full_experiment_complete=False, at=now()))
    print(json.dumps(dict(status='complete_exploratory_rq2', methods=len(names), tests=len(paired), families=families,
        matched_networks=[r for r in overall if r['method'] in selected])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    if args.wait_pid:
        require_host()
        progress = args.project/'dataset_simulation/baseline_results/20260925_full_baselines/evaluation_real_response_cnn/progress.json'
        def start_ticks():
            try: return Path('/proc/%d/stat' % args.wait_pid).read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError: return None
        expected = start_ticks()
        while not progress.exists() or json.loads(progress.read_text())['status'] != 'complete':
            if expected is None or start_ticks() != expected:
                raise RuntimeError('指定实数CNN评测进程已终止或身份改变，但完整结果未提交。')
            time.sleep(15)
    run(args.project.resolve(), args.output.resolve())
