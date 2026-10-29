"""绘制完整敏感性统计；不读取中途成绩、不重新训练或重算统计量。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform


SCENARIOS = [
    'nominal', 'delay_x1p5', 'angular_x2', 'power_minus6',
    'rin_minus150', 'rin_minus140', 'rin_minus130',
    'delay_error_5ps', 'delay_error_10ps', 'loss_error_0p5db', 'loss_error_1db',
]
METHODS = ['cnn_warm64', 'covariance_warm64', 'initial_select64']
LABELS = {
    'en': [
        'Nominal', 'Path delay ×1.5', 'Angular spread ×2', 'Signal power −6 dB',
        'RIN −150 dB/Hz', 'RIN −140 dB/Hz', 'RIN −130 dB/Hz',
        'Delay error σ = 5 ps', 'Delay error σ = 10 ps',
        'Loss error σ = 0.5 dB', 'Loss error σ = 1 dB',
    ],
    'zh': [
        '名义条件', '路径时延 ×1.5', '角度扩展 ×2', '信号功率 −6 dB',
        'RIN −150 dB/Hz', 'RIN −140 dB/Hz', 'RIN −130 dB/Hz',
        '延时误差 σ = 5 ps', '延时误差 σ = 10 ps',
        '损耗误差 σ = 0.5 dB', '损耗误差 σ = 1 dB',
    ],
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_inputs(analysis):
    """全部完成标记及绑定文件通过以后，才允许打开汇总数值。"""
    proof = read(analysis / 'complete.json')
    assert proof['status'] == 'complete'
    assert proof['raw_records_verified'] == 864 * 17 * 11
    for name, expected in proof['file_sha256'].items():
        assert sha(analysis / name) == expected, name
    full = analysis.parent / 'full'
    full_proof = read(full / 'complete.json')
    assert full_proof['status'] == 'complete' and full_proof['preflight'] is False
    assert [full_proof[k] for k in ['environments', 'carriers', 'scenarios']] == [864, 17, 11]
    protocol = read(full / 'protocol.json')
    assert protocol['scenarios'] == SCENARIOS and protocol['methods'] == METHODS
    analysis_protocol = read(analysis / 'protocol.json')
    assert analysis_protocol['source_file_sha256'] == full_proof['file_sha256']
    for name, expected in full_proof['file_sha256'].items():
        assert sha(full / name) == expected, name

    summary_rows = read(analysis / 'summary.json')
    summary = {(r['scenario'], r['method']): r for r in summary_rows}
    assert len(summary) == len(summary_rows) == 33
    assert set(summary) == {(s, m) for s in SCENARIOS for m in METHODS}
    ber_rows = [r for r in read(analysis / 'comparisons.json') if r['metric'] == 'ber']
    comparisons = {(r['family'], r['left'], r['right']): r for r in ber_rows}
    assert len(comparisons) == len(ber_rows) == 52
    return proof, summary, comparisons


def point_interval(row, summary=False):
    """BER 转成百分数或百分点；保持统计脚本给出的区间方向。"""
    value = row['ber'] if summary else row['estimate']
    lower, upper = row['ci95']['ber'] if summary else row['ci95']
    assert all(math.isfinite(x) for x in [value, lower, upper]) and lower <= upper
    return [100 * value, 100 * lower, 100 * upper]


def render(analysis, output, zh_font):
    assert platform.node() == 'qyb-HuaShuo', 'Run result plotting on the experiment host.'
    proof, summary, comparisons = checked_inputs(analysis)
    assert zh_font.is_file(), zh_font
    # 延后导入绘图依赖，保证未完成统计不会产生任何图文件。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.unicode_minus': True})
    colors = ['#0072B2', '#D55E00', '#555555']
    markers = ['o', 's', '^']
    plotted = []
    outputs = []
    for language in ['en', 'zh']:
        cn = language == 'zh'
        # 中文 CFF/TTC 以轮廓嵌入，避免合并 PDF 时出现字符映射错误。
        plt.rcParams['pdf.fonttype'] = 3 if cn else 42
        font = FontProperties(fname=str(zh_font), size=10) if cn else FontProperties(size=10)
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 7.4), sharey=True)
        for i, scenario in enumerate(SCENARIOS):
            for j, method in enumerate(METHODS):
                value, lower, upper = point_interval(summary[scenario, method], summary=True)
                y = i + (j - 1) * .21
                left.hlines(y, lower, upper, color=colors[j], linewidth=1.2)
                left.plot(value, y, marker=markers[j], color=colors[j], markersize=4.2)
                if not cn:
                    plotted.append(dict(panel='absolute', scenario=scenario, method=method,
                                        value_percent=value, ci95_percent=[lower, upper]))
            for j, (family, reference) in enumerate([
                ('AF_vs_conventional', 'covariance_warm64'),
                ('AF_vs_scan', 'initial_select64'),
            ]):
                row = comparisons[family, scenario + ':cnn_warm64', scenario + ':' + reference]
                value, lower, upper = point_interval(row)
                y = i + (j - .5) * .22
                right.hlines(y, lower, upper, color=colors[j + 1], linewidth=1.2)
                right.plot(value, y, marker=markers[j + 1], color=colors[j + 1], markersize=4.2)
                if not cn:
                    plotted.append(dict(panel='contrast', scenario=scenario, family=family,
                                        value_pp=value, ci95_pp=[lower, upper]))
        left.set_yticks(range(len(SCENARIOS)), LABELS[language], fontproperties=font)
        left.set_ylim(len(SCENARIOS) - .45, -.55)
        right.tick_params(axis='y', left=False, labelleft=False)
        left.set_xlim(left=0)
        right.axvline(0, color='black', linestyle='--', linewidth=.8, zorder=0)
        left.set_title('(a) 三种控制方法' if cn else '(a) Three controllers', fontproperties=font)
        right.set_title('(b) 配对差值' if cn else '(b) Paired contrasts', fontproperties=font)
        left.set_xlabel('BER (%)', fontproperties=font)
        right.set_xlabel('A+F 减参照方法（百分点）' if cn else 'A+F minus reference (pp)', fontproperties=font)
        for ax in [left, right]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            for boundary in [.5, 3.5, 6.5]:
                ax.axhline(boundary, color='#bbbbbb', linewidth=.6, linestyle=':')
        names = ['A+F', '传统估计+F', '固定扫描'] if cn else ['A+F', 'Conventional+F', 'Fixed scan']
        handles = [Line2D([0], [0], marker=markers[j], color=colors[j], markersize=5,
                          linewidth=1.2, label=names[j]) for j in range(3)]
        fig.legend(handles=handles, loc='lower center', ncol=3, prop=font,
                   bbox_to_anchor=(.58, .055), frameon=False)
        note = ('全部条件均为 64 次探测；误差线为 95% 环境配对区间。' if cn else
                'All conditions use 64 probes; intervals use paired environment resampling.')
        fig.text(.58, .025, note, ha='center', fontproperties=font, fontsize=8.5)
        fig.subplots_adjust(left=.285, right=.985, top=.94, bottom=.15, wspace=.22)
        for suffix in ['pdf', 'png']:
            path = output / f'rq1_sensitivity_{language}.{suffix}'
            fig.savefig(path, dpi=160)
            outputs.append(path)
        plt.close(fig)
    assert len(plotted) == 55
    (output / 'plotted_values.json').write_text(json.dumps(plotted, ensure_ascii=False, indent=2) + '\n')
    outputs.append(output / 'plotted_values.json')
    # 绘图后复核输入，避免统计结果被同时替换后形成混合来源。
    for name, expected in proof['file_sha256'].items():
        assert sha(analysis / name) == expected, name
    record = dict(at=datetime.now(timezone.utc).isoformat(), status='rendered_pending_visual_review',
                  statistical_values_recomputed=False, all_scenarios_included=True,
                  renderer_sha256=sha(Path(__file__)), font_sha256=sha(zh_font),
                  analysis_complete_sha256=sha(analysis / 'complete.json'),
                  source_files=proof['file_sha256'],
                  outputs={p.name: sha(p) for p in outputs})
    (output / 'provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zh-font', type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis, args.output, args.zh_font)
