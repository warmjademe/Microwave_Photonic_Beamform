"""复现论文图 3--10：读取冻结数值，校验后绘图，不运行训练或接收实验。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
METHODS = ['initial_select64', 'codebook', 'coordinate', 'spsa', 'done', 'de',
           'mlp_feedback64', 'dnn_feedback64', 'cnn_feedback64', 'rescnn_feedback64',
           'transformer_feedback64', 'complex_cnn_feedback64', 'jct_feedback64', 'cnn_warm64']
NAMES = {
    'en': ['Fixed scan', 'Geometric codebook', 'Coordinate search', 'SPSA', 'DONE', 'DE',
           'MLP + feedback', 'DNN + feedback', 'CNN + feedback', 'ResCNN + feedback',
           'Transformer + feedback', 'Complex CNN + feedback', 'JCT + feedback', 'RLF'],
    'zh': ['固定扫描', '几何码本', '坐标搜索', 'SPSA', 'DONE', '差分进化',
           'MLP＋反馈', 'DNN＋反馈', 'CNN＋反馈', 'ResCNN＋反馈',
           'Transformer＋反馈', '复数 CNN＋反馈', 'JCT＋反馈', 'RLF'],
}
COLORS = ['#7f7f7f', '#0072B2', '#999933', '#CC79A7', '#E69F00', '#009E73',
          '#332288', '#44AA99', '#882255', '#88CCEE', '#AA4499', '#661100', '#117733', '#D55E00']
MARKERS = ['x', 's', 'v', 'D', 'P', '^', 'o', '<', '>', 'h', '*', '+', 'd', 'o']
SCENARIOS = ['nominal', 'delay_x1p5', 'angular_x2', 'power_minus6', 'rin_minus150',
             'rin_minus140', 'rin_minus130', 'delay_error_5ps', 'delay_error_10ps',
             'loss_error_0p5db', 'loss_error_1db']
FIGURES = [
    (3, 'rq1_metric_relations_all13_64', 'fig:metric-relations', 'BER、EVM 与 SNR 的关系'),
    (4, 'rq1_quality_profiles_all13_64', 'fig:frequency', '载频与接收功率分组'),
    (5, 'array_ttd_phase', 'fig:array-phase', '固定阵列的真时延与相移方向图'),
    (6, 'rq1_selection_origins_64', 'fig:selection-origins', '最终控制的选择来源'),
    (7, 'rq1_constellation_all13_12_64', 'fig:rq1-constellation', '相同输入的 QPSK 星座'),
    (8, 'rq1_sensitivity', 'fig:rq1-sensitivity', '传播和接收扰动的敏感性'),
    (9, 'rq2_af_ablation', 'fig:rq2-af', 'CRL 与 CFS 的消融'),
    (10, 'rq3_training_quality', 'fig:rq3-training', '训练规模与更新次数'),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    # 不用 assert 作为输入完整性门禁，python -O 也必须执行这些检查。
    if not condition:
        raise ValueError(message)


def metrics(raw):
    """在频率/环境上先聚合计数与功率，最后开方或取对数。"""
    return np.stack([raw[..., 0] / raw[..., 1], raw[..., 3] / raw[..., 4],
                     raw[..., 5] / raw[..., 6], 100 * np.sqrt(raw[..., 2] / raw[..., 10]),
                     10 * np.log10(raw[..., 7] / raw[..., 8])], axis=-1)


def load_checked(inputs):
    manifest = read(inputs / 'manifest.json')
    for name, info in manifest['files'].items():
        check(sha(inputs / name) == info['sha256'], f'Input hash mismatch: {name}')
    with np.load(inputs / 'rq1_metrics.npz', allow_pickle=False) as a:
        raw = a['metrics'].copy()
        order = a['methods'].tolist()
        env_ids = a['environment_ids'].tolist()
    check(raw.shape == (864, 17, 14, 13), 'RQ1 shape mismatch')
    check(set(order) == set(METHODS) and len(set(env_ids)) == 864, 'Method or environment set mismatch')
    # 原数组的存储顺序与论文展示顺序不同，必须按保存的方法名重排。
    raw = raw[:, :, [order.index(name) for name in METHODS], :]
    # 前十项是误码数、比特数、NMSE 等充分统计量；第十一项记录聚合的载频数。
    total = raw[..., :10].sum(axis=1)
    suff = np.concatenate([total, np.full(total.shape[:-1] + (1,), 17)], axis=-1)
    env_metrics = metrics(suff)
    overall = metrics(suff.sum(axis=0))
    summary = {r['method']: r for r in read(inputs / 'rq1_summary.json')}
    keys = ['ber', 'ser', 'block_error_rate', 'rms_evm_percent', 'paired_output_snr_db']
    for i, name in enumerate(METHODS):
        np.testing.assert_allclose(overall[i], [summary[name][k] for k in keys], rtol=1e-12)
    groups = read(inputs / 'rq1_groups.json')
    # 独立复算每个载频曲线的点，避免用错测试集或直接平均 dB。
    carrier_suff = np.concatenate([raw[..., :10].sum(axis=0), np.full((17, 14, 1), 864)], axis=-1)
    carrier_values = metrics(carrier_suff)
    for row in groups:
        if row['group'] == 'carrier_ghz':
            np.testing.assert_allclose(carrier_values[int(row['value']) - 4, METHODS.index(row['method'])],
                                       [row[k] for k in keys], rtol=1e-12)
    origins = read(inputs / 'selection_origins.json')
    for row in origins.values():
        check(row['total'] == 864 * 17 and sum(row[k] for k in ['initial', 'model', 'codebook']) == row['total'],
              'Invalid selection counts')
    ablation = read(inputs / 'ablation_summary.json')
    np.testing.assert_allclose(ablation[3]['ber'], summary['cnn_warm64']['ber'], rtol=1e-12)
    sensitivity = read(inputs / 'sensitivity_summary.json')
    check(len(sensitivity) == 33, 'All 11 scenarios and 3 controllers are required')
    return dict(manifest=manifest, env_metrics=env_metrics, overall=overall, env_ids=env_ids,
                groups=groups, origins=origins, ablation=ablation, sensitivity=sensitivity)


def find_font(explicit):
    if explicit:
        check(explicit.is_file(), f'Font not found: {explicit}')
        return explicit
    candidates = [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                  Path('/System/Library/Fonts/PingFang.ttc')]
    candidates += list((Path.home() / 'Library/texlive').glob('*/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf'))
    for p in candidates:
        if p.is_file():
            return p
    for entry in font_manager.fontManager.ttflist:
        if any(x in entry.name for x in ['Noto Sans CJK', 'Microsoft YaHei', 'SimHei', 'FandolHei']):
            return Path(entry.fname)
    raise FileNotFoundError('Chinese font missing. Pass --zh-font /path/to/font or --language en.')


class Renderer:
    def __init__(self, inputs, output, language, data, font):
        self.inputs, self.output, self.language, self.data = inputs, output, language, data
        self.cn = language == 'zh'
        self.names = NAMES[language]
        self.output.mkdir(parents=True, exist_ok=True)
        fonts = ['DejaVu Sans']
        if self.cn:
            font_manager.fontManager.addfont(str(font))
            fonts.append(font_manager.FontProperties(fname=str(font)).get_name())
        plt.rcParams.update({'font.family': fonts, 'font.size': 10, 'axes.labelsize': 10,
                             'axes.titlesize': 11, 'legend.fontsize': 9, 'pdf.fonttype': 3 if self.cn else 42,
                             'svg.fonttype': 'path', 'axes.unicode_minus': False})
        self.pdf = PdfPages(output / 'all_analysis_figures.pdf', metadata={'Title': 'Paper analysis figures'})

    def text(self, en, zh):
        return zh if self.cn else en

    def save(self, fig, stem, sources):
        fig.savefig(self.output / (stem + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None})
        fig.savefig(self.output / (stem + '.png'), dpi=220, bbox_inches='tight')
        self.pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return {'figure': stem, 'source_inputs': sources, 'pdf_sha256': sha(self.output / (stem + '.pdf'))}

    def relations(self):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
        for i, (name, color) in enumerate(zip(self.names, COLORS)):
            v = self.data['env_metrics'][:, i]
            for ax, k in zip(axes, [3, 4]):
                ax.scatter(v[:, k], v[:, 0] * 100, s=5, alpha=.25, color=color,
                           label=name, rasterized=True)
        for ax, xlabel in zip(axes, [self.text('Post-equalisation RMS EVM (%)', '均衡后 RMS EVM（%）'),
                                    self.text('Pre-equalisation paired SNR (dB)', '均衡前配对 SNR（dB）')]):
            ax.set(xlabel=xlabel, ylabel=self.text('Environment BER (%)', '环境 BER（%）'))
            ax.grid(alpha=.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .78))
        return self.save(fig, FIGURES[0][1], ['rq1_metrics.npz'])

    def profiles(self):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.5))
        configs = [('carrier_ghz', 'ber', 100), ('carrier_ghz', 'rms_evm_percent', 1),
                   ('power_bin', 'ber', 100), ('power_bin', 'paired_output_snr_db', 1)]
        for ax, (group, metric, factor) in zip(axes.flat, configs):
            for i, name in enumerate(METHODS):
                rows = sorted([r for r in self.data['groups'] if r['method'] == name and r['group'] == group],
                              key=lambda r: r['value'])
                check(len(rows) == (17 if group == 'carrier_ghz' else 6), 'Missing group')
                x = [r['value'] if group == 'carrier_ghz' else -102.5 + 5 * r['value'] for r in rows]
                ax.plot(x, [factor * r[metric] for r in rows], label=self.names[i], color=COLORS[i],
                        marker=MARKERS[i], linewidth=1.8 if i == 13 else .9, markersize=3,
                        linestyle='--' if '_feedback64' in name else '-')
            ax.set_xlabel(self.text('Carrier (GHz)', '载频（GHz）') if group == 'carrier_ghz' else
                          self.text('Received-power bin centre (dBm)', '接收功率档中心（dBm）'))
            ax.set_ylabel({'ber': 'BER (%)', 'rms_evm_percent': 'RMS EVM (%)',
                           'paired_output_snr_db': self.text('Paired SNR (dB)', '配对 SNR（dB）')}[metric])
            ax.grid(alpha=.2)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .86))
        return self.save(fig, FIGURES[1][1], ['rq1_groups.json'])

    def array(self):
        # 方向图是确定性阵列因子：仅对已保存的几何/量化延时绘制角度切面。
        cfg = read(self.inputs / 'array_geometry.json')
        with np.load(self.inputs / 'array_response.npz', allow_pickle=False) as a:
            steering = a['steering_deg'].copy()
            codes = a['delay_codes'].copy()
            saved = a['normalised_power'].copy()
        x, y = np.meshgrid(np.arange(cfg['nx']) - (cfg['nx'] - 1) / 2,
                           np.arange(cfg['ny']) - (cfg['ny'] - 1) / 2)
        positions = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=-1) * cfg['pitch_m']
        look = np.linspace(-90, 90, 3601)
        # baseline_common.controls.direction 的约定：[cos(el)sin(az), sin(el), cos(el)cos(az)]。
        directions = np.stack([np.sin(np.deg2rad(look)), np.zeros_like(look), np.cos(np.deg2rad(look))])
        propagation = positions @ directions / 299792458.0
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.8), sharex=True, sharey=True, layout='constrained')
        exported = []
        for row, fc in enumerate([4e9, 12e9, 20e9]):
            for col, target in enumerate([0, 35]):
                idx = np.flatnonzero((steering == [target, 0]).all(axis=1))[0]
                tau = codes[idx] / cfg['sample_rate_hz']
                ax = axes[row, col]
                for f, cf, color, ls, label in [
                    (fc, fc, '.35', '-', self.text('Centre: both controls', '中心频率：两种控制')),
                    (fc + 50e6, fc + 50e6, '#0072B2', '-', self.text('+50 MHz: true delay', '+50 MHz：真时延')),
                    (fc + 50e6, fc, '#D55E00', '--', self.text('+50 MHz: matched phase', '+50 MHz：匹配相移')),
                ]:
                    power = abs(np.mean(np.exp(2j * np.pi * (f * propagation - cf * tau[:, None])), axis=0)) ** 2
                    ax.plot(look, 10 * np.log10(np.maximum(power, 1e-9)), color=color, ls=ls, lw=1, label=label)
                    # 切面在目标方向的值必须与原483021条件诊断保存的值相同。
                    at = np.argmin(abs(look - target))
                    carrier = int(fc / 1e9) - 4
                    offset = 16 if f == fc else 32
                    control = 0 if cf == f else 1
                    np.testing.assert_allclose(power[at], saved[carrier, idx, offset, control], rtol=1e-11, atol=1e-13)
                    exported.extend(zip(np.full(len(look), fc / 1e9), np.full(len(look), target),
                                        np.full(len(look), (f - fc) / 1e6), np.full(len(look), control), look, power))
                ax.axvline(target, color='.65', lw=.6, ls=':')
                ax.set_title(self.text(f'{fc / 1e9:g} GHz; steering {target} deg', f'{fc / 1e9:g} GHz；指向 {target}°'))
                ax.set(ylim=(-40, 1), xlim=(-90, 90), xticks=[-90, -45, 0, 45, 90])
                ax.grid(alpha=.2)
                if col == 0:
                    ax.set_ylabel(self.text('Normalised power (dB)', '归一化功率（dB）'))
                if row == 2:
                    ax.set_xlabel(self.text('Azimuth (deg); elevation = 0', '方位角（°）；俯仰角为 0'))
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', frameon=False)
        write_csv(self.output.parent / 'plot_data/array_cuts.csv',
                  ['carrier_ghz', 'steering_azimuth_deg', 'offset_mhz', 'control_0ttd_1phase', 'look_deg', 'linear_power'], exported)
        return self.save(fig, FIGURES[2][1], ['array_response.npz', 'array_geometry.json'])

    def origins(self):
        origins = self.data['origins']
        names = list(origins)
        fig, ax = plt.subplots(figsize=(10.6, 4.5))
        left = np.zeros(len(names))
        for key, color, en, zh in [('initial', '#999999', 'Shared initial controls', '共同初始控制'),
                                   ('model', '#D55E00', 'New model candidate', '模型新增候选'),
                                   ('codebook', '#0072B2', 'New codebook controls', '码本新增控制')]:
            values = np.array([origins[n][key] / origins[n]['total'] * 100 for n in names])
            ax.barh(range(len(names)), values, left=left, label=self.text(en, zh), color=color)
            left += values
        np.testing.assert_allclose(left, 100)
        ax.set(yticks=range(len(names)), yticklabels=[self.names[METHODS.index(n)] for n in names],
               xlim=(0, 100), xlabel=self.text('Source of selected control (%)', '最终控制来源（%）'))
        ax.invert_yaxis()
        ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(.5, 1), frameon=False)
        fig.tight_layout()
        return self.save(fig, FIGURES[3][1], ['selection_origins.json'])

    def constellation(self):
        with np.load(self.inputs / 'constellation.npz', allow_pickle=False) as a:
            got, sent, codes, raw = [a[k].copy() for k in ['received_qpsk', 'sent_qpsk', 'control_code', 'metrics']]
            check(a['methods'].tolist() == METHODS, 'Constellation method order changed')
        check(got.shape == (14, 8, 31), 'Constellation sample count changed')
        groups, values = [], []
        for i, name in enumerate(METHODS):
            # 使用发送符号及收到的 I/Q 重新计算该案例的 BER/EVM。
            errors = np.count_nonzero((got[i].real >= 0) != (sent.real >= 0)) + np.count_nonzero((got[i].imag >= 0) != (sent.imag >= 0))
            nmse = float(np.mean(abs(got[i] - sent) ** 2) / np.mean(abs(sent) ** 2))
            check(errors == raw[i, 0] and raw[i, 1] == 496, 'Constellation BER mismatch')
            np.testing.assert_allclose(nmse, raw[i, 2], rtol=1e-12)
            values.append([name, int(errors), 496, 100 * errors / 496, 100 * np.sqrt(nmse)])
            for group in groups:
                if np.array_equal(codes[i], codes[group[0]]):
                    np.testing.assert_array_equal(got[i], got[group[0]])
                    group.append(i)
                    break
            else:
                groups.append([i])
        check([len(g) for g in groups] == [12, 1, 1], 'Constellation group membership changed')
        check(METHODS[groups[1][0]] == 'coordinate' and METHODS[groups[2][0]] == 'cnn_warm64', 'Case changed')
        write_csv(self.output.parent / 'plot_data/constellation_metrics.csv',
                  ['method', 'bit_errors', 'bits', 'ber_percent', 'rms_evm_percent'], values)
        ideal = np.array([-1 - 1j, 1 - 1j, -1 + 1j, 1 + 1j]) / np.sqrt(2)
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
        target = np.tile(sent, 8)
        quadrant = (target.real >= 0).astype(int) + 2 * (target.imag >= 0).astype(int)
        limit = max(1.2, float(max(abs(got.real).max(), abs(got.imag).max())) * 1.05)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), layout='constrained')
        titles = [self.text('(a) Transmitted QPSK', '(a) 发送的 QPSK'),
                  self.text('(b) 12 baselines: identical output', '(b) 12 个基线：相同输出'),
                  self.text('(c) Coordinate search', '(c) 坐标搜索'), '(d) RLF']
        for ax, title in zip(axes.flat, titles):
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='I', ylabel='Q', title=title)
            ax.axhline(0, c='gray', lw=.7, ls='--')
            ax.axvline(0, c='gray', lw=.7, ls='--')
            ax.grid(alpha=.18)
            ax.scatter(ideal.real, ideal.imag, c='black', marker='x', s=52, zorder=5)
        for i, color in enumerate(colors):
            axes.flat[0].scatter(ideal[i].real, ideal[i].imag, s=125, c=color, alpha=.6)
        for group, ax in zip(groups, list(axes.flat)[1:]):
            v = got[group[0]].ravel()
            for q, color in enumerate(colors):
                ax.scatter(v.real[quadrant == q], v.imag[quadrant == q], s=14, c=color, alpha=.65, edgecolors='none')
        return self.save(fig, FIGURES[4][1], ['constellation.npz', 'constellation_manifest.json'])

    def sensitivity(self):
        summary = {(r['scenario'], r['method']): r for r in self.data['sensitivity']}
        contrasts = {(r['left'], r['right']): r for r in read(self.inputs / 'sensitivity_comparisons.json')
                     if r['metric'] == 'ber' and r['family'] in ['AF_vs_scan', 'AF_vs_conventional']}
        names = ['cnn_warm64', 'covariance_warm64', 'initial_select64']
        labels = ['Nominal', 'Path delay ×1.5', 'Angular spread ×2', 'Signal power -6 dB',
                  'RIN -150 dB/Hz', 'RIN -140 dB/Hz', 'RIN -130 dB/Hz',
                  'Delay error σ = 5 ps', 'Delay error σ = 10 ps', 'Loss error σ = 0.5 dB', 'Loss error σ = 1 dB']
        if self.cn:
            labels = ['标称条件', '路径时延 ×1.5', '角度扩展 ×2', '信号功率 -6 dB',
                      'RIN -150 dB/Hz', 'RIN -140 dB/Hz', 'RIN -130 dB/Hz',
                      '延时误差 σ = 5 ps', '延时误差 σ = 10 ps', '损耗误差 σ = 0.5 dB', '损耗误差 σ = 1 dB']
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 7.4), sharey=True)
        colors, markers = ['#0072B2', '#D55E00', '#555555'], ['o', 's', '^']
        for i, scenario in enumerate(SCENARIOS):
            for j, method in enumerate(names):
                r = summary[scenario, method]
                v = 100 * r['ber']; lo, hi = np.array(r['ci95']['ber']) * 100
                left.errorbar(v, i + (j - 1) * .18, xerr=[[v - lo], [hi - v]], fmt=markers[j], color=colors[j], capsize=3, ms=4)
            for j, method in enumerate(names[1:], 1):
                r = contrasts[f'{scenario}:cnn_warm64', f'{scenario}:{method}']
                v = 100 * r['estimate']; lo, hi = np.array(r['ci95']) * 100
                right.errorbar(v, i + (j - 1.5) * .22, xerr=[[v - lo], [hi - v]], fmt=markers[j], color=colors[j], capsize=3, ms=4)
        left.set(yticks=range(11), yticklabels=labels, ylim=(10.55, -.55), xlabel='BER (%)', xlim=(0, None))
        right.tick_params(axis='y', left=False, labelleft=False)
        right.axvline(0, color='black', ls='--', lw=.8)
        right.set_xlabel(self.text('RLF minus reference (pp)', 'RLF 减参照方法（百分点）'))
        left.set_title(self.text('(a) Three controllers', '(a) 三种控制方法'))
        right.set_title(self.text('(b) Paired contrasts', '(b) 配对差值'))
        for ax in [left, right]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', alpha=.2)
            for boundary in [.5, 3.5, 6.5]:
                ax.axhline(boundary, color='#bbbbbb', lw=.6, ls=':')
        labels = ['RLF', self.text('Conventional+CFS', '传统估计+CFS'), self.text('Fixed scan', '固定扫描')]
        handles = [Line2D([0], [0], marker=markers[j], color=colors[j], ms=5, label=labels[j]) for j in range(3)]
        fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.58, .055), frameon=False)
        fig.text(.58, .025, self.text('64 probes; paired 95% environment intervals.', '64 次探测；95% 环境配对区间。'), ha='center', fontsize=9)
        fig.subplots_adjust(left=.285, right=.985, top=.94, bottom=.15, wspace=.22)
        return self.save(fig, FIGURES[5][1], ['sensitivity_summary.json', 'sensitivity_comparisons.json'])

    def ablation(self):
        rows = self.data['ablation']
        pairs = {r['id']: r for r in read(self.inputs / 'ablation_comparisons.json') if r['metric'] == 'ber'}
        keys = ['A_without_F', 'A_with_F', 'F_without_A', 'F_with_A', 'AF_vs_fixed_scan64']
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 7.1))
        v = np.array([r['ber'] for r in rows]) * 100
        ci = np.array([r['ci95']['ber'] for r in rows]) * 100
        top.bar(range(5), v, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=4, width=.6,
                color=['#999999', '#4477AA', '#CCBB44', '#228833', '#AA3377'])
        labels = ['Traditional\n16 probes', 'CRL\n16 probes', 'Traditional+CFS\n64 probes', 'RLF\n64 probes', 'Fixed scan\n64 probes']
        if self.cn:
            labels = ['传统估计\n16 次', 'CRL\n16 次', '传统估计+CFS\n64 次', 'RLF\n64 次', '固定扫描\n64 次']
        top.set(xticks=range(5), xticklabels=labels, ylabel='BER (%)', ylim=(0, ci.max() + 1.9))
        for i, value in enumerate(v):
            top.text(i, ci[i, 1] + .17, f'{value:.4f}', ha='center', fontsize=10)
        top.set_title(self.text('(a) Receive quality and 95% intervals', '(a) 接收质量与 95% 区间'))
        labels = ['Add CRL; no CFS (16 probes)', 'Add CRL; with CFS (64 probes)',
                  'Add CFS; no CRL (16 to 64)', 'Add CFS; with CRL (16 to 64)', 'RLF vs. fixed scan (64)']
        if self.cn:
            labels = ['加入 CRL，无 CFS（16 次）', '加入 CRL，有 CFS（64 次）',
                      '加入 CFS，无 CRL（16→64 次）', '加入 CFS，有 CRL（16→64 次）', 'RLF 对比固定扫描（64 次）']
        for i, key in enumerate(keys):
            r = pairs[key]
            # 原记录为“加入后减加入前”；图展示降低值，故需同时翻转点估计及区间。
            gain, lo, hi = -100 * r['estimate'], -100 * r['ci95'][1], -100 * r['ci95'][0]
            bottom.errorbar(gain, i, xerr=[[gain - lo], [hi - gain]], fmt='o', capsize=4, color='#4477AA')
        bottom.set(yticks=range(5), yticklabels=labels,
                   xlabel=self.text('BER reduction (percentage points)', 'BER 降低值（百分点）'))
        bottom.invert_yaxis()
        bottom.axvline(0, color='black', lw=.8, ls='--')
        bottom.set_title(self.text('(b) Paired effects and 95% intervals', '(b) 配对效应与 95% 区间'))
        for ax in [top, bottom]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y' if ax is top else 'x', alpha=.2)
            ax.set_axisbelow(True)
        fig.tight_layout(h_pad=2.4)
        return self.save(fig, FIGURES[6][1], ['ablation_summary.json', 'ablation_comparisons.json'])

    def training(self):
        curves = read(self.inputs / 'training_curves.json')['curves']
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
        for curve, color, marker, label in zip(curves, ['#0072B2', '#D55E00'], ['o', 's'],
                                               [self.text('40 epochs', '40 轮训练'), self.text('9200 updates', '9200 次更新')]):
            rows = curve['rows']
            check([r['training_environments'] for r in rows] == [216, 432, 864, 1728, 3456], 'Training sweep changed')
            for ax, key, factor in zip(axes, ['ber', 'rms_evm_percent'], [100, 1]):
                ax.plot(range(5), [r[key] * factor for r in rows], color=color, marker=marker, label=label)
        for ax, label in zip(axes, [self.text('Validation BER (%)', '验证 BER（%）'), self.text('Validation RMS EVM (%)', '验证 RMS EVM（%）')]):
            ax.set(xticks=range(5), xticklabels=['216', '432', '864', '1728', '3456'],
                   xlabel=self.text('Training environments', '训练环境数'), ylabel=label)
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
        fig.tight_layout()
        return self.save(fig, FIGURES[7][1], ['training_curves.json'])


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(columns); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=HERE / 'inputs')
    p.add_argument('--output', type=Path, default=HERE / 'generated')
    p.add_argument('--language', choices=['both', 'en', 'zh'], default='both')
    p.add_argument('--zh-font', type=Path)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    data = load_checked(args.inputs)
    if args.verify_only:
        print('Input hashes, RQ1 aggregates, carrier curves, selection counts and RQ2 identity: passed')
        return
    # 只覆盖本程序拥有的同名图片；不修改 latex/ 中的论文图或原始实验材料。
    args.output.mkdir(parents=True, exist_ok=True)
    languages = ['en', 'zh'] if args.language == 'both' else [args.language]
    font = find_font(args.zh_font) if 'zh' in languages else None
    records = []
    for language in languages:
        renderer = Renderer(args.inputs, args.output / language, language, data, font)
        try:
            for method in ['relations', 'profiles', 'array', 'origins', 'constellation', 'sensitivity', 'ablation', 'training']:
                record = getattr(renderer, method)()
                records.append(dict(language=language, **record))
        finally:
            renderer.pdf.close()
    # 保存可直接核对图3每一个点的数据，不从PDF反向抄数。
    rows = []
    for e, env in enumerate(data['env_ids']):
        for m, name in enumerate(METHODS):
            ber, ser, bler, evm, snr = data['env_metrics'][e, m]
            rows.append([env, name, ber * 100, ser * 100, bler * 100, evm, snr])
    write_csv(args.output / 'plot_data/environment_metrics.csv',
              ['environment_id', 'method', 'ber_percent', 'ser_percent', 'bler_percent', 'rms_evm_percent', 'paired_snr_db'], rows)
    proof = {'status': 'generated_and_numeric_checks_passed', 'at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'script_sha256': sha(Path(__file__)), 'input_manifest_sha256': sha(args.inputs / 'manifest.json'),
        'font_sha256': sha(font) if font else None, 'new_training_or_receiver_evaluation': False,
        'figures': records, 'figure_registry': [dict(number=n, stem=s, label=l, description=d) for n, s, l, d in FIGURES]}
    (args.output / 'manifest.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    parts = ['<!doctype html><meta charset="utf-8"><title>Paper analysis figures</title>',
             '<style>body{font:16px sans-serif;max-width:1050px;margin:35px auto}img{width:100%}section{margin:45px 0}a{margin-right:20px}</style>',
             '<h1>论文数据分析图 / Paper analysis figures</h1><p>输入来自冻结结果；图号对应当前论文。</p>']
    for number, stem, label, description in FIGURES:
        parts.append(f'<section><h2>Figure {number}: {description}</h2><p>{label}</p>')
        for language in languages:
            parts.append(f'<a href="{language}/{stem}.pdf">{language.upper()} PDF</a><a href="{language}/{stem}.png">{language.upper()} PNG</a>')
        parts.append(f'<img src="{languages[0]}/{stem}.png" alt="Figure {number}"></section>')
    (args.output / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(f'Generated {len(records)} figures (PDF + PNG), 8 pages per language. Gallery: {args.output / "index.html"}')


if __name__ == '__main__':
    main()
