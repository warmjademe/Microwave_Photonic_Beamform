"""从冻结结果绘制论文图；仅在华硕重放固定测试案例，不重训或重选控制。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
METHODS = ['codebook', 'coordinate', 'spsa', 'done', 'de', 'cnn_warm64']
LABELS = ['几何码本', '坐标搜索', 'SPSA', 'DONE', '差分进化', '本文方法']
COLORS = ['#0072B2', '#999999', '#CC79A7', '#E69F00', '#009E73', '#D55E00']
MARKERS = ['s', 'v', 'D', 'P', '^', 'o']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(project, output):
    if 'huashuo' not in socket.gethostname().lower():
        raise RuntimeError('论文统计绘图及固定接收重放在华硕执行。')
    if output.exists():
        raise FileExistsError('不覆盖已有绘图证据。')
    root = project/'dataset_simulation/baseline_results/20260926_final864_selected'
    complete = read(root/'analysis/complete.json')
    for name, digest in complete['file_sha256'].items():
        assert sha(root/'analysis'/name) == digest, name
    protocol = read(root/'protocol.json')
    audit = read(root/'analysis/audit.json')
    assert sha(root/'protocol.json') == audit['protocol_sha256']
    assert len(protocol['rows']) == 864
    summary = {r['method']: r for r in read(root/'analysis/summary.json')}
    groups = read(root/'analysis/groups.json')
    comparisons = read(root/'analysis/comparisons.json')
    assert all(summary[n]['measurement_budget'] == 64 for n in METHODS)
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import cairosvg
    font = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'svg.fonttype': 'path', 'font.size': 9})
    figures = []

    def save(fig, name):
        for ext in ['svg', 'png']:
            p = output/(name+'.'+ext)
            fig.savefig(p, dpi=220, bbox_inches='tight')
        # 字形转为矢量路径，避免 XeLaTeX 合并 CJK 子集字体时出现字符错配。
        cairosvg.svg2pdf(url=str(output/(name+'.svg')), write_to=str(output/(name+'.pdf')))
        figures.append(name+'.pdf')
        plt.close(fig)

    # 各面板来自同一全测试集；不平均单条件dB，不以不同量纲合成总分。
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.3))
    configs = [('carrier_ghz', 'ber', 100, '载频（GHz）', 'BER（%）'),
               ('carrier_ghz', 'rms_evm_percent', 1, '载频（GHz）', 'RMS EVM（%）'),
               ('power_bin', 'ber', 100, '名义接收功率档中心（dBm）', 'BER（%）'),
               ('power_bin', 'paired_output_snr_db', 1, '名义接收功率档中心（dBm）', '配对 SNR（dB）')]
    for ax, (group, metric, scale, xlabel, ylabel) in zip(axes.ravel(), configs):
        for name, label, color, marker in zip(METHODS, LABELS, COLORS, MARKERS):
            rows = sorted([r for r in groups if r['method'] == name and r['group'] == group], key=lambda r: r['value'])
            x = [r['value'] if group == 'carrier_ghz' else -102.5+5*r['value'] for r in rows]
            ax.plot(x, [scale*r[metric] for r in rows], label=label, color=color,
                    marker=marker, markersize=3, linewidth=1.1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, 'rq1_quality_profiles_64')

    # 以环境为单位显示指标关系，每个点先聚合该环境全部17载频。
    from study_full_baselines.paired_statistics import sufficient, metrics
    with np.load(root/'analysis/environment_metrics.npz') as f:
        env_values = metrics(sufficient(f['metrics']))
        names = f['methods'].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.7))
    for name, label, color in zip(METHODS, LABELS, COLORS):
        v = env_values[:, names.index(name)]
        axes[0].scatter(v[:, 3], v[:, 0]*100, s=7, alpha=.3, color=color,
                        label=label, rasterized=True)
        axes[1].scatter(v[:, 4], v[:, 0]*100, s=7, alpha=.3, color=color, rasterized=True)
    for ax, xlabel in zip(axes, ['均衡后 RMS EVM（%）', '均衡前配对 SNR（dB）']):
        ax.set(xlabel=xlabel, ylabel='环境 BER（%）')
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, 'rq1_metric_relations_64')

    # 固定索引0、4/12/20 GHz，在读取案例成绩前固定选择规则。
    # 重放已提交控制，保存全部8次噪声的31个业务符号，并逐项核对原成绩。
    from study_full_baselines.common import verify_sources
    from study_full_baselines.confirmation_receiver import scoring_engines
    from study_full_baselines.export_signal_examples import photonic_trace, checked_equal, quality
    from our_method_response_control.train import precision
    import torch
    assert torch.cuda.is_available()
    precision()
    verify_sources(protocol['source_sha256'])
    row = next(r for r in protocol['rows'] if r['index'] == 0)
    folder = root/'records/environment_00000'
    environment = read(folder/'environment.json')
    bundle = Path(protocol['runtime_bundle'])
    assert sha(bundle/'manifest.json') == protocol['runtime_manifest_sha256']
    with np.load(bundle/'public.npz') as f:
        public = {k: f[k].copy() for k in f.files}
    cases = []
    for fc in [4, 12, 20]:
        p = folder/('carrier_%02d.npz'%fc)
        assert audit['raw_file_sha256'][str(p.relative_to(root))] == sha(p)
        with np.load(p) as f:
            codes, recorded = f['control_code'].copy(), f['metrics'].copy()
        engine, clean, payload = scoring_engines(environment, fc, public, protocol['backend'])
        traces = []
        for name in METHODS:
            idx = protocol['methods'].index(name)
            trace = photonic_trace(engine, clean, codes[idx], payload, environment['seed'], fc)
            checked_equal(trace['metrics'], recorded[idx, :10], name)
            traces.append(trace)
        received = np.stack([t['received_qpsk'] for t in traces])
        np.savez_compressed(output/('constellation_%02d.npz'%fc), methods=METHODS,
                            received_qpsk=received, sent_qpsk=payload,
                            metrics=np.stack([t['metrics'] for t in traces]))
        # 248点来自8次噪声重放；所有方法使用同一坐标、同一符号与配对噪声。
        limit = max(1.2, float(np.max(np.maximum(abs(received.real), abs(received.imag))))*1.05)
        fig, axes = plt.subplots(2, 3, figsize=(9.1, 6.0), sharex=True, sharey=True)
        target = np.tile(payload, 8)
        quadrant = (target.real > 0).astype(int)+2*(target.imag > 0).astype(int)
        palette = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
        for ax, name, label, trace in zip(axes.ravel(), METHODS, LABELS, traces):
            got = trace['received_qpsk'].ravel()
            for q in range(4):
                ax.scatter(got.real[quadrant == q], got.imag[quadrant == q], s=9,
                           alpha=.48, color=palette[q], edgecolors='none')
            points = np.asarray([(-1-1j), (1-1j), (-1+1j), (1+1j)])/np.sqrt(2)
            ax.scatter(points.real, points.imag, marker='x', s=50, color='black', linewidths=1.1)
            ax.axhline(0, color='grey', linewidth=.6, linestyle='--')
            ax.axvline(0, color='grey', linewidth=.6, linestyle='--')
            v = quality(trace['metrics'])
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='I', ylabel='Q',
                   title=label+'\nBER %.2f%%；EVM %.2f%%'%(100*v['ber'], v['rms_evm_percent']))
            ax.grid(alpha=.15)
        fig.tight_layout()
        save(fig, 'rq1_constellation_%02d_64'%fc)
        cases.append(dict(carrier_ghz=fc, environment_id=row['environment_id'],
                          methods={n: quality(t['metrics']) for n, t in zip(METHODS, traces)}))
        print(json.dumps({'carrier':fc, 'replay':'matches_committed_scores'}), flush=True)
    verify_sources(protocol['source_sha256'])
    record = dict(source_protocol_sha256=sha(root/'protocol.json'),
                  analysis_complete_sha256=sha(root/'analysis/complete.json'),
                  environment_index=0, environment=environment, cases=cases,
                  measurements=64, methods=METHODS, draws=8, symbols_per_draw=31,
                  controls_reoptimized=False, figures={n: sha(output/n) for n in figures})
    (output/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (output/'rq1_summary.json').write_text(json.dumps([summary[n] for n in METHODS], indent=2))
    (output/'rq1_comparisons.json').write_text(json.dumps([r for r in comparisons if r['family']=='same_budget_64'], indent=2))
    print(json.dumps({'status':'complete','figures':len(figures)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.project.resolve(), args.output.resolve())
