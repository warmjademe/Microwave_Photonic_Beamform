"""合并控制与接收点完全一致的面板，保留每种方法真实指标，不替换其最终控制。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from paper_results_20260927.build_uniform64 import METHODS, LABELS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(project, output):
    if socket.gethostname() != 'qyb-HuaShuo':
        raise RuntimeError('数值检查与绘图在 huashuo 执行')
    if output.exists():
        raise FileExistsError(output)
    source = project / 'dataset_simulation/diagnostics/20260927_constellation_reconstruction'
    manifest = json.loads((source / 'manifest.json').read_text())
    data = source / 'reconstruction_case.npz'
    assert sha(data) == manifest['files'][data.name]
    with np.load(data, allow_pickle=False) as f:
        names, codes = f['methods'].tolist(), f['control_code'].copy()
        received, sent, metrics = f['received_qpsk'].copy(), f['sent_qpsk'].copy(), f['metrics'].copy()
    assert names == METHODS and received.shape == (14, 8, 31)
    groups = []
    for i in range(len(names)):
        for group in groups:
            if np.array_equal(codes[i], codes[group[0]]):
                group.append(i)
                break
        else:
            groups.append([i])
    # 只有控制码和每个复数接收点确实相同，才允许合并面板。
    for group in groups:
        for i in group:
            np.testing.assert_array_equal(received[i], received[group[0]])
            np.testing.assert_allclose(metrics[i], metrics[group[0]], rtol=1e-12, atol=1e-28)
    assert [len(g) for g in groups] == [12, 1, 1]
    assert names[groups[1][0]] == 'coordinate' and names[groups[2][0]] == 'cnn_warm64'
    ber, evm = [], []
    for i in range(14):
        bad_i = (received[i].real >= 0) != (sent.real >= 0)
        bad_q = (received[i].imag >= 0) != (sent.imag >= 0)
        bit_errors = int(bad_i.sum() + bad_q.sum())
        nmse = np.mean(abs(received[i] - sent[None]) ** 2) / np.mean(abs(sent) ** 2)
        assert bit_errors == metrics[i, 0] and metrics[i, 1] == 496
        np.testing.assert_allclose(nmse, metrics[i, 2], rtol=1e-12, atol=0)
        ber.append(100 * bit_errors / 496)
        evm.append(float(100 * np.sqrt(nmse)))
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import cairosvg
    font = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus':False, 'svg.fonttype':'path', 'font.size':12})
    fig = plt.figure(figsize=(10.3, 10.3), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .82])
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    ideal = np.array([-1-1j, 1-1j, -1+1j, 1+1j]) / np.sqrt(2)
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    target = np.tile(sent, 8)
    quadrant = (target.real >= 0).astype(int) + 2 * (target.imag >= 0).astype(int)
    limit = max(1.2, float(max(abs(received.real).max(), abs(received.imag).max())) * 1.05)
    for ax in axes:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='I', ylabel='Q')
        ax.axhline(0, color='grey', linewidth=.65, linestyle='--')
        ax.axvline(0, color='grey', linewidth=.65, linestyle='--')
        ax.grid(alpha=.15)
        ax.scatter(ideal.real, ideal.imag, marker='x', s=48, color='black', linewidths=1.2, zorder=5)
    for i, color in enumerate(colors):
        axes[0].scatter(ideal[i].real, ideal[i].imag, s=150, color=color, alpha=.55)
    axes[0].set_title('(a) 发送端：理想 QPSK 星座\n黑叉为同色接收点应恢复的位置', fontsize=12)
    titles = ['(b) 相同控制的 12 个基线', '(c) 坐标搜索', '(d) 本文方法']
    mapping = {}
    for group, ax, title, panel in zip(groups, axes[1:], titles, ['b', 'c', 'd']):
        i = group[0]
        got = received[i].ravel()
        for q, color in enumerate(colors):
            ax.scatter(got.real[quadrant == q], got.imag[quadrant == q], s=13, alpha=.62, color=color, edgecolors='none')
        ax.set_title(title + '\nBER %.4f%% / EVM %.4f%%' % (ber[i], evm[i]), fontsize=12)
        for j in group:
            mapping[names[j]] = panel
    for spine in axes[-1].spines.values():
        spine.set_color('#0072B2'); spine.set_linewidth(1.6)
    # 每个方法保留一行指标及对应面板，合并绘图不合并或重写实验结果。
    for col, indices in enumerate([range(7), range(7, 14)]):
        ax = fig.add_subplot(grid[2, col]); ax.axis('off')
        rows = [[LABELS[i], '('+mapping[names[i]]+')', '%.4f' % ber[i], '%.4f' % evm[i]] for i in indices]
        table = ax.table(cellText=rows, colLabels=['方法', '面板', 'BER (%)', 'EVM (%)'],
                         colWidths=[.46, .12, .20, .22], cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10.8); table.scale(1, 1.6)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#d5d5d5'); cell.set_linewidth(.4)
            if r == 0:
                cell.set_facecolor('#eef1f4')
            elif c == 0:
                cell.get_text().set_ha('left')
    fig.suptitle('同输入下的恢复对照：仅合并完全相同的输出\n测试索引 363 · 12 GHz · 每方法 64 次探测 · 每面板 248 个接收符号', fontsize=13)
    stem = 'rq1_constellation_all13_12_64'
    for extension in ['png', 'svg']:
        fig.savefig(output / (stem+'.'+extension), dpi=240, bbox_inches='tight')
    cairosvg.svg2pdf(url=str(output/(stem+'.svg')), write_to=str(output/(stem+'.pdf')))
    plt.close(fig)
    rows = [dict(method=names[i], panel=mapping[names[i]], ber_percent=ber[i], evm_percent=evm[i]) for i in range(14)]
    with (output/'method_panel_mapping.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    record = dict(status='passed', source_case_manifest_sha256=sha(source/'manifest.json'),
                  source_case_arrays_sha256=sha(data), script_sha256=sha(__file__),
                  source_case=manifest['case'], selection_rule=manifest['selection_rule'],
                  eligible_cases=manifest['eligible_cases'], examined_cases=manifest['examined_cases'],
                  group_members=[[names[i] for i in g] for g in groups], methods=rows,
                  identical_controls_and_received_arrays_verified=True, metrics_independently_recomputed=True,
                  original_controls_or_samples_modified=False, worse_baseline_controls_substituted=False,
                  files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='passed', methods=14, received_panels=3, source_case=manifest['case']['environment_id'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); main(a.project, a.output)
