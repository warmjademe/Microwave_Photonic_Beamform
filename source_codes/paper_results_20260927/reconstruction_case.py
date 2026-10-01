"""从真实冻结测试中选取中位改善案例，展示同输入下的 QPSK 恢复差异。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host, sha256, now, verify_sources
from paper_results_20260927.build_uniform64 import METHODS, LABELS
from study_uniform64.run import load_original
from study_full_baselines.confirmation_receiver import scoring_engines
from study_full_baselines.export_signal_examples import photonic_trace, checked_equal, quality
from our_method_response_control.train import precision


RULE = dict(carrier_ghz=12, measurement_budget=64,
            eligibility='our BER and EVM both strictly lower than every one of the thirteen baselines',
            strength='relative EVM reduction versus the lowest baseline EVM in the same case',
            selection='eligible case nearest median strength; break ties by test index',
            role='outcome-conditioned illustrative improvement case; not an unbiased full-test estimate',
            fixed_inputs='same environment, payload, antenna noise, eight APD noise seeds, equalizer and plot axes',
            if_empty='stop; do not relax conditions or manufacture data',
            excluded='no retraining, parameter tuning, independent per-method examples or changed noise')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def select(project, output):
    require_host()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # 在读取结果前保存明确规则，保留筛选分母、所有符合条件的案例及选择过程。
    save(output / 'selection_rule.json', dict(RULE, frozen_at=now(), script_sha256=sha256(__file__)))
    root = project / 'dataset_simulation/baseline_results/20260927_uniform64_all13'
    protocol = read(root / 'protocol.json')
    complete = read(root / 'analysis/complete.json')
    for file, digest in complete['file_sha256'].items():
        assert sha256(root / 'analysis' / file) == digest
    with np.load(root / 'analysis/environment_metrics.npz') as f:
        values, names = f['metrics'], f['methods'].tolist()
    values = values[:, 12 - 4]
    assert values.shape == (864, 14, 13)
    ix = [names.index(n) for n in METHODS]
    values = values[:, ix]
    ber, evm = values[:, :, 0] / values[:, :, 1], np.sqrt(values[:, :, 2]) * 100
    minimum_ber, minimum_evm = ber[:, :-1].min(1), evm[:, :-1].min(1)
    gains = (minimum_evm - evm[:, -1]) / minimum_evm
    eligible = np.flatnonzero((ber[:, -1] < minimum_ber) & (evm[:, -1] < minimum_evm))
    if not len(eligible):
        raise RuntimeError('没有符合已固定规则的案例；停止绘图。')
    median = float(np.median(gains[eligible]))
    index = min(eligible.tolist(), key=lambda i: (abs(gains[i] - median), protocol['rows'][i]['index']))
    def item(i):
        return dict(row=protocol['rows'][i], relative_evm_reduction=float(gains[i]),
                    minimum_baseline_evm_percent=float(minimum_evm[i]), our_evm_percent=float(evm[i, -1]),
                    minimum_baseline_ber_percent=float(100 * minimum_ber[i]), our_ber_percent=float(100 * ber[i, -1]),
                    methods={n:dict(ber_percent=float(100 * ber[i, j]), evm_percent=float(evm[i, j]))
                             for j, n in enumerate(METHODS)})
    chosen = dict(rule=RULE, chosen=item(index), eligible_cases=len(eligible), examined_cases=864,
                  eligible_median_relative_evm_reduction=median,
                  all_eligible=[item(int(i)) for i in eligible],
                  source_protocol_sha256=sha256(root / 'protocol.json'),
                  source_analysis_complete_sha256=sha256(root / 'analysis/complete.json'),
                  selected_at=now())
    save(output / 'selection.json', chosen)
    print(json.dumps({k:v for k,v in chosen.items() if k != 'all_eligible'}, ensure_ascii=False), flush=True)


def render(project, output):
    require_host(); precision()
    choice = read(output / 'selection.json')
    assert choice['rule'] == RULE
    assert read(output / 'selection_rule.json')['script_sha256'] == sha256(__file__)
    root = project / 'dataset_simulation/baseline_results/20260927_uniform64_all13'
    protocol = read(root / 'protocol.json')
    assert sha256(root / 'protocol.json') == choice['source_protocol_sha256']
    verify_sources(protocol['source_sha256'])
    row = choice['chosen']['row']; fc = 12
    env, old, old_sha = load_original(row, fc, protocol)
    new_path = root / 'records' / ('environment_%05d' % row['index']) / 'carrier_12.npz'
    assert sha256(new_path) == read(root / 'analysis/audit.json')['raw_file_sha256'][str(new_path.relative_to(root))]
    with np.load(new_path) as f:
        new = {k:f[k].copy() for k in f.files}
    old_names = read(Path(protocol['source_output']) / 'protocol.json')['methods']
    with np.load(Path(protocol['runtime_bundle']) / 'public.npz') as f:
        public = {k:f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    engine, clean, payload = scoring_engines(env, fc, public, protocol['backend'])
    traces, codes, raw = [], [], []
    for name in METHODS:
        src, names = (new, protocol['methods']) if name in protocol['methods'] else (old, old_names)
        i = names.index(name)
        code, expected = src['control_code'][i], src['metrics'][i, :10]
        trace = photonic_trace(engine, clean, code, payload, env['seed'], fc)
        checked_equal(trace['metrics'], expected, name)
        v = quality(trace['metrics'])
        checked_equal(v['ber'] * 100, choice['chosen']['methods'][name]['ber_percent'], name + ' BER')
        checked_equal(v['rms_evm_percent'], choice['chosen']['methods'][name]['evm_percent'], name + ' EVM')
        traces.append(trace); codes.append(code); raw.append(expected)
    received = np.stack([t['received_qpsk'] for t in traces])
    np.savez_compressed(output / 'reconstruction_case.npz', methods=METHODS, sent_qpsk=payload,
                        received_qpsk=received, control_code=np.array(codes), metrics=np.array(raw))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import cairosvg
    font = subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus':False, 'svg.fonttype':'path', 'font.size':10})
    ideal = np.array([-1-1j, 1-1j, -1+1j, 1+1j]) / np.sqrt(2)
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    sent = np.tile(payload, 8)
    quadrant = (sent.real >= 0).astype(int) + 2 * (sent.imag >= 0).astype(int)
    limit = max(1.2, float(max(abs(received.real).max(), abs(received.imag).max())) * 1.05)
    fig, grid = plt.subplots(4, 4, figsize=(10.5, 10.2), sharex=True, sharey=True)
    axes = grid.ravel()
    for ax in axes[:15]:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='I', ylabel='Q')
        ax.axhline(0, color='grey', linewidth=.5, linestyle='--')
        ax.axvline(0, color='grey', linewidth=.5, linestyle='--')
        ax.grid(alpha=.15)
        ax.scatter(ideal.real, ideal.imag, marker='x', s=34, color='black', linewidths=1, zorder=5)
    for q in range(4):
        axes[0].scatter(ideal[q].real, ideal[q].imag, s=90, color=colors[q], marker='o', alpha=.6)
    axes[0].set_title('发送端：理想 QPSK 星座\n相同业务符号作为恢复目标', fontsize=10)
    for i, (name, label) in enumerate(zip(METHODS, LABELS)):
        ax = axes[i+1]; got = received[i].ravel()
        for q, color in enumerate(colors):
            ax.scatter(got.real[quadrant == q], got.imag[quadrant == q], s=7, alpha=.55, color=color, edgecolors='none')
        v = quality(traces[i]['metrics'])
        ax.set_title(label + '\nBER %.2f%% / EVM %.2f%%' % (100*v['ber'], v['rms_evm_percent']), fontsize=10)
        if name == 'cnn_warm64':
            for spine in ax.spines.values():
                spine.set_color('#0072B2'); spine.set_linewidth(1.6)
    axes[-1].axis('off')
    axes[-1].text(.0, .92, '同输入恢复对照\n测试索引 %d · 12 GHz\n每方法 64 次探测\n每图 248 个接收符号\n\n黑叉：应恢复到的位置\n颜色：实际发送类别\n虚线：比特判决边界\n\n中位改善案例（%d/864）' %
                  (row['index'], choice['eligible_cases']), transform=axes[-1].transAxes, va='top', fontsize=10, linespacing=1.5)
    fig.tight_layout()
    stem = 'rq1_constellation_all13_12_64'
    for ext in ['svg', 'png']:
        fig.savefig(output / (stem + '.' + ext), dpi=240, bbox_inches='tight')
    cairosvg.svg2pdf(url=str(output / (stem + '.svg')), write_to=str(output / (stem + '.pdf')))
    plt.close(fig)
    manifest = dict(status='passed', created_at=now(), methods=METHODS, case=row, carrier_ghz=fc,
                    environment=env, selection=choice['chosen'], selection_rule=RULE,
                    eligible_cases=choice['eligible_cases'], examined_cases=864,
                    raw_new_sha256=sha256(new_path), raw_old_sha256=old_sha,
                    source_builder_sha256=sha256(__file__), source_analysis_complete_sha256=choice['source_analysis_complete_sha256'],
                    metrics_replayed_and_matched=True, input_noise_and_payload_paired=True,
                    files={p.name:sha256(p) for p in output.iterdir() if p.suffix in ['.pdf','.svg','.png','.npz']})
    save(output / 'manifest.json', manifest)
    print(json.dumps(dict(status='passed', index=row['index'], methods=14, environment=env['power_dbm'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['select', 'render'])
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    (select if a.action == 'select' else render)(a.project, a.output)
