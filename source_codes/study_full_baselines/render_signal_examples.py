"""固定案例预览图；完整逐方法数字由export_signal_examples导出的JSON提供。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, write_json, now, source_record

SELECTED = [('classic/ttd_das', '初始实测选择（16次）'),
            ('classic/codebook', '几何码本反馈（64次）'),
            ('learned_evaluation/transformer', '直接控制 Transformer（16次）'),
            ('learned_evaluation/complex_response_cnn', '复响应 CNN＋逐路控制（16次）')]


def from_complex(a):
    a = np.asarray(a)
    return a[..., 0]+1j*a[..., 1]


def run(data, output, carriers):
    require_host()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
    meta = json.loads((data/'summary.json').read_text())
    if meta['status'] not in ['complete', 'passed_preflight'] or not meta['all_quality_replays_passed']:
        raise ValueError('案例数据尚未完整核对。')
    for entry in meta['entries']:
        for name, digest in entry['files'].items():
            if sha256(data/name) != digest:
                raise ValueError('案例数据改变：'+name)
    output.mkdir(parents=True, exist_ok=False)
    files = []; selected_sources = {}
    scope = '预检查子集' if meta['status'] == 'passed_preflight' else '旧测试探索性案例'
    def save(fig, name):
        for ext in ['png', 'pdf', 'svg']:
            path = output/(name+'.'+ext); fig.savefig(path, dpi=200, bbox_inches='tight'); files.append(path)
        plt.close(fig)
    for fc in carriers:
        path = data/('carrier_%02d.json' % fc); page = json.loads(path.read_text())
        selected_sources[path.name] = sha256(path)
        lookup = {v['method']: v for v in page['methods']}
        sent = from_complex(page['sent_qpsk']); limit = page['constellation_axis_limit']
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
        for ax, (method, label) in zip(axes.ravel(), SELECTED):
            case = lookup[method]; got = from_complex(case['received']); q = case['quality']
            ax.scatter(sent.real, sent.imag, marker='x', s=95, color='#222222', label='发送点', zorder=3)
            ax.scatter(got.real, got.imag, s=30, alpha=.75, color='#0072B2', label='接收点（均衡后）')
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel='同相 I', ylabel='正交 Q', aspect='equal')
            ax.grid(alpha=.2); ax.legend(loc='upper right', fontsize=8)
            ax.set_title(label+'\nBER %.2f%%；EVM %.2f%%' % (100*q['ber'], q['rms_evm_percent']), fontsize=11)
        fig.suptitle('%d GHz：同一发射数据的接收星座' % fc, fontsize=15)
        fig.text(.5, .015, scope+'；固定测试索引0。图：第1次噪声抽样；指标：全部8次抽样。', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .05, 1, .95)); save(fig, 'constellation_%02d' % fc)
        time = np.asarray(page['time_us']); tx = from_complex(page['tx_iq_normalized'])
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(time, tx.real, color='#0072B2', label='I'); axes[0].plot(time, tx.imag, color='#D55E00', alpha=.7, label='Q')
        axes[0].set_ylabel('归一化幅度'); axes[0].set_title('发射基带：2个已知导频块＋1个独立数据块', fontsize=11)
        axes[0].legend(loc='upper right', ncol=2)
        for ax, (method, label) in zip(axes[1:], SELECTED):
            iq = from_complex(lookup[method]['raw_iq_a'])*1e6
            ax.plot(time, iq.real, color='#0072B2'); ax.plot(time, iq.imag, color='#D55E00', alpha=.7)
            ax.set_ylabel('电流（μA）'); ax.set_title(label+'：均衡前接收I/Q', fontsize=10)
        for ax in axes:
            ax.axvspan(.0, .62, alpha=.045, color='#0072B2')
            ax.axvspan(.62, 1.24, alpha=.045, color='#0072B2')
            ax.axvspan(1.24, 1.86, alpha=.06, color='#009E73'); ax.grid(alpha=.15)
        axes[-1].set_xlabel('时间（μs）')
        fig.suptitle('%d GHz：真实基带波形与接收电流' % fc, fontsize=15)
        fig.text(.5, .012, '发射与接收量纲不同，分轴展示；浅绿区域为独立数据块。'+scope+'，固定抽样0。', ha='center')
        fig.tight_layout(rect=(0, .04, 1, .96)); save(fig, 'waveform_%02d' % fc)
        fig, axes = plt.subplots(4, 2, figsize=(10, 13))
        for row, (method, label) in zip(axes, SELECTED):
            for ax, key, title, upper in zip(row, ['delay_ps', 'attenuation_db'], ['有效延时（ps）', '可调光衰减（dB）'], [1484.375, 12]):
                im = ax.imshow(np.asarray(lookup[method][key]).reshape(8, 8), vmin=0, vmax=upper, cmap='viridis')
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
                ax.set_title(label+'\n'+title, fontsize=10); ax.set_xlabel('阵列列号'); ax.set_ylabel('阵列行号')
        fig.suptitle('%d GHz：8×8面阵的64路器件控制' % fc, fontsize=15)
        fig.text(.5, .014, '另外包含固定5 dB光损；显示有效软件控制码，物理开关接线尚未核验。'+scope+'。', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, .96)); save(fig, 'controls_%02d' % fc)
    write_json(output/'build.json', dict(status='rendered_not_visually_verified', scope=scope,
        source_sha256=source_record(['study_full_baselines/render_signal_examples.py']),
        input_summary_sha256=sha256(data/'summary.json'), input_files=selected_sources,
        selected_methods=[v[0] for v in SELECTED], carriers=carriers,
        files={p.name: sha256(p) for p in files}, at=now()))
    print(json.dumps(dict(status='rendered', figures=len(carriers)*3)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--carriers', type=int, nargs='+', default=[4, 12, 20]); a = p.parse_args()
    run(a.data, a.output, a.carriers)
