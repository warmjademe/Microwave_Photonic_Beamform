"""固定4/12/20 GHz和六种预设方法，绘制已逐项核查的确认信号。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE, require_host, sha256, source_record, verify_sources, write_json, now
from study_full_baselines.render_signal_examples import from_complex

SELECTED = [('covariance_response', '传统协方差估计（16次）'),
            ('complex_response_cnn', '复响应 CNN（16次）'),
            ('transformer', '直接控制 Transformer（16次）'),
            ('covariance_warm64', '传统估计＋反馈确认（64次）'),
            ('cnn_warm64', '复响应 CNN＋反馈确认（64次）'),
            ('codebook', '几何码本反馈（64次）')]
CARRIERS = [4, 12, 20]


def run(data, output):
    require_host()
    data, output = data.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('信号图像不覆盖已有结果。')
    meta = json.loads((data/'summary.json').read_text())
    protocol = json.loads((data/'protocol.json').read_text())
    if (meta['status'] not in ['complete', 'passed_preflight'] or not meta['all_quality_replays_passed']
            or meta['protocol_sha256'] != sha256(data/'protocol.json')
            or protocol['schema'] != 'confirmation-fixed-signal-v1'
            or meta['scope'] != protocol['scope']):
        raise ValueError('固定信号尚未完整核对。')
    verify_sources(protocol['source_sha256'])
    for entry in meta['entries']:
        for name, digest in entry['files'].items():
            if sha256(data/name) != digest:
                raise ValueError('信号导出文件改变。')
    if not set(CARRIERS).issubset(protocol['carriers_ghz']):
        raise ValueError('缺少预设绘图载频。')
    sources = source_record(['study_full_baselines/render_confirmation_signals.py',
        'study_full_baselines/render_signal_examples.py', 'study_full_baselines/CONFIRMATION_SIGNAL_PROTOCOL.md'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False, 'pdf.fonttype':42, 'svg.fonttype':'path'})
    output.mkdir(parents=True)
    scope = '新独立确认案例' if meta['final_confirmation'] else '旧案例演练（非新测试结果）'
    files, selected_sources = [], {}
    def save(fig, stem):
        for ext in ['png', 'pdf', 'svg']:
            path = output/(stem+'.'+ext)
            fig.savefig(path, dpi=200, bbox_inches='tight')
            files.append(path.name)
        plt.close(fig)
    for fc in CARRIERS:
        path = data/('carrier_%02d.json'%fc)
        page = json.loads(path.read_text())
        selected_sources[path.name] = sha256(path)
        lookup = {r['method'].split('/')[-1]:r for r in page['methods']}
        sent = from_complex(page['sent_qpsk'])
        limit = page['constellation_axis_limit']
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.7))
        for ax, (name, label) in zip(axes.ravel(), SELECTED):
            case = lookup[name]
            got = from_complex(case['received'])
            ax.scatter(sent.real, sent.imag, marker='x', s=75, color='#222222', label='发送点', zorder=3)
            ax.scatter(got.real, got.imag, s=25, alpha=.75, color='#0072B2', label='均衡后接收点')
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='同相 I', ylabel='正交 Q')
            ax.set_title(label+'\nBER %.2f%%；EVM %.2f%%'%(case['quality']['ber']*100, case['quality']['rms_evm_percent']), fontsize=10)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='upper right')
        fig.suptitle('%d GHz：同一发射数据、不同控制设置的接收星座'%fc, fontsize=15)
        fig.text(.5, .02, scope+'；固定索引0。图为固定噪声抽样0，指标为全部8次抽样。', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .05, 1, .95))
        save(fig, 'confirmation_constellation_%02d'%fc)
        time = np.asarray(page['time_us'])
        tx = from_complex(page['tx_iq_normalized'])
        fig, axes = plt.subplots(7, 1, figsize=(12.5, 13), sharex=True)
        axes[0].plot(time, tx.real, color='#0072B2', label='I')
        axes[0].plot(time, tx.imag, color='#D55E00', alpha=.7, label='Q')
        axes[0].set_title('发送：两个已知导频块＋一个独立数据块', fontsize=10)
        axes[0].set_ylabel('归一化幅度')
        axes[0].legend(fontsize=9, loc='upper right', ncol=2)
        for ax, (name, label) in zip(axes[1:], SELECTED):
            iq = from_complex(lookup[name]['raw_iq_a'])*1e6
            ax.plot(time, iq.real, color='#0072B2')
            ax.plot(time, iq.imag, color='#D55E00', alpha=.7)
            ax.set_title(label+'：均衡前接收电流', fontsize=10)
            ax.set_ylabel('电流（μA）')
        for ax in axes:
            ax.axvspan(0, 1.24, color='#0072B2', alpha=.04)
            ax.axvspan(1.24, 1.86, color='#009E73', alpha=.06)
            ax.grid(alpha=.15)
        axes[-1].set_xlabel('时间（μs）')
        fig.suptitle('%d GHz：从发送波形到合并后的接收I/Q'%fc, fontsize=15)
        fig.text(.5, .012, scope+'；固定抽样0。发送与接收量纲不同，分轴展示；浅绿为数据块。', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .97))
        save(fig, 'confirmation_waveform_%02d'%fc)
        fig, axes = plt.subplots(3, 4, figsize=(15.5, 12))
        for i, (name, label) in enumerate(SELECTED):
            pair = axes[i//2, (i%2)*2:(i%2)*2+2]
            for ax, key, title, upper in zip(pair, ['delay_ps', 'attenuation_db'],
                    ['有效延时（ps）', '可调光衰减（dB）'], [1484.375, 12]):
                im = ax.imshow(np.asarray(lookup[name][key]).reshape(8, 8), vmin=0, vmax=upper, cmap='viridis')
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
                ax.set_title(label+'\n'+title, fontsize=9)
                ax.set_xlabel('阵列列号')
                ax.set_ylabel('阵列行号')
        fig.suptitle('%d GHz：8×8面阵各支路实际采用的器件设置'%fc, fontsize=15)
        fig.text(.5, .014, scope+'；另含固定5 dB光损。显示有效软件控制码，物理开关接线尚未核验。', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .96))
        save(fig, 'confirmation_controls_%02d'%fc)
    for name in sources:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    verify_sources(sources)
    write_json(output/'build.json', dict(status='rendered_not_visually_verified', at=now(), scope=scope,
        final_confirmation=meta['final_confirmation'], input_summary_sha256=sha256(data/'summary.json'),
        selected_methods=[r[0] for r in SELECTED], carriers=CARRIERS, input_files=selected_sources,
        source_sha256=sources, files={name:sha256(output/name) for name in files}))
    print(json.dumps(dict(status='rendered', figures=9, final_confirmation=meta['final_confirmation'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
