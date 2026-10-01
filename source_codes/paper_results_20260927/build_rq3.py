"""从既有训练规模统计绘制验证曲线，不重新训练或选择模型。"""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

assert 'huashuo' in socket.gethostname().lower()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import cairosvg

project = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=False)
base = project/'dataset_simulation'
p1 = base/'diagnostics/20260926_training_scale_budget/curves.json'
p2 = base/'baseline_results/20260925_full_baselines/evaluation_scaling_3456/summary.json'
small = json.loads(p1.read_text())['records']
large = json.loads(p2.read_text())['records']
font = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                     'axes.unicode_minus':False, 'svg.fonttype':'path', 'font.size':10})
fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.7))
curves = []
for curve, suffix, label, color, marker in [
        ('cnn_fixed40', 'fixed_epochs', '固定 40 轮', '#0072B2', 'o'),
        ('cnn_equal_updates', 'equal_updates', '固定 9,200 次更新', '#D55E00', 's')]:
    rows = sorted([r for r in small if r['curve']==curve], key=lambda r:r['training_environments'])
    last = next(r for r in large if r['group']=='all' and r['method']=='response_n3456_'+suffix+'__base_2sweeps')
    rows.append(dict(last, training_environments=3456))
    assert [r['training_environments'] for r in rows]==[216,432,864,1728,3456]
    for ax, metric, scale, ylabel in zip(axes, ['ber','rms_evm_percent'],[100,1],['BER（%）','RMS EVM（%）']):
        ax.plot(range(5), [scale*r[metric] for r in rows], color=color, marker=marker, label=label)
        ax.set(xticks=range(5), xticklabels=[216,432,864,1728,3456], xlabel='训练环境数', ylabel=ylabel)
        ax.grid(alpha=.2)
    curves.append(dict(curve=curve, rows=rows))
axes[0].legend(frameon=False)
fig.tight_layout()
for ext in ['svg','png']:
    fig.savefig(output/('rq3_training_quality.'+ext),dpi=220,bbox_inches='tight')
cairosvg.svg2pdf(url=str(output/'rq3_training_quality.svg'), write_to=str(output/'rq3_training_quality.pdf'))
plt.close(fig)
(output/'manifest.json').write_text(json.dumps(dict(
    sources={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [p1,p2]},
    evaluation='216 validation environments; 17 carriers; component A; B off',curves=curves),indent=2))
print('Rendered verified training-scale curves.')
