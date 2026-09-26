"""核对全量覆盖，输出六网络的中文结果表、曲线、星座图和可打开的HTML。"""
import argparse
import json
from pathlib import Path
import numpy as np
from compact_dataset import sha256
from generate_native_dataset import write_json
from summarize_compact_baselines import read_phase
from run_compact_baselines import METRICS

NAMES=dict(dnn='DNN',cnn='CNN',rescnn='ResCNN',transformer='Transformer',complex_cnn='复数CNN',jct='JCT适配')


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve(); protocol,arrays,identities=read_phase(root/'evaluation')
    study=json.loads((root/'study_protocol.json').read_text())
    if len(identities)!=study['test_environments']:raise ValueError('测试覆盖不足。')
    rows=[];by_frequency={};environment_nmse={}
    bootstrap=np.random.default_rng(20260924).integers(0,len(identities),(1000,len(identities)))
    figure=root/'figures';figure.mkdir(exist_ok=True)
    for mi,method in enumerate(protocol['methods']):
        data=arrays[mi,0];flat=data.reshape(-1,len(METRICS))
        row=dict(method=method,name=NAMES[method],samples=len(flat))
        for ki,key in enumerate(METRICS):
            row['mean_'+key]=None if np.isnan(flat[:,ki]).all() else float(flat[:,ki].mean())
        row['ber']=float(flat[:,METRICS.index('bit_errors')].sum()/flat[:,METRICS.index('bits_tested')].sum())
        if int(flat[:,METRICS.index('bits_tested')].sum())!=study['test_samples']*62:
            raise ValueError('比特计数不完整。')
        env_nmse=data[:,:,METRICS.index('payload_nmse')].mean(axis=1)
        environment_nmse[method]=env_nmse
        row['nmse_ci95']=np.quantile(env_nmse[bootstrap].mean(axis=1),[.025,.975]).tolist()
        row['training']=json.loads((root/method/'metadata.json').read_text())
        row['train_final_mse']=json.loads((root/method/'training_loss.json').read_text())[-1]['mse']
        rows.append(row);by_frequency[method]=[]
        for ci,fc in enumerate(protocol['carriers_ghz']):
            a=data[:,ci]
            by_frequency[method].append(dict(carrier_ghz=fc,
                ber=float(a[:,METRICS.index('bit_errors')].sum()/a[:,METRICS.index('bits_tested')].sum()),
                evm=float(a[:,METRICS.index('evm_percent')].mean()),
                snr=float(a[:,METRICS.index('effective_snr_db')].mean()),
                nmse=float(a[:,METRICS.index('payload_nmse')].mean())))
    differences={m:dict(mean=float((a-environment_nmse['dnn']).mean()),
        ci95=np.quantile((a-environment_nmse['dnn'])[bootstrap].mean(axis=1),[.025,.975]).tolist())
        for m,a in environment_nmse.items() if m!='dnn'}
    report=dict(status='complete',test_samples=study['test_samples'],test_environments=len(identities),
        methods=rows,by_frequency=by_frequency,paired_nmse_difference_vs_dnn=differences,
        uncertainty='95% percentile CI; bootstrap independent environments, not independent training runs',
        ranking='no test-based checkpoint or hyperparameter selection',
        snr_definition='-10 log10(payload NMSE): equivalent quality including distortion and equalizer error')
    write_json(root/'summary.json',report)
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for method,color in zip(protocol['methods'],colors):
        frequency=by_frequency[method]
        for ax,key,label in zip(axes,['ber','evm','snr'],['Bit error rate','EVM (%)','Equivalent SNR (dB)']):
            ax.plot([p['carrier_ghz'] for p in frequency],[p[key] for p in frequency],label=method,color=color,marker='.',lw=1.5)
            ax.set(xlabel='Carrier (GHz)',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figure/'frequency.png',dpi=200);fig.savefig(figure/'frequency.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for method,color in zip(protocol['methods'],colors):
        history=json.loads((root/method/'training_loss.json').read_text())
        ax.plot([r['epoch'] for r in history],[r['mse'] for r in history],label=method,color=color)
    ax.set(xlabel='Epoch',ylabel='Training control MSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figure/'training.png',dpi=200);fig.savefig(figure/'training.pdf');plt.close(fig)
    with np.load(root/'evaluation'/'signal_examples.npz') as examples:
        fig,axes=plt.subplots(4,6,figsize=(15,10))
        for ri,fc in enumerate([4,8,12,20]):
            for ci,method in enumerate(protocol['methods']):
                ax=axes[ri,ci];z=examples[method+'_'+str(fc)+'_received'];target=examples['target_'+str(fc)]
                ax.scatter(z.real,z.imag,s=9,alpha=.7)
                ax.scatter(target.real,target.imag,s=24,marker='x',color='#D55E00')
                limit=max(1.1,float(np.max(np.abs(np.r_[z.real,z.imag])))*1.05)
                ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal');ax.grid(alpha=.2)
                ax.set_title(method+' / '+str(fc)+' GHz',fontsize=9)
        fig.tight_layout();fig.savefig(figure/'constellation.png',dpi=200);fig.savefig(figure/'constellation.pdf');plt.close(fig)
    lines=['# 六种监督网络基线实验结果','',
        '全部使用117,504条训练样本和29,376条测试样本，固定seed=0、40轮、batch=256、Adam学习率0.001。最终轮检查点用于评测，没有验证集和测试集调参。',
        '', '所有方法读取同一2513维公开测量，输出128维控制，硬件量化与接收端评分一致；没有增加干扰、温漂或器件误差。', '',
        '| 方法 | BER↓ | EVM (%)↓ | 等效SNR (dB)↑ | 控制MSE↓ | 延时MAE (ps)↓ | 衰减MAE (dB)↓ | 参数量 | 训练时间 (s) | GPU单条推理 (ms) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['ber']:.6f} | {r['mean_evm_percent']:.3f} | {r['mean_effective_snr_db']:.3f} | {r['mean_normalized_control_mse']:.6f} | {r['mean_delay_mae_ps']:.2f} | {r['mean_attenuation_mae_db']:.3f} | {r['training']['parameters']:,} | {r['training']['training_seconds']:.1f} | {r['training']['inference_single_mean_seconds']*1000:.3f} |")
    lines += ['', 'BER越低表示误码越少；EVM越低表示收到的星座点更接近发送点。这里的等效SNR由独立数据块误差计算，包含噪声、失真和均衡误差，不是单独测量的热噪声SNR。', '',
        '控制标签由离线教师产生，并非已证明的全局最优。因此控制误差与接收质量需要分别评价。GPU推理计时不包含输入拷贝和标准化，完整控制流程还包含16次测量及硬件切换。', '',
        '模型是文献架构的监督控制适配。JCT包含并行CNN/Transformer和学习融合权重，不包含原文多级提前退出。复数CNN严格实现复权重乘法。', '',
        '置信区间对1728个独立测试环境整体重采样，保留各环境17个载频；单次训练不提供跨训练随机性的估计。', '',
        '![分频率结果](figures/frequency.png)','![训练曲线](figures/training.png)',
        '![固定第0号测试环境的星座图，橙色叉为发送符号](figures/constellation.png)']
    (root/'实验结果.md').write_text('\n'.join(lines)+'\n')
    # 自包含结构的本地结果页：图片采用相对路径，整个结果目录可直接部署。
    import html
    table=''.join('<tr>'+''.join('<td>'+html.escape(cell.strip())+'</td>' for cell in line.strip('|').split('|'))+'</tr>'
                  for line in lines if line.startswith('|') and not line.startswith('|---'))
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>六种深度学习基线</title><style>body{font:16px/1.7 system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#f5f7fb;color:#172b4d}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td{padding:10px;border-bottom:1px solid #dde3eb}tr:first-child{font-weight:bold;background:#e5eef9}img{width:100%;background:white;margin:20px 0}p{max-width:1100px}</style><h1>六种深度学习基线：相同数据、相同控制接口</h1>'
    page+='<p>'+html.escape(lines[2])+'</p><p>'+html.escape(lines[4])+'</p><div style="overflow:auto"><table>'+table+'</table></div>'
    page+='<p>BER和EVM越低越好，等效SNR越高越好。每种方法均接受29,376条完整测试。控制误差仅表示与教师标签的差距，实际信号质量需要结合下面的分频率曲线判断。</p>'
    for name,title in [('frequency','4–20 GHz 分频率接收质量'),('training','固定40轮训练过程'),('constellation','固定样例星座图：橙色叉为发送符号')]:
        page+='<h2>'+title+'</h2><img src="figures/'+name+'.png">'
    page+='<p>JCT为并行卷积/注意力和自适应融合模块的监督控制适配；不含多级提前退出。所有网络均不读取真实信道。详情与机器可读结果：<a href="实验结果.md">实验报告</a> · <a href="summary.json">summary.json</a> · <a href="study_protocol.json">冻结协议</a>。</p></html>'
    (root/'index.html').write_text(page)
    write_json(root/'result_manifest.json',dict(status='complete',files={str(p.relative_to(root)):sha256(p)
        for p in [root/'summary.json',root/'实验结果.md',root/'index.html',*figure.glob('*')]}))
    print(json.dumps(dict(status='complete',methods=[dict(method=r['method'],ber=r['ber'],evm=r['mean_evm_percent'],snr=r['mean_effective_snr_db']) for r in rows])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    summarize(p.parse_args().output)
