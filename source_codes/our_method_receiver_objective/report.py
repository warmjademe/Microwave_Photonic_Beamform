"""核对已完成的训练内诊断并生成中文表格；不重新选择控制或训练模型。"""
import argparse
import json
import subprocess
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,sha256,write_json,now,LEVELS

LABELS = ['传统估计＋MMSE控制','传统估计＋BER控制','CNN估计＋MMSE控制',
          'CNN估计＋BER控制','真实响应＋MMSE（参考）','真实响应＋BER（参考）','离线教师（参考）']


def run(root):
    require_host()
    protocol=json.loads((root/'protocol.json').read_text())
    complete=json.loads((root/'complete.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if complete['status']!='complete' or protocol['test_used'] or protocol['cases']!=72:
        raise ValueError('诊断未完整完成或错误读取测试。')
    if sha256(root/'all_metrics.npz')!=complete['all_metrics_sha256'] or sha256(root/'summary.json')!=complete['summary_sha256']:
        raise ValueError('原始数组或汇总哈希不符。')
    with np.load(root/'all_metrics.npz') as f:
        m=f['metrics'];codes=f['control_code'];seconds=f['controller_seconds']
    if m.shape!=(72,7,10) or codes.shape!=(72,7,128) or np.any(codes<0) or np.any(codes>LEVELS):
        raise ValueError('结果形状或档位不符。')
    numeric=0
    for i,r in enumerate(summary['records']):
        a=m[:,i]
        expected={'ber':a[:,0].sum()/a[:,1].sum(), 'ser':a[:,3].sum()/a[:,4].sum(),
            'rms_evm_percent':100*np.sqrt(a[:,2].mean()),
            'paired_output_snr_db':10*np.log10(a[:,7].sum()/a[:,8].sum()),
            'block_error_rate':a[:,5].sum()/a[:,6].sum(),
            'mean_stage_b_seconds':seconds[:,i].mean()}
        for k,v in expected.items():
            if not np.isclose(v,r[k],rtol=1e-12,atol=0):raise ValueError('汇总值不同：'+k)
            numeric+=1
    for i in range(72):
        p=root/'records'/('%03d.npz'%i);meta=json.loads(p.with_suffix('.json').read_text())
        if meta['sha256']!=sha256(p):raise ValueError('逐案例哈希改变。')
        with np.load(p) as f:
            if not np.array_equal(f['metrics'],m[i]) or not np.array_equal(f['control_code'],codes[i]):
                raise ValueError('逐案例与全表不同。')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    colors=['#0072B2','#56B4E9','#D55E00','#E69F00','#999999','#BBBBBB','#666666']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    y=np.arange(7)
    for ax,key,title in zip(axes,['ber','rms_evm_percent'],['误码率 BER（%），越低越好','星座误差 EVM（%），越低越好']):
        vals=np.array([r[key] for r in summary['records']])*(100 if key=='ber' else 1)
        ax.barh(y,vals,color=colors)
        ax.set_yticks(y,LABELS if ax is axes[0] else ['']*7)
        ax.invert_yaxis();ax.set_title(title);ax.axhline(3.5,color='#555555',ls='--',lw=.8)
        ax.set_xlim(0,max(vals)*1.18)
        for yi,v in zip(y,vals):ax.text(v+max(vals)*.02,yi,f'{v:.2f}',va='center')
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('训练内机制诊断：24 个环境 × 3 个载频\n上方四项同为16次公开测量；下方三项使用额外信息，仅作参考',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(root/('receiver_objective_comparison.'+ext),dpi=300)
    plt.close(fig)
    lines=['# 估计误差与接收目标的训练内诊断','',
        '仅使用预先固定的24个训练环境、4/12/20 GHz，共72案例。复用原864环境模型，没有新增训练或读取测试环境。以下是训练内机制证据，不是扩展版最终成绩。','',
        '| 方法 | BER（%）↓ | EVM（%）↓ | 输出SNR（dB）↑ |', '|---|---:|---:|---:|']
    for label,r in zip(LABELS,summary['records']):
        lines.append(f"| {label} | {100*r['ber']:.4f} | {r['rms_evm_percent']:.4f} | {r['paired_output_snr_db']:.4f} |")
    lines+=['','前四项只有控制目标不同，使用相同起点、两轮扫描、128个合法控制量和16次初始测量。后三项有额外信息，不能作为在线方法或严格上界。所有方法的控制确定后，在相同frame5、8次配对噪声上评价。','',
        '| 配对比较（前者减后者） | BER差值（百分点） | 诊断性95%区间 | 胜/平/负环境数 |','|---|---:|---|---|']
    names=dict(zip(protocol['methods'],LABELS))
    for r in summary['paired']:
        lo,hi=r['ci95_pp']
        lines.append(f"| {names[r['a']]} − {names[r['b']]} | {r['ber_difference_pp']:.4f} | [{lo:.4f}, {hi:.4f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['','区间按24个环境整体重采样，属于固定训练小组的诊断描述，未覆盖重新训练的不确定性。控制耗时保存在summary.json；当前主机有并行任务，且CNN响应读取缓存，不能将该耗时当作完整在线速度排名。','',
        '原MMSE的两项控制逐码复现，接收指标按相对容差1e-12复现。逐案例原始结果与汇总已重新核对；全部负结果保留。']
    (root/'报告.md').write_text('\n'.join(lines)+'\n')
    audit=dict(status='passed',at=now(),cases=72,independent_training_environments=24,
        fields_recomputed=numeric,legal_control_codes=int(codes.size),test_used=False,
        source_summary_sha256=sha256(root/'summary.json'),
        report_source_sha256=sha256(Path(__file__)),visual_review='pending')
    write_json(root/'report_audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    run(p.parse_args().input.resolve())
