"""只读取本轮已完成的结果，核查覆盖并生成 RQ2 复核报告和图。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, write_json, now


def read(path):
    return json.loads(Path(path).read_text())


def main(output):
    require_host()
    import numpy as np
    done=read(output/'complete.json'); protocol=read(output/'protocol.json')
    assert done['status']=='complete' and not done['preflight']
    assert done['environments']==864 and done['carriers']==17 and done['methods']==5
    assert sha256(output/'records.json')==done['records_sha256']
    assert hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()==done['protocol_sha256']
    records=read(output/'records.json')
    expected=[r['environment_id'] for r in protocol['rows']]
    assert len(set(expected))==864
    assert [r['environment_id'] for r in records]==expected
    assert protocol['carriers']==list(range(4,21))
    for name,digest in protocol['source_sha256'].items():
        assert sha256(output/'source_snapshot'/name)==digest
    checked=0
    selection=dict(initial=0,learned_candidate=0,codebook_fallback=0)
    payload=dict(better=0,equal=0,worse=0)
    for env in records:
        assert len(env['carriers'])==17
        assert [r['carrier_ghz'] for r in env['carriers']]==protocol['carriers']
        for entry in env['carriers']:
            assert sha256(output/entry['path'])==entry['sha256']
            assert entry['protocol_sha256']==done['protocol_sha256']
            for field in ['input_bitwise_equal','all_controls_bitwise_equal','integer_counts_equal',
                          'metrics_within_frozen_tolerance','RQ1_primary_reproduced',
                          'proposal_matches_direct_output','feedback_trace_equal']:
                assert entry['checks'][field],field
            assert entry['checks']['joint_correction_used'] is False
            with np.load(output/entry['path'],allow_pickle=False) as f:
                trace=f['cnn_warm64__trace_control_code'];scores=f['cnn_warm64__trace_scores']
                winner=int(scores.argmax());proposal=f['cnn_warm64__proposal_control_code']
                role='initial' if winner<16 else ('learned_candidate' if np.array_equal(trace[winner],proposal) else 'codebook_fallback')
                selection[role]+=1
                delta=int(f['metrics'][3,0])-int(f['metrics'][1,0])
                payload['better' if delta<0 else 'worse' if delta>0 else 'equal']+=1
            checked+=1
    assert checked==864*17
    rows=read(output/'summary.json'); comparisons=read(output/'comparisons.json')
    ber={r['id']:r for r in comparisons if r['metric']=='ber'}
    lines=['','## 如何理解本轮结果','',
        'A 为复响应学习；F 为候选实测与选择。两者分工是“提出器件设置”与“通过链路反馈选取设置”。',
        '旧的联合测量校正没有参与，不将旧 B 的结果转用为 F 的证据。','']
    for key,label in [('A_with_F','在共同 64 次测量下，去掉 A、改用传统响应估计'),
                      ('F_with_A','去掉 F、直接执行 A 生成的设置，并将测量数从 64 减为 16')]:
        r=ber[key];lo,hi=-100*r['ci95'][1],-100*r['ci95'][0]
        lines.append(f"- {label}：BER 变化为 {-100*r['estimate']:+.6f} 个百分点，配对 95% 区间 [{lo:+.6f}, {hi:+.6f}]。")
    r=ber['AF_vs_fixed_scan64'];lo,hi=[100*x for x in r['ci95']]
    lines+=['',f"相同 64 次测量时，A+F 相对固定扫描的 BER 差为 {100*r['estimate']:+.6f} 个百分点，95% 区间 [{lo:+.6f}, {hi:+.6f}]。",'']
    r=ber['interaction'];lo,hi=[100*x for x in r['ci95']]
    lines.append(f"交互差（A+F−A−传统+F+传统）为 {100*r['estimate']:+.6f} 个百分点，95% 区间 [{lo:+.6f}, {hi:+.6f}]。")
    if r['ci95'][0]>0:
        lines.append('该交互为正，表示按 BER 的绝对百分点计算，A 在已有反馈时的增益小于无反馈时；不能把这组结果描述为超加性的协同增益。')
    lines+=['','EVM 与输出 SNR 也分别检查，不能把 BER 的结论自动推广到所有指标。']
    for key,label in [('A_with_F','共同反馈下增加 A'),('F_with_A','保留 A 后增加 F')]:
        evm=next(x for x in comparisons if x['id']==key and x['metric']=='rms_evm_percent')
        snr=next(x for x in comparisons if x['id']==key and x['metric']=='paired_output_snr_db')
        lines.append(f"- {label}：RMS EVM 差 {evm['estimate']:+.4f} 个百分点，95% 区间 [{evm['ci95'][0]:+.4f}, {evm['ci95'][1]:+.4f}]；输出 SNR 差 {snr['estimate']:+.4f} dB，95% 区间 [{snr['ci95'][0]:+.4f}, {snr['ci95'][1]:+.4f}]。")
    lines.append('RMS EVM 衡量均衡后符号与发送符号的偏差，配对输出 SNR 根据接收输出的信号/噪声功率计算，两者定义不同。若 SNR 差值区间包含零，则不据此主张汇总 SNR 获得改善。')
    lines+=['','## 反馈到底选择了什么','',
        f"在 {checked} 个环境-载频条件中，最终控制来自初始探测集合 {selection['initial']} 次、A 提出的候选 {selection['learned_candidate']} 次、补充码本 {selection['codebook_fallback']} 次。",
        '若候选与初始控制相同，按初始集合归类；导频并列时按原执行器的 argmax 顺序选择。',
        f"相对直接执行 A 的候选，A+F 的业务帧 BER 在 {payload['better']} 个条件降低、{payload['equal']} 个相同、{payload['worse']} 个升高。",
        '这些条件内计数用于解释选择过程，不当作独立环境的显著性检验。',
        'F 使用当前导频评分选择设置，不重新训练 A，也不读取待评价业务帧的正确答案。','',
        '这套拆分能否用于论文，取决于上述实际效应与方法贡献的表述。候选反馈是共同控制策略，',
        '有消融收益不自动等于独立的新算法创新；需要如实交代其与现有反馈选择方法的关系。',
        '这里复核的是既有测试集上的固定模型，不新增独立测试证据。','',
        '## 执行核查','',f'- {checked} 个环境-载频条件、{checked*5} 个方法输出均通过来源及一致性核查。',
        '- A+F 的控制与 RQ1 的 cnn_warm64 逐条件一致，所有五组的错误计数与既有记录完全一致。',
        '- 数值计算在远程主机完成；没有恢复 E2，没有修改论文或审稿回复。',
        '- 并发由 4 调整为 12，保留已提交条件。一次新并发入口缺少 main 保护，被执行锁阻止；',
        '  修复入口后通过子进程加载核对，使用同一冻结算法与数据续跑。失败日志和调整记录单独保留。','']
    text=(output/'REPORT_zh.md').read_text().split('\n## 如何理解本轮结果')[0]
    (output/'REPORT_zh.md').write_text(text+'\n'.join(lines))
    write_json(output/'selection_diagnostics.json',dict(conditions=checked,selected_origin=selection,
        payload_error_change_vs_A=payload,unit='environment-carrier condition; descriptive only'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,(left,right)=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1.15,1]})
    x=np.arange(5); y=np.array([r['ber']*100 for r in rows])
    ci=np.array([r['ci95']['ber'] for r in rows])*100
    colors=['#9C9C9C','#4477AA','#CCBB44','#228833','#AA3377']
    left.bar(x,y,color=colors,width=.64,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),capsize=3)
    left.set_xticks(x,['Traditional\n16 probes','A\n16 probes','Traditional+F\n64 probes','A+F\n64 probes','Fixed scan\n64 probes'],fontsize=9)
    for i,v in enumerate(y):left.text(i,ci[i,1]+.12,f'{v:.4f}',ha='center',fontsize=9)
    left.set(ylabel='BER (%)',ylim=(0,float(ci[:,1].max()+1.8)),title='(a) Fixed-model receive quality')
    left.grid(axis='y',alpha=.18);left.set_axisbelow(True)
    keys=['A_without_F','A_with_F','F_without_A','F_with_A','AF_vs_fixed_scan64']
    labels=['Add A; no F (16 probes)','Add A; with F (64 probes)',
            'Add F; no A (16 to 64)','Add F; with A (16 to 64)','A+F vs. fixed scan (64)']
    for i,key in enumerate(keys):
        r=ber[key];gain=-100*r['estimate'];low=-100*r['ci95'][1];high=-100*r['ci95'][0]
        right.errorbar(gain,i,xerr=[[gain-low],[high-gain]],fmt='o',color='#4477AA',capsize=4)
    right.set_yticks(range(len(keys)),labels,fontsize=9);right.invert_yaxis()
    right.axvline(0,color='black',linewidth=.7,linestyle='--');right.grid(axis='x',alpha=.18)
    right.set(xlabel='BER reduction (percentage points)',title='(b) Paired effects and 95% intervals')
    for ax in [left,right]:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A: complex-response learning; F: candidate feedback selection',fontsize=12)
    fig.text(.5,.015,'864 existing test environments; 17 carriers per environment; frozen seed-0 model. Intervals: paired environment bootstrap.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(output/'rq2_af_ablation.pdf');fig.savefig(output/'rq2_af_ablation.png',dpi=170);plt.close(fig)
    files=['REPORT_zh.md','summary.json','comparisons.json','environment_metrics.npz',
           'rq2_af_ablation.pdf','rq2_af_ablation.png','complete.json','protocol.json','records.json','selection_diagnostics.json']
    write_json(output/'verification.json',dict(status='passed',at=now(),conditions=checked,
        method_outputs=checked*5,report_code_sha256=sha256(Path(__file__)),
        files_sha256={n:sha256(output/n) for n in files},manuscript_changed=False,
        RQ1_primary_reproduced=True,old_joint_correction_used=False))
    print(json.dumps({'status':'passed','conditions':checked,'report':str(output/'REPORT_zh.md')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
