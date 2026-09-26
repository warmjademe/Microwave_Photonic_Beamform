"""全量基线结果汇总；先校验协议/覆盖/SHA，再按独立环境统计不确定性。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compact_dataset import sha256,CompactDataset
from generate_native_dataset import json_bytes,write_json,now
from run_compact_baselines import METRICS

NAMES=dict(ttd_das='粗方向TTD/DAS',codebook='细方向码本',coordinate='坐标搜索',
    spsa='SPSA',done='DONE适配',de='差分进化',mlp='MLP',teacher='离线教师参考',mrc='理想数字MRC参考')
ORDER=['ttd_das','codebook','coordinate','spsa','done','de','mlp','teacher','mrc']


def verify_training(root,study,learned_protocol):
    """汇总前再次核对实际模型的训练来源，拒绝把其它模型混入本轮结果。"""
    parent=Path(learned_protocol['test_dataset']).parent
    train=CompactDataset(parent/'dataset_train',verify_hashes=True)
    test=CompactDataset(parent/'dataset_test',verify_hashes=True)
    if train.split!='train' or test.split!='test':raise ValueError('训练/测试身份错误。')
    for split,data in [('train',train),('test',test)]:
        if sha256(data.root/'manifest.json')!=study[split+'_manifest_sha256']:
            raise ValueError('预定数据清单发生变化：'+split)
        if len(data)!=study[split+'_samples'] or len(data.environments)!=study[split+'_environments']:
            raise ValueError('数据规模与预定协议不同。')
    train_ids={row['environment_id'] for row in train.environments}
    test_ids={row['environment_id'] for row in test.environments}
    if train_ids&test_ids:raise ValueError('训练与测试环境重叠。')
    evidence=[]
    for record in learned_protocol['checkpoints']:
        path=Path(record['path']);document=json.loads((path/'checkpoint.json').read_text())
        metadata=document['metadata'];settings=document['settings']
        if sha256(path/'checkpoint.json')!=record['metadata_sha256'] or sha256(path/'weights.npz')!=record['weights_sha256']:
            raise ValueError('已评价模型发生变化。')
        if metadata['training_compact_manifest_sha256']!=study['train_manifest_sha256']:
            raise ValueError('模型并非使用预定训练数据。')
        if set(metadata['training_environment_ids'])!=train_ids:
            raise ValueError('模型训练环境与数据集不一致。')
        for key in ('epochs','batch_size','hidden','learning_rate'):
            if settings[key]!=study['mlp'][key]:raise ValueError('模型超参数不同：'+key)
        if settings['seed']!=record['seed']:raise ValueError('模型种子错误。')
        history=json.loads((path/'training_loss.json').read_text())
        if history[-1]['epoch']!=study['mlp']['epochs']:raise ValueError('模型未完成固定轮次。')
        if not all(np.isfinite(item['mse']) for item in history):raise ValueError('训练损失非有限。')
        evidence.append(dict(seed=record['seed'],training_manifest_sha256=metadata['training_compact_manifest_sha256'],
            training_environments=len(train_ids),training_samples=len(train),completed_epochs=history[-1]['epoch'],
            initial_training_mse=history[0]['mse'],final_training_mse=history[-1]['mse'],
            checkpoint_sha256=record['metadata_sha256'],weights_sha256=record['weights_sha256']))
    if [row['seed'] for row in evidence]!=study['seeds']:raise ValueError('训练种子未齐全。')
    return dict(environment_overlap=0,models=evidence)


def read_phase(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    progress=json.loads((root/'progress.json').read_text())
    if progress['status']!='complete' or not progress.get('final_sha_verification'):
        raise ValueError('基线阶段未完成：'+str(root))
    fingerprint=hashlib.sha256(json_bytes(p)).hexdigest();n=p['environment_count']
    if p['metric_order']!=METRICS:raise ValueError('指标顺序与评测器不同。')
    arrays=np.empty((len(p['methods']),len(p['seeds']),n,17,len(METRICS)),np.float64)
    identities=[]
    for ei in range(n):
        path=root/'records'/('environment_%05d.npz'%ei);marker=json.loads(path.with_suffix('.json').read_text())
        if marker['protocol_fingerprint']!=fingerprint or sha256(path)!=marker['sha256']:
            raise ValueError('结果记录SHA/协议不符。')
        with np.load(path,allow_pickle=False) as f:
            a=f['metrics'];control=f['control_code']
            if a.shape!=(len(p['methods']),len(p['seeds']),17,len(METRICS)):
                raise ValueError('结果形状不同。')
            if control.shape!=(len(p['methods']),len(p['seeds']),17,128):raise ValueError('控制形状错误。')
            if str(f['protocol_fingerprint'].item())!=fingerprint:raise ValueError('记录内部协议不同。')
            identities.append(str(f['environment_id'].item()))
            for mi,method in enumerate(p['methods']):
                if method!='mrc':
                    levels=np.r_[np.full(64,76),np.full(64,24)]
                    if np.any(control[mi]<0) or np.any(control[mi]>levels):raise ValueError('控制档位非法。')
                elif not np.all(control[mi]==-1):raise ValueError('MRC应显式无光子控制标签。')
            arrays[:,:,ei]=a
    if len(set(identities))!=n:raise ValueError('测试环境重复。')
    if not np.all(np.isfinite(arrays[...,:5])):raise ValueError('接收指标含非有限数。')
    return p,arrays,identities


def summarize(campaign):
    root=Path(campaign);study=json.loads((root/'study_protocol.json').read_text())
    pa,a,ids=read_phase(root/'nonlearning');pb,b,ids2=read_phase(root/'learned')
    for key in ('test_manifest_sha256','seeds','budget','carriers_ghz','environment_count','sample_count',
                'generation_fingerprint','selection_fingerprint','verified_native_core'):
        if pa[key]!=pb[key]:raise ValueError('两阶段比较条件不同：'+key)
    if ids!=ids2:raise ValueError('两阶段环境顺序不同。')
    training_audit=verify_training(root,study,pb)
    if set(pa['methods'])&set(pb['methods']):raise ValueError('方法重复。')
    values={method:a[i] for i,method in enumerate(pa['methods'])}
    values.update({method:b[i] for i,method in enumerate(pb['methods'])})
    if set(values)!=set(ORDER):raise ValueError('九基线未齐全。')
    for key in ('test_manifest_sha256','seeds','budget','environment_count','sample_count'):
        field={'environment_count':'test_environments','sample_count':'test_samples'}.get(key,key)
        if pa[key]!=study[field]:raise ValueError('实际评测与预定研究协议不同：'+key)
    groups={**pa['groups'],**pb['groups']};rng=np.random.default_rng(study['statistics']['bootstrap_seed'])
    bootstrap_indices=rng.integers(0,len(ids),(study['statistics']['bootstrap_replicates'],len(ids)))
    statistics={};environment_nmse={};by_frequency={}
    for method in ORDER:
        data=values[method];out=dict(method=method,name=NAMES[method],group=groups[method],
            test_samples_per_seed=pa['sample_count'],seeds=pa['seeds'])
        for key in METRICS:
            arr=data[...,METRICS.index(key)]
            if np.all(np.isnan(arr)):
                out['mean_'+key]=None;out['std_'+key+'_across_seeds']=None;continue
            if np.any(~np.isfinite(arr)):raise ValueError('指标存在部分非有限值：'+method+':'+key)
            mean_by_seed=arr.mean(axis=(1,2))
            out['mean_'+key]=float(mean_by_seed.mean())
            out['std_'+key+'_across_seeds']=float(mean_by_seed.std(ddof=1)) if len(mean_by_seed)>1 else None
        errors=data[...,METRICS.index('bit_errors')];bits=data[...,METRICS.index('bits_tested')]
        out.update(pooled_bit_errors=int(errors.sum()),pooled_bits_tested=int(bits.sum()),
            pooled_ber=float(errors.sum()/bits.sum()),ber_by_seed=(errors.sum(axis=(1,2))/bits.sum(axis=(1,2))).tolist())
        environment_nmse[method]=data[...,METRICS.index('payload_nmse')].mean(axis=(0,2))
        bootstrap_means=environment_nmse[method][bootstrap_indices].mean(axis=1)
        out['payload_nmse_environment_bootstrap_ci95']=np.quantile(bootstrap_means,[.025,.975]).tolist()
        statistics[method]=out
        by_frequency[method]=[]
        for ci,carrier in enumerate(pa['carriers_ghz']):
            sub=data[:,:,ci,:]
            by_frequency[method].append(dict(carrier_ghz=carrier,
                mean_evm_percent=float(sub[...,METRICS.index('evm_percent')].mean()),
                mean_payload_nmse=float(sub[...,METRICS.index('payload_nmse')].mean()),
                pooled_ber=float(sub[...,METRICS.index('bit_errors')].sum()/sub[...,METRICS.index('bits_tested')].sum()),
                mean_effective_snr_db=float(sub[...,METRICS.index('effective_snr_db')].mean())))
    paired={}
    for method in ORDER[:6]:
        difference=environment_nmse['mlp']-environment_nmse[method]
        distribution=difference[bootstrap_indices].mean(axis=1)
        paired[method]=dict(mean_nmse_difference_mlp_minus_baseline=float(difference.mean()),
            environment_bootstrap_ci95=np.quantile(distribution,[.025,.975]).tolist(),
            lower_is_better_for_mlp=True,independent_units=len(ids))
    report=dict(schema='mwp-baseline-summary-v1',status='complete',completed_at_utc=now(),
        study_protocol_sha256=sha256(root/'study_protocol.json'),test_samples=pa['sample_count'],
        independent_test_environments=pa['environment_count'],seeds=pa['seeds'],
        training_audit=training_audit,
        total_method_case_evaluations=pa['sample_count']*len(pa['seeds'])*len(ORDER),
        summary=statistics,by_frequency=by_frequency,paired_environment_analysis=paired,
        definitions=dict(effective_snr='-10log10 per-record payload NMSE; reported mean across records',
            wall_time='measured in concurrent CPU evaluation processes; includes scheduling effects',
            confidence_intervals='descriptive paired environment-cluster bootstrap after averaging carriers and seeds',
            teacher='stored offline label, not globally optimal; no claimed online runtime',
            mrc='different ideal digital RF hardware, privileged channel, independent digital payload',
            random_repeats='new model/optimizer seeds and additional APD noise; original probes/antenna noise frozen'),
        phase_sha256=dict(nonlearning=sha256(root/'nonlearning/protocol.json'),learned=sha256(root/'learned/protocol.json')))
    write_json(root/'summary.json',report)
    lines=['# 统一数值数据与基线实验结果','',
        f"训练集{study.get('train_samples',0):,}条（{study.get('train_environments',0):,}个环境），测试集{pa['sample_count']:,}条（{len(ids):,}个环境），均覆盖4–20 GHz的17个载频。没有验证集。",
        f"MLP仅用dataset_train训练，固定40轮、batch 64、128个隐层单元、学习率0.001，种子{pa['seeds']}；未用测试成绩选择模型。",
        f"全部九个基线在完整测试集上各运行{len(pa['seeds'])}次，共{report['total_method_case_evaluations']:,}次方法×样本评价。原始观测与训练标签在所有学习方法间统一；当前已有离线监督学习基线为MLP。单次运行不报告跨种子标准差。",'',
        '## 可实现的控制方法','',
        '| 方法 | EVM / % | 汇总BER | 平均等效SNR / dB | 实际探测次数 | 控制计算时间 / ms |',
        '|---|---:|---:|---:|---:|---:|']
    for method in ORDER[:7]:
        s=statistics[method]
        evm=f"{s['mean_evm_percent']:.3f}"
        if s['std_evm_percent_across_seeds'] is not None:evm+=f" ± {s['std_evm_percent_across_seeds']:.3f}"
        lines.append(f"| {s['name']} | {evm} | {s['pooled_ber']:.6f} | {s['mean_effective_snr_db']:.3f} | {s['mean_feedback_calls']:.1f} | {s['mean_controller_wall_seconds']*1000:.3f} |")
    lines.extend(['',
        '各方法起点为同一16套探测输入。MLP和粗方向TTD/DAS直接输出；其余自适应方法允许最多48次追加反馈，总预算64次，表中列出实际使用次数。结果因此同时体现接收质量与反馈成本，不能把追加反馈解释成完全相同的静态输入。',
        '等效SNR由接收误差计算，包含器件失真与估计误差。控制时间来自并发CPU运行，并非真实器件控制延迟实测。','',
        '## 额外信息参考','',
        '| 参考 | EVM / % | 汇总BER | 条件 |','|---|---:|---:|---|'])
    for method,condition in [('teacher','使用内部逐路缓存生成的离线控制标签'),('mrc','使用真实信道与64路理想数字接收；硬件和噪声模型不同')]:
        s=statistics[method];lines.append(f"| {s['name']} | {s['mean_evm_percent']:.3f} | {s['pooled_ber']:.6f} | {condition} |")
    lines.extend(['','## MLP与传统控制的配对差值','',
        '下表为MLP减去基线的平均payload NMSE，负值表示MLP的接收误差更小。以独立环境为单位重采样1,000次，每次保留环境内全部17载频，给出描述性95%区间。该区间描述环境差异，不是重复训练的不确定性。',
        '', '| 对照 | NMSE差值 | 95%区间 |','|---|---:|---:|'])
    for method,s in paired.items():
        lo,hi=s['environment_bootstrap_ci95'];lines.append(f"| {NAMES[method]} | {s['mean_nmse_difference_mlp_minus_baseline']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines.extend(['','## 复核文件','',
        '- `study_protocol.json`：训练和评测前固定的完整方案。',
        '- `mlp_seed*`：本次指定种子的最终模型、训练损失、训练环境与来源哈希。',
        '- `nonlearning/records`、`learned/records`：逐环境、逐载频、逐种子的控制码与接收指标。',
        '- `summary.json`：完整数值结果、17载频分项、误码计数、配对区间及定义。',
        '- `logs/`：完整运行日志；失败记录不覆盖、不删样本。','',
        '接收质量评价始终在算法返回控制后进行。公开X不含真实信道、内部光场或payload答案；教师与MRC分别作为额外信息参考。该实验沿用用户接受的器件近似，不改变历史OSD校准记录。',''])
    (root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='complete',methods=len(ORDER),test_samples=pa['sample_count'],
        method_case_evaluations=report['total_method_case_evaluations'])),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    summarize(p.parse_args().campaign)
