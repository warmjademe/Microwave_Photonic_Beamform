"""从实际优化器状态核对训练规模、更新次数与覆盖，并对应完整接收质量。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE, require_host, sha256, verify_sources, source_record, write_json, now
from study_full_baselines.analyze_reception import write_csv

COUNTS=[216,432,864,1728,3456]
METRICS=[('ber',100,'BER（%）↓'),('ser',100,'SER（%）↓'),
         ('block_error_rate',100,'块错误率（%）↓'),('rms_evm_percent',1,'RMS EVM（%）↓'),
         ('paired_output_snr_db',1,'配对输出 SNR（dB）↑')]


def read(path):return json.loads(path.read_text())


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def check(path,digest):
    if sha256(path)!=digest:raise ValueError('训练规模来源改变：'+str(path))


def check_optimizer(resume,weights,expected_steps):
    if set(resume['model'])!=set(weights):raise ValueError('断点和最终权重字段不同。')
    for key,value in weights.items():
        if not torch.equal(value,resume['model'][key]):raise ValueError('优化器断点不属于最终权重。')
    state=resume['optimizer']['state']
    parameters=[v for g in resume['optimizer']['param_groups'] for v in g['params']]
    if set(state)!=set(parameters) or len(state)!=len(weights):
        raise ValueError('并非每个模型参数都具有优化器记录。')
    steps=[]
    for value in state.values():
        step=float(value['step'])
        if not math.isfinite(step) or step!=expected_steps:
            raise ValueError('实际优化器更新次数不符。')
        steps.append(int(step))
    return len(steps)


def inspect_model(project,count,schedule):
    base=project/'dataset_simulation';study=base/'baseline_results/20260925_full_baselines'
    if count==864:
        name='complex_response_cnn';folder=base/'baseline_results/20260925_response_control'
    else:
        name='response_n%04d_%s'%(count,schedule)
        folder=study/('scale_small' if count<864 else 'scale_%d'%count)/name
    meta,protocol,history=[read(folder/n) for n in ['complete.json','protocol.json','history.json']]
    if meta['status']!='complete' or meta['epochs']!=40 or meta['seed']!=0:
        raise ValueError('规模模型尚未完整训练。')
    if (protocol['epochs']!=40 or protocol['seed']!=0 or protocol['batch_size']!=64
            or protocol['train_samples']!=count*17 or protocol['validation']):
        raise ValueError('规模训练条件不同。')
    verify_sources(protocol['source_sha256'])
    check(folder/'weights.pt',meta['weights_sha256'])
    if 'history_sha256' in meta:check(folder/'history.json',meta['history_sha256'])
    if [r['epoch'] for r in history]!=list(range(1,41)) or not all(math.isfinite(r['loss']) for r in history):
        raise ValueError('40段训练历史不完整或损失非法。')
    if 'prediction_sha256' in meta:check(folder/'predicted_response.npy',meta['prediction_sha256'])
    if count!=864 and meta['fingerprint']!=fingerprint(protocol):
        raise ValueError('模型完成记录与训练协议不同。')
    draws=count*17 if schedule=='fixed_epochs' else 14688
    updates=math.ceil(draws/64)*40
    if count!=864 and (protocol['sample_draws_per_epoch']!=draws or protocol['updates_per_epoch']!=math.ceil(draws/64)):
        raise ValueError('声明更新次数不同。')
    # 均为本项目保存的受信任训练断点；在CPU读取，不启动任何训练。
    resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
    weights=torch.load(folder/'weights.pt',map_location='cpu',weights_only=True)
    parameter_states=check_optimizer(resume,weights,updates)
    if sum(x.numel() for x in weights.values())!=meta['parameters'] or meta['parameters']!=17540:
        raise ValueError('规模曲线改变了网络参数量。')
    if count==864:
        if resume['epoch']!=40:raise ValueError('基准断点不是最后一轮。')
    elif resume['history']!=history or resume['fingerprint']!=meta['fingerprint']:
        raise ValueError('实际断点与训练历史不同。')
    data=base/('outputs/quality_rank_hybrid_20260925' if count<=864 else 'outputs/scaling_train_%d_20260925'%count)
    check(data/'manifest.json',protocol['data_manifest_sha256'])
    manifest=read(data/'manifest.json');all_ids=[r['environment_id'] for r in manifest['environments'] if r['split']=='train']
    ids=protocol['training_environment_ids'] if count<864 else all_ids
    if len(ids)!=count or len(set(ids))!=count or not set(ids)<=set(all_ids):
        raise ValueError('训练成员数量或身份不同。')
    if count>864:
        check(folder/'used_training_samples.npy',meta['used_samples_sha256'])
        used=np.load(folder/'used_training_samples.npy',allow_pickle=False)
        if (used.shape!=(count*17,) or used.dtype!=np.bool_ or not np.array_equal(used,resume['used'].numpy())
                or int(used.sum())!=meta['used_samples']
                or int(used.reshape(count,17).any(1).sum())!=meta['used_environments']):
            raise ValueError('实际训练覆盖记录不一致。')
        used_samples=int(used.sum());used_environments=int(used.reshape(count,17).any(1).sum())
        coverage='checkpoint bitmap'
    else:
        # 小规模每段至少一个完整无放回遍历，由冻结循环和完整断点支持。
        used_samples=count*17;used_environments=count;coverage='complete shuffled passes in verified source'
    sources={str(p.relative_to(project)):sha256(p) for p in [folder/n for n in
        ['protocol.json','complete.json','history.json','weights.pt','resume.pt']]}
    result=dict(method=name,training_environments=count,training_samples=count*17,schedule=schedule,
        macro_epochs=40,optimizer_updates=updates,optimizer_parameter_states_checked=parameter_states,
        sample_draws=draws*40,equivalent_data_passes=draws*40/(count*17),batch_size=64,seed=0,
        used_samples=used_samples,used_environments=used_environments,coverage_evidence=coverage,
        parameters=meta['parameters'],training_seconds_diagnostic=meta['training_seconds'],
        weights_sha256=meta['weights_sha256'],training_ids_sha256=fingerprint(ids),
        training_time_scope='different background loads; not controlled speed comparison')
    return result,ids,sources,resume,weights


def quality_rows(project,through):
    base=project/'dataset_simulation'
    folder=base/('diagnostics/20260926_scale_through_1728' if through else 'baseline_results/20260925_full_baselines/analysis')
    done=read(folder/'complete.json');field='files' if through else 'file_sha256'
    if done['status']!=('complete_through_1728' if through else 'complete'):
        raise ValueError('接收统计尚未完成。')
    for name,digest in done[field].items():check(folder/name,digest)
    verify_sources(done['source_sha256'])
    if through:
        rows=read(folder/'curves.json')['records'];paired=read(folder/'paired.json')['records']
    else:
        original={r['method']:r for r in read(folder/'overall.json')['records']};rows=[]
        for kind,schedule in [('cnn_fixed40','fixed_epochs'),('cnn_equal_updates','equal_updates'),('covariance',None)]:
            for count in COUNTS:
                name=('covariance_response' if count==864 else 'covariance_n%04d'%count) if schedule is None else ('complex_response_cnn' if count==864 else 'response_n%04d_%s'%(count,schedule))
                r=original[name]
                if r['training_environments']!=count or r['feedback_calls']!=16:
                    raise ValueError('规模曲线比较条件不同。')
                rows.append(dict(r,curve=kind,training_environments=count))
        paired=[r for r in read(folder/'comparisons.json')['records'] if r['family'].startswith('RQ3_')]
        plan=read(folder/'plan.json')
        expected={(r['id'],m) for r in plan['comparisons'] if r['family'].startswith('RQ3_') for m in plan['metrics']}
        if {(r['comparison'],r['metric']) for r in paired}!=expected:
            raise ValueError('完整RQ3比较清单缺失。')
    return rows,paired,{str(folder/'complete.json'):sha256(folder/'complete.json')}


def render(output,counts,budgets,quality,paired,full):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    assets=[]
    def save(fig,name):
        for ext in ['png','pdf','svg']:
            dest=output/(name+'.'+ext);fig.savefig(dest,dpi=300,bbox_inches='tight');assets.append(dest.name)
        plt.close(fig)
    styles=[('fixed_epochs','固定40次完整遍历','#0072B2','-'),('equal_updates','固定9,200次更新','#D55E00','--')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,field,title in zip(axes,['optimizer_updates','equivalent_data_passes'],['实际优化器更新次数','等价完整数据遍历次数']):
        for schedule,label,color,style in styles:
            rows=[r for r in budgets if r['schedule']==schedule]
            ax.plot(range(len(counts)),[r[field] for r in rows],style,marker='o',color=color,label=label)
        ax.set_xticks(range(len(counts)),[str(n) for n in counts]);ax.set_xlabel('独立训练环境数');ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('训练规模与计算预算：逐参数读取最终优化器状态核对',fontsize=14)
    fig.text(.5,.015,'864环境的两条曲线共用同一模型；固定seed0、batch64。更新次数因末批舍入而不严格成倍。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.93));save(fig,'training_scale_budget')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    curves=[('cnn_fixed40','CNN：固定40遍','#0072B2','-'),('cnn_equal_updates','CNN：相同更新次数','#D55E00','--'),('covariance','同规模传统协方差','#777777',':')]
    for ax,(field,scale,title) in zip(axes.flat,METRICS):
        for kind,label,color,style in curves:
            rows=[r for r in quality if r['curve']==kind]
            y=np.asarray([r[field] for r in rows])*scale;ci=np.asarray([r[field+'_ci95'] for r in rows])*scale
            ax.plot(range(len(counts)),y,style,marker='o',color=color,label=label)
            ax.fill_between(range(len(counts)),ci[:,0],ci[:,1],alpha=.08,color=color)
        ax.set_xticks(range(len(counts)),[str(n) for n in counts]);ax.set_xlabel('独立训练环境数');ax.set_title(title,fontsize=11);ax.grid(alpha=.18)
    axes.flat[5].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles,labels,loc='upper left',frameon=False)
    axes.flat[5].text(.03,.58,'同一批旧216环境 × 17载频\n全部使用16次公开测量\n阴影：95%环境区间\n单次训练seed0\n'+('全部五档规模已完成' if full else '3456环境尚未完成'),transform=axes.flat[5].transAxes,fontsize=11,va='top',linespacing=1.7)
    fig.suptitle('数据规模的五项接收质量：分别观察更多环境与更多更新',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'scale_complete_curve' if full else 'training_scale_quality')
    if full:
        selected=[('scale_%s_%d_minus_%d'%(s,a,b),('%s：%d−%d'%(label,a,b))) for s,label,_,_ in styles for a,b in zip(counts[1:],counts[:-1])]
        fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
        for ax,(field,scale,title) in zip(axes,[METRICS[0],METRICS[3],METRICS[4]]):
            for i,(name,label) in enumerate(selected):
                r=next(x for x in paired if x['comparison']==name and x['metric']==field)
                lo,hi=np.asarray(r['ci95'])*scale;color=styles[i//(len(counts)-1)][2]
                ax.plot([lo,hi],[i,i],color=color,lw=2);ax.scatter(r['estimate']*scale,i,color=color,s=35)
            ax.axvline(0,ls='--',color='#444',lw=1);ax.set_yticks(range(len(selected)),[v[1] for v in selected]);ax.set_xlabel(title.replace('（%）','差值（百分点）').replace('（dB）','差值（dB）'));ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis();fig.suptitle('逐档增加训练环境后的配对差异',fontsize=14)
        fig.text(.5,.015,'同一旧216环境，10,000次环境块bootstrap；全部预设RQ3族的校正结果见paired.csv。',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.055,1,.93));save(fig,'scale_complete_paired')
    return assets


def run(project,output,through):
    require_host()
    if output.exists():raise FileExistsError('训练规模核查另建目录，不覆盖既有产物。')
    quality,paired,quality_source=quality_rows(project,through)
    counts=COUNTS[:-1] if through else COUNTS
    budgets=[];members={};files={};unique={};states=0;negative_checked=False
    for count in counts:
        for schedule in ['fixed_epochs','equal_updates']:
            if count==864 and schedule=='equal_updates':
                budgets.append(dict(budgets[-1],schedule=schedule));continue
            row,ids,hashes,resume,weights=inspect_model(project,count,schedule)
            if count in members and members[count]!=ids:raise ValueError('同规模两种日程使用不同成员。')
            members[count]=ids;budgets.append(row);files.update(hashes);unique[row['method']]=row['weights_sha256'];states+=row['optimizer_parameter_states_checked']
            if not negative_checked:
                try:check_optimizer(resume,weights,row['optimizer_updates']+1)
                except ValueError:negative_checked=True
                else:raise AssertionError('错误更新次数未被拒绝。')
            del resume,weights
    for left,right in zip(counts[:-1],counts[1:]):
        if not set(members[left])<set(members[right]):raise ValueError('训练规模不是严格嵌套。')
    if {r['optimizer_updates'] for r in budgets if r['schedule']=='equal_updates'}!={9200}:
        raise ValueError('等更新次数对照不相等。')
    if len(quality)!=len(counts)*3:raise ValueError('接收曲线覆盖不完整。')
    output.mkdir(parents=True)
    source_names=['study_full_baselines/report_training_scale.py','study_full_baselines/SCALE_LARGE_PROTOCOL.md']
    sources=source_record(source_names)
    for name in sources:
        dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
    write_json(output/'training_budget.json',dict(records=budgets,counts=counts,unique_models=len(unique),
        equivalent_864_weights_shared=True,source_file_sha256=files,quality_source_sha256=quality_source,
        source_sha256=sources,formal_online_timing=False,full_scale_study_complete=not through))
    write_csv(output/'training_budget.csv',budgets)
    write_json(output/'curves.json',dict(records=quality,counts=counts,scope='old216 exploration'))
    write_csv(output/'curves.csv',quality);write_json(output/'paired.json',dict(records=paired));write_csv(output/'paired.csv',paired)
    figures=render(output,counts,budgets,quality,paired,not through)
    report=dict(status='passed_training_scale_optimizer_audit',at=now(),counts=counts,unique_models=len(unique),
        optimizer_parameter_states_checked=states,equal_updates_exactly_9200=True,
        weights_equal_final_optimizer_model=True,strictly_nested_training_members=True,
        wrong_optimizer_steps_rejected=negative_checked,full_scale_study_complete=not through)
    write_json(output/'audit.json',report)
    output_files=['training_budget.json','training_budget.csv','curves.json','curves.csv','paired.json','paired.csv','audit.json']+figures
    verify_sources(sources)
    write_json(output/'complete.json',dict(status='complete_training_scale_audit',at=now(),
        source_sha256=sources,files={n:sha256(output/n) for n in output_files},
        full_scale_study_complete=not through,visual_review='pending'))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--through-1728',action='store_true');args=parser.parse_args()
    run(args.project.resolve(),args.output.resolve(),args.through_1728)
