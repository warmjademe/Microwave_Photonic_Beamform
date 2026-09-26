"""固定响应CNN的嵌套训练规模对照，以及匹配优化步数的辅助对照。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from our_method_response_control.physics import ridge_estimate
from our_method_response_control.model import Model,conditions
from our_method_response_control.train import precision,verify_targets
from run_deep_baselines import atomic_torch


def selection(rows,count):
    if count not in [216,432,864]:raise ValueError('当前入口只研究原864环境的嵌套子集。')
    selected=[r for r in rows if r['stratum_repeat_index']<count//216]
    groups={}
    for r in selected:groups[r['joint_stratum_id']]=groups.get(r['joint_stratum_id'],0)+1
    if len(selected)!=count or len(groups)!=216 or set(groups.values())!={count//216}:
        raise ValueError('训练子集未保持联合分层均衡。')
    indices=np.concatenate([np.arange(r['index']*17,(r['index']+1)*17) for r in selected])
    return selected,indices


def run(data,targets,output,counts,schedules):
    require_host()
    if not torch.cuda.is_available():raise RuntimeError('仅华硕GPU。')
    check_data(data);verify_targets(data,targets);precision()
    raw,_,_,rows=load_split(data,'train',labels=False)
    public=public_data(data);pilots=public['pilot_qpsk']
    all_target=np.load(targets/'train_response.npy',mmap_mode='r')
    source_names=['study_full_baselines/scale_response.py','study_full_baselines/common.py',
        'our_method_response_control/model.py','our_method_response_control/physics.py',
        'our_method_response_control/train.py']
    source_hashes=source_record(source_names)
    output.mkdir(parents=True,exist_ok=True)
    for count in counts:
        subset,indices=selection(rows,count)
        # 协方差对照也只使用该规模的训练标签，不能让小数据网络独自少看数据。
        target_array=np.asarray(all_target[indices])
        grouped=target_array.reshape(count,17,64,31).astype(np.complex128)
        covariance=np.einsum('ecnk,ecmk->cnm',grouped,grouped.conj())/(count*31)
        np.save(output/('covariance_%04d.npy'%count),covariance,allow_pickle=False)
        del grouped,covariance
        selected_x=raw[indices]
        initial_array=ridge_estimate(selected_x,pilots).astype(np.complex64)
        cond_array=conditions(selected_x,initial_array,pilots)
        initial=torch.from_numpy(initial_array).cuda()
        cond=torch.from_numpy(cond_array).cuda();target=torch.from_numpy(target_array).cuda()
        for schedule in schedules:
            name='response_n%04d_%s'%(count,schedule)
            folder=output/name;folder.mkdir(exist_ok=True)
            draws_per_epoch=len(indices) if schedule=='fixed_epochs' else len(raw)
            protocol=dict(method='unchanged complex response CNN',train_environments=count,
                train_samples=len(indices),training_environment_ids=[r['environment_id'] for r in subset],
                seed=0,epochs=40,batch_size=64,learning_rate=.001,schedule=schedule,
                sample_draws_per_epoch=draws_per_epoch,updates_per_epoch=int(np.ceil(draws_per_epoch/64)),
                effective_unique_data_passes=40*draws_per_epoch/len(indices),
                equal_updates_definition='repeat complete shuffled small-data passes until14688 draws per epoch, then truncate; no new environments',
                loss='relative complex response MSE',checkpoint='last epoch40',validation=False,
                data_manifest_sha256=sha256(data/'manifest.json'),target_sha256=sha256(targets/'complete.json'),
                source_sha256=source_hashes,scope='exploratory size curve; no test-based checkpoint selection')
            fp=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
            if (folder/'protocol.json').exists():
                if json.loads((folder/'protocol.json').read_text())!=protocol:raise ValueError('规模协议改变。')
            else:
                write_json(folder/'protocol.json',protocol)
                for source in source_names:
                    dest=folder/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(SOURCE/source,dest)
            if (folder/'complete.json').exists():
                m=json.loads((folder/'complete.json').read_text())
                if m['fingerprint']!=fp or sha256(folder/'weights.pt')!=m['weights_sha256']:
                    raise ValueError('已完成规模模型发生变化。')
                continue
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            model=Model().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            shuffle=torch.Generator(device='cuda').manual_seed(0)
            history=[];elapsed=0.
            if (folder/'resume.pt').exists():
                old=torch.load(folder/'resume.pt',map_location='cuda',weights_only=False)
                if old['fingerprint']!=fp:raise ValueError('规模断点不同。')
                model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
                shuffle.set_state(old['shuffle'].cpu());history=old['history'];elapsed=old['seconds']
            started=time.perf_counter()
            for epoch in range(len(history)+1,41):
                model.train();parts=[];total_length=0
                while total_length<draws_per_epoch:
                    part=torch.randperm(len(target),device='cuda',generator=shuffle)
                    parts.append(part);total_length+=len(part)
                order=torch.cat(parts)[:draws_per_epoch];loss_sum=0.
                for ix in order.split(64):
                    predicted=model(initial[ix],cond[ix])
                    loss=((predicted-target[ix]).abs().square().mean((-1,-2))
                        /target[ix].abs().square().mean((-1,-2)).clamp_min(1e-24)).mean()
                    if not torch.isfinite(loss):raise ValueError('规模训练出现非有限损失。')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    loss_sum+=float(loss.detach())*len(ix)
                history.append(dict(epoch=epoch,loss=loss_sum/draws_per_epoch))
                seconds=elapsed+time.perf_counter()-started
                atomic_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    shuffle=shuffle.get_state(),history=history,seconds=seconds,fingerprint=fp),folder/'resume.pt')
                write_json(folder/'history.json',history)
                progress=dict(status='training',method=name,epoch=epoch,loss=history[-1]['loss'],at=now())
                write_json(output/'progress.json',progress)
                if epoch%10==0:print(json.dumps(progress),flush=True)
            atomic_torch(model.state_dict(),folder/'weights.pt');model.eval()
            # 只在权重固定后加载测试公开X，不读取任何测试监督标签。
            test_x,_,_,_=load_split(data,'test',labels=False)
            test_initial_np=ridge_estimate(test_x,pilots).astype(np.complex64)
            test_cond=torch.from_numpy(conditions(test_x,test_initial_np,pilots)).cuda()
            test_initial=torch.from_numpy(test_initial_np).cuda()
            with torch.inference_mode():
                predicted=np.concatenate([model(test_initial[i:i+128],test_cond[i:i+128]).cpu().numpy()
                    for i in range(0,len(test_x),128)])
                np.save(folder/'predicted_response.npy',predicted,allow_pickle=False)
                replay=Model().cuda()
                replay.load_state_dict(torch.load(folder/'weights.pt',map_location='cuda',weights_only=True));replay.eval()
                for i in range(0,len(test_x),128):
                    check=replay(test_initial[i:i+128],test_cond[i:i+128]).cpu().numpy()
                    if not np.array_equal(check,predicted[i:i+128]):raise ValueError('规模模型回放不同。')
            write_json(folder/'complete.json',dict(status='complete',fingerprint=fp,
                training_seconds=seconds,epochs=40,seed=0,parameters=sum(p.numel() for p in model.parameters()),
                weights_sha256=sha256(folder/'weights.pt'),prediction_sha256=sha256(folder/'predicted_response.npy'),
                full_prediction_replay=True,at=now()))
            del model,optimizer,replay,test_cond,test_initial;torch.cuda.empty_cache()
        del initial,cond,target
    verify_sources(source_hashes);verify_targets(data,targets)
    write_json(output/'progress.json',dict(status='complete',counts=counts,schedules=schedules,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','targets','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--counts',nargs='+',type=int,default=[216,432])
    p.add_argument('--schedules',nargs='+',choices=['fixed_epochs','equal_updates'],default=['fixed_epochs','equal_updates'])
    a=p.parse_args()
    try:run(a.data,a.targets,a.output,a.counts,a.schedules)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
