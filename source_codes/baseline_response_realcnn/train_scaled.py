"""共同训练成员的可变规模实数响应CNN；原训练算法及超参数保持不变。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.scale_large import freeze,fingerprint
from our_method_response_control.physics import ridge_estimate
from our_method_response_control.model import conditions
from our_method_response_control.train import precision,verify_targets
from baseline_response_realcnn.model import Model
from run_deep_baselines import atomic_torch


from study_full_baselines.fair_training_common import cohort_metadata


def run(data,targets,output):
    require_host();precision()
    if not torch.cuda.is_available():raise RuntimeError('实数CNN仅在华硕GPU训练。')
    manifest=check_data(data); cohort=cohort_metadata(data); verify_targets(data,targets)
    raw,_,_,rows=load_split(data,'train',labels=False)
    if len(rows)!=cohort['train_environments'] or len(raw)!=cohort['train_samples']:
        raise ValueError('实际训练数组与共同成员不同。')
    protocol=dict(cohort=cohort, epochs=40,seed=0,batch_size=64,learning_rate=.001,gradient_clip=5.,
        loss='mean per-sample relative complex response MSE',checkpoint='last epoch',validation=False,
        parameter_count=17524,complex_reference_parameters=17540,
        train_ids=[r['environment_id'] for r in rows],data_manifest_sha256=sha256(data/'manifest.json'),
        target_sha256=sha256(targets/'complete.json'),source_sha256=source_record([
            'baseline_response_realcnn/model.py','baseline_response_realcnn/train_scaled.py','baseline_response_realcnn/train.py','baseline_response_realcnn/README.md',
            'study_full_baselines/fair_training_common.py','study_full_baselines/SCALED_FAIR_TRAINING_PROTOCOL.md',
            'our_method_response_control/train.py',
            'our_method_response_control/physics.py','our_method_response_control/model.py',
            'study_full_baselines/common.py','study_full_baselines/scale_large.py']))
    freeze(output,protocol);fp=fingerprint(protocol)
    if (output/'complete.json').exists():raise FileExistsError('已完成模型不重复训练。')
    public=public_data(data);base=ridge_estimate(raw,public['pilot_qpsk']).astype(np.complex64)
    cond=torch.from_numpy(conditions(raw,base,public['pilot_qpsk'])).cuda()
    initial=torch.from_numpy(base).cuda();target=torch.from_numpy(np.load(targets/'train_response.npy')).cuda()
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    model=Model().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    if sum(p.numel() for p in model.parameters())!=17524:raise ValueError('参数量与协议不符。')
    shuffle=torch.Generator(device='cuda').manual_seed(0)
    history=[];elapsed=0.
    if (output/'resume.pt').exists():
        old=torch.load(output/'resume.pt',map_location='cuda',weights_only=False)
        if old['fingerprint']!=fp:raise ValueError('断点协议不同。')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        shuffle.set_state(old['shuffle'].cpu());history=old['history'];elapsed=old['seconds']
    started=time.perf_counter();seconds=elapsed
    for epoch in range(len(history)+1,41):
        model.train();total=0.
        for ix in torch.randperm(len(raw),device='cuda',generator=shuffle).split(64):
            prediction=model(initial[ix],cond[ix])
            loss=((prediction-target[ix]).abs().square().mean((-1,-2))
                /target[ix].abs().square().mean((-1,-2)).clamp_min(1e-24)).mean()
            if not torch.isfinite(loss):raise ValueError('实数响应训练损失非有限。')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(ix)
        history.append(dict(epoch=epoch,loss=total/len(raw)));seconds=elapsed+time.perf_counter()-started
        atomic_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),shuffle=shuffle.get_state(),
            history=history,seconds=seconds,fingerprint=fp),output/'resume.pt')
        write_json(output/'history.json',history)
        state=dict(status='training',epoch=epoch,loss=history[-1]['loss'],pid=os.getpid(),at=now())
        write_json(output/'progress.json',state)
        if epoch%10==0:print(json.dumps(state),flush=True)
    atomic_torch(model.state_dict(),output/'weights.pt');model.eval()
    # 训练结束后才加载旧测试公开X，不读取测试响应真值。
    x,_,_,test_rows=load_split(data,'test',labels=False)
    base=ridge_estimate(x,public['pilot_qpsk']).astype(np.complex64)
    test_cond=torch.from_numpy(conditions(x,base,public['pilot_qpsk'])).cuda()
    test_initial=torch.from_numpy(base).cuda()
    with torch.inference_mode():
        predictions=np.concatenate([model(test_initial[i:i+128],test_cond[i:i+128]).cpu().numpy()
            for i in range(0,len(x),128)])
        np.save(output/'predicted_response.npy',predictions,allow_pickle=False)
        replay=Model().cuda();replay.load_state_dict(torch.load(output/'weights.pt',map_location='cuda',weights_only=True));replay.eval()
        for i in range(0,len(x),128):
            if not np.array_equal(replay(test_initial[i:i+128],test_cond[i:i+128]).cpu().numpy(),predictions[i:i+128]):
                raise ValueError('实数响应CNN回放不一致。')
    verify_targets(data,targets);verify_sources(protocol['source_sha256'])
    write_json(output/'complete.json',dict(status='complete',fingerprint=fp,parameters=17524,
        epochs=40,seed=0,train_environments=cohort['train_environments'],train_samples=len(raw),training_seconds=seconds,weights_sha256=sha256(output/'weights.pt'),
        prediction_sha256=sha256(output/'predicted_response.npy'),full_prediction_replay=True,at=now()))
    write_json(output/'progress.json',dict(status='complete',epochs=40,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','targets','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:run(a.data,a.targets,a.output)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()));raise
