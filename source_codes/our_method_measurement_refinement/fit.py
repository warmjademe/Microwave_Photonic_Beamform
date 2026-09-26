"""华硕GPU仅推理原训练成员，拟合固定CNN的残差矩阵；不重新训练权重。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from our_method_response_control.physics import ridge_estimate
from our_method_response_control.model import Model,conditions
from our_method_response_control.train import precision,verify_targets


def run(project,output):
    require_host();precision()
    if not torch.cuda.is_available():raise RuntimeError('冻结模型推理须用华硕GPU。')
    if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('GPU空闲显存不足4GiB。')
    b=project/'dataset_simulation';data=b/'outputs/quality_rank_hybrid_20260925'
    targets=b/'diagnostics/20260925_response_control_targets';checkpoint=b/'baseline_results/20260925_response_control'
    verify_targets(data,targets)
    meta=json.loads((checkpoint/'complete.json').read_text())
    if sha256(checkpoint/'weights.pt')!=meta['weights_sha256']:raise ValueError('固定权重发生改变。')
    raw,_,_,rows=load_split(data,'train',labels=False)
    if len(rows)!=864 or len(raw)!=14688:raise ValueError('本诊断只拟合原864训练成员。')
    public=public_data(data)
    sources=source_record(['our_method_measurement_refinement/'+n for n in ['fit.py','method.py','PROTOCOL.md']]+[
        'our_method_response_control/model.py','our_method_response_control/physics.py','our_method_response_control/train.py'])
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(test_used=False,scope='training residual regularizer fit; frozen network',
        train_environment_ids=[r['environment_id'] for r in rows],train_samples=len(raw),
        weights_sha256=meta['weights_sha256'],data_manifest_sha256=sha256(data/'manifest.json'),
        targets_complete_sha256=sha256(targets/'complete.json'),source_sha256=sources,
        inference_batch=64,network_updates=0,seed=0)
    write_json(output/'protocol.json',protocol)
    for name in sources:
        dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dst)
    h0=ridge_estimate(raw,public['pilot_qpsk']).astype(np.complex64)
    cond=conditions(raw,h0,public['pilot_qpsk'])
    target=np.load(targets/'train_response.npy',mmap_mode='r')
    model=Model().cuda();model.load_state_dict(torch.load(checkpoint/'weights.pt',map_location='cuda',weights_only=True));model.eval()
    cov=np.zeros((17,64,64),complex);counts=np.zeros(17,int)
    error_sum=np.zeros(17);energy_sum=np.zeros(17)
    for begin in range(0,len(raw),64):
        end=min(begin+64,len(raw))
        with torch.inference_mode():pred=model(torch.from_numpy(h0[begin:end]).cuda(),torch.from_numpy(cond[begin:end]).cuda()).cpu().numpy()
        err=np.asarray(target[begin:end],complex)-pred
        scale=np.maximum(np.mean(abs(pred.astype(complex))**2,axis=(1,2)),1e-24)
        normalized=err/np.sqrt(scale)[:,None,None]
        for fc in np.unique(raw[begin:end,1984].astype(int)):
            mask=raw[begin:end,1984]==fc;e=normalized[mask]
            cov[fc-4]+=np.sum(e @ e.conj().transpose(0,2,1),axis=0)
            counts[fc-4]+=int(mask.sum())*31
            error_sum[fc-4]+=np.sum(abs(err[mask])**2)
            energy_sum[fc-4]+=np.sum(abs(target[begin:end][mask])**2)
        if begin%1024==0:write_json(output/'progress.json',dict(status='fitting',completed=end,total=len(raw),at=now()))
    cov/=counts[:,None,None];cov=(cov+cov.conj().transpose(0,2,1))/2
    if np.any(counts!=864*31) or np.linalg.eigvalsh(cov).min()<-1e-10*np.max(abs(cov)):
        raise ValueError('训练样本数或残差矩阵正半定检查失败。')
    np.save(output/'residual_covariance.npy',cov,allow_pickle=False)
    verify_sources(sources);verify_targets(data,targets)
    if sha256(checkpoint/'weights.pt')!=meta['weights_sha256']:raise ValueError('拟合期间权重改变。')
    write_json(output/'complete.json',dict(status='complete',at=now(),train_environments=864,
        covariance_sha256=sha256(output/'residual_covariance.npy'),weights_sha256=meta['weights_sha256'],
        residual_energy_ratio_by_carrier=(error_sum/energy_sum).tolist(),network_updates=0))
    write_json(output/'progress.json',dict(status='complete',completed=len(raw),total=len(raw),at=now()))
    print(json.dumps(dict(status='complete',train_samples=len(raw),network_updates=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.project.resolve(),a.output.resolve())
