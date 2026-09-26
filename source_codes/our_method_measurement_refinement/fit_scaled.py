"""共同3,456训练成员：分别拟合CNN和传统估计的残差权重。"""
import argparse
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
from study_full_baselines.scale_large import fingerprint
from our_method_response_control.physics import ridge_estimate,covariance_estimate
from our_method_response_control.model import Model,conditions
from our_method_response_control.train import precision,verify_targets


def run(project,output):
    require_host();precision()
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*1024**3:
        raise RuntimeError('需要华硕GPU且至少4GiB空闲显存。')
    b=project/'dataset_simulation';data=b/'outputs/scaling_train_3456_20260925'
    scale=b/'baseline_results/20260925_full_baselines/scale_3456'
    targets=scale/'targets';checkpoint=scale/'response_n3456_fixed_epochs'
    verify_targets(data,targets)
    meta=json.loads((checkpoint/'complete.json').read_text());cp=json.loads((checkpoint/'protocol.json').read_text())
    if (cp['train_environments']!=3456 or cp['epochs']!=40 or cp['seed']!=0
        or cp['data_manifest_sha256']!=sha256(data/'manifest.json')
        or meta['fingerprint']!=fingerprint(cp) or sha256(checkpoint/'weights.pt')!=meta['weights_sha256']):
        raise ValueError('共同规模模型来源或训练协议不一致。')
    raw,_,_,rows=load_split(data,'train',labels=False)
    if len(rows)!=3456 or len(raw)!=58752:raise ValueError('本入口只允许共同3,456环境。')
    public=public_data(data)
    sources=source_record(['our_method_measurement_refinement/'+n for n in ['fit_scaled.py','method.py','SCALED_PROTOCOL.md']]+
        ['our_method_response_control/model.py','our_method_response_control/physics.py','our_method_response_control/train.py'])
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope='3456 common training residual fit',test_used=False,
        train_environment_ids=[r['environment_id'] for r in rows],train_samples=len(raw),train_environments=3456,
        estimators=['covariance','cnn'],weights_sha256=meta['weights_sha256'],
        data_manifest_sha256=sha256(data/'manifest.json'),targets_complete_sha256=sha256(targets/'complete.json'),
        source_sha256=sources,inference_batch=64,network_updates=0,seed=0)
    write_json(output/'protocol.json',protocol)
    for name in sources:
        dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dst)
    initial=ridge_estimate(raw,public['pilot_qpsk']).astype(np.complex64)
    cond=conditions(raw,initial,public['pilot_qpsk'])
    target=np.load(targets/'train_response.npy',mmap_mode='r');prior=np.load(targets/'training_covariance.npy')
    model=Model().cuda();model.load_state_dict(torch.load(checkpoint/'weights.pt',map_location='cuda',weights_only=True));model.eval()
    cov=np.zeros((2,17,64,64),complex);counts=np.zeros(17,int);started=time.perf_counter()
    for begin in range(0,len(raw),64):
        end=min(begin+64,len(raw))
        with torch.inference_mode():cnn=model(torch.from_numpy(initial[begin:end]).cuda(),torch.from_numpy(cond[begin:end]).cuda()).cpu().numpy()
        traditional=np.stack([covariance_estimate(x,public['pilot_qpsk'],prior) for x in raw[begin:end]])
        for mi,prediction in enumerate([traditional,cnn]):
            scale2=np.maximum(np.mean(abs(prediction.astype(complex))**2,axis=(1,2)),1e-24)
            error=(np.asarray(target[begin:end],complex)-prediction)/np.sqrt(scale2)[:,None,None]
            for carrier in np.unique(raw[begin:end,1984].astype(int)):
                mask=raw[begin:end,1984]==carrier;e=error[mask]
                cov[mi,carrier-4]+=np.sum(e @ e.conj().transpose(0,2,1),axis=0)
                if mi==0:counts[carrier-4]+=int(mask.sum())*31
        if begin%512==0:
            state=dict(status='fitting',completed=end,total=len(raw),pid=os.getpid(),at=now(),seconds=time.perf_counter()-started)
            write_json(output/'progress.json',state);print(json.dumps(state),flush=True)
    cov/=counts[None,:,None,None];cov=(cov+cov.conj().transpose(0,1,3,2))/2
    if np.any(counts!=3456*31) or np.linalg.eigvalsh(cov).min()<-1e-10*np.max(abs(cov)):
        raise ValueError('拟合计数或正半定性错误。')
    files={}
    for mi,name in enumerate(['covariance','cnn']):
        path=output/(name+'_residual_covariance.npy');np.save(path,cov[mi],allow_pickle=False);files[path.name]=sha256(path)
    verify_sources(sources);verify_targets(data,targets)
    if sha256(checkpoint/'weights.pt')!=meta['weights_sha256']:raise ValueError('固定权重改变。')
    write_json(output/'complete.json',dict(status='complete',at=now(),fingerprint=fingerprint(protocol),
        train_environments=3456,weights_sha256=meta['weights_sha256'],file_sha256=files,network_updates=0))
    write_json(output/'progress.json',dict(status='complete',completed=len(raw),total=len(raw),at=now()))
    print(json.dumps(dict(status='complete',train_samples=len(raw),network_updates=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.project.resolve(),a.output.resolve())
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/('failure_%d.json'%time.time()),dict(at=now(),traceback=traceback.format_exc()));raise
