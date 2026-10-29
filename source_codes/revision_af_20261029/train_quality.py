"""四个预先固定的 CNN/Transformer 质量训练配置；只加载训练环境。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import SOURCE,require_host,sha256,verify_sources,write_json,now,load_split,atomic_npz,LEVELS
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.online_controller import OnlineController
from our_method_response_control.train import verify_targets,precision
from our_method_response_control.physics import decode
from our_method_quality_rank.train import TrainingArray
from deep_common.preprocessing import fit,apply
from deep_common.layers import build
from run_deep_baselines import atomic_torch
from revision_af_20261029.quality_control import QualityObjective,legal_codes

CONFIGS=[(m,i) for m in ['cnn','transformer'] for i in ['scratch','mse_init']]


def read(p):return json.loads(Path(p).read_text())


def fp(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def preflight(output,raw,target,stats,bundle,public):
    ids=[int(np.flatnonzero(raw[:,1984]==f)[0]) for f in [4,12,20]]
    response=torch.as_tensor(target[ids],device='cuda',dtype=torch.complex128)
    frequency=torch.as_tensor(raw[ids,1984]-4,device='cuda',dtype=torch.long)
    rng=np.random.default_rng(20261030)
    controls=rng.integers(0,77,size=(3,128));controls[:,64:]=rng.integers(0,25,size=(3,64))
    objective=QualityObjective(torch.float64).cuda()
    normalized=torch.as_tensor(controls/LEVELS,device='cuda',dtype=torch.float64)
    losses=objective(normalized,response,frequency).detach().cpu().numpy()
    references=[-decode(target[i],int(raw[i,1984]),u,sweeps=0)[1]['initial_proxy'] for i,u in zip(ids,controls/LEVELS)]
    np.testing.assert_allclose(losses,references,rtol=1e-10,atol=1e-12)
    np.testing.assert_array_equal(legal_codes(normalized).cpu().numpy(),controls)
    boundaries=torch.as_tensor(np.stack([np.zeros(128),np.ones(128)]),device='cuda',dtype=torch.float64)
    np.testing.assert_array_equal(legal_codes(boundaries).cpu().numpy(),np.stack([np.zeros(128),LEVELS]))
    rotate=torch.exp(torch.tensor(.713j,device='cuda',dtype=torch.complex128))
    np.testing.assert_allclose(objective(normalized,response*rotate,frequency).detach().cpu().numpy(),losses,rtol=1e-12,atol=1e-13)
    continuous=torch.as_tensor(rng.uniform(.1,.9,(3,128)),device='cuda',dtype=torch.float64).requires_grad_()
    value=objective(continuous,response,frequency,quantize=False).sum();value.backward()
    gradient=continuous.grad.detach().cpu().numpy();assert np.isfinite(gradient).all() and np.any(gradient!=0)
    errors=[]
    for row,coord in [(0,0),(1,37),(2,88)]:
        a=continuous.detach().clone();b=a.clone();eps=1e-7
        a[row,coord]+=eps;b[row,coord]-=eps
        finite=float((objective(a,response,frequency,False).sum()-objective(b,response,frequency,False).sum())/(2*eps))
        np.testing.assert_allclose(finite,gradient[row,coord],rtol=3e-5,atol=1e-7)
        errors.append(abs(finite-gradient[row,coord]))
    single=QualityObjective().cuda()
    float_result=single(normalized.float(),response.to(torch.complex64),frequency).detach().cpu().numpy()
    np.testing.assert_allclose(float_result,losses,rtol=2e-4,atol=2e-5)
    replay={}
    x=torch.from_numpy(apply(raw[ids],stats)).cuda()
    for method,initialization in CONFIGS:
        torch.manual_seed(0);torch.cuda.manual_seed_all(0);model=build(method).cuda()
        original=None
        if initialization=='mse_init':
            original=OnlineController(bundle,method,public);model.load_state_dict(original.model.state_dict())
        model.eval()
        with torch.inference_mode(): y=model(x)
        copied=build(method).cuda();copied.load_state_dict(model.state_dict());copied.eval()
        with torch.inference_mode():np.testing.assert_array_equal(y.cpu().numpy(),copied(x).cpu().numpy())
        if original is not None:
            with torch.inference_mode():np.testing.assert_array_equal(y.cpu().numpy(),original.model(x).cpu().numpy())
            original.verify()
        loss=single(model(x),response.to(torch.complex64),frequency).mean();loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        replay[method+'_'+initialization]=dict(parameters=sum(p.numel() for p in model.parameters()),loss=float(loss.detach()))
        del original,model,copied
    proof=dict(status='passed',at=now(),carriers=[4,12,20],decode_objective_equal=True,
        phase_rotation_invariant=True,continuous_gradient_absolute_errors=errors,
        straight_through_scope='continuous-relaxation gradient only; discrete true derivative not asserted',
        float32_absolute_error=float(np.max(abs(float_result-losses))),models=replay)
    write_json(output/'preflight.json',proof)
    del objective,single,response,x;torch.cuda.empty_cache()


def run(project,preflight_only=False):
    require_host();precision()
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<5*1024**3:raise RuntimeError('Need 5 GiB free GPU memory')
    root=project/'dataset_simulation';output=root/'revision_af_20261029/C/training'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=root/'outputs/scaling_train_3456_20260925';targets=root/'baseline_results/20260925_full_baselines/scale_3456/targets'
        base=read(root/'revision_rq2_af_20261029/full/protocol.json')
        bundle=Path(base['prior']['runtime_bundle']);verify_bundle(bundle)
        verify_targets(data,targets)
        raw,_,_,rows=load_split(data,'train',labels=False)
        assert len(rows)==3456 and raw.shape==(58752,2513)
        target=np.load(targets/'train_response.npy',allow_pickle=False);assert target.shape==(58752,64,31)
        with np.load(bundle/'public.npz') as f:public={k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
        original=OnlineController(bundle,'cnn',public);stats=original.stats
        fitted=fit(TrainingArray(raw))
        for k in stats:np.testing.assert_array_equal(stats[k],fitted[k])
        original.verify();del original
        sources=dict(base['source_sha256'])
        for n in ['train_quality.py','quality_control.py','C_PROTOCOL.md']:
            path=SOURCE/'revision_af_20261029'/n;sources[str(path.relative_to(SOURCE))]=sha256(path)
        verify_sources(sources)
        artifacts={str(data/'manifest.json'):sha256(data/'manifest.json'),str(targets/'complete.json'):sha256(targets/'complete.json'),str(targets/'train_response.npy'):sha256(targets/'train_response.npy'),str(bundle/'manifest.json'):sha256(bundle/'manifest.json')}
        for method in ['cnn','transformer']:
            old=OnlineController(bundle,method,public);artifacts.update(old.artifacts);del old
        p=dict(schema='quality-objective-direct-controls-v1',configs=CONFIGS,seed=0,quality_epochs=40,
            mse_initialization_epochs=40,batch_size=256,learning_rate=.001,optimizer='Adam',gradient_clip=5.,
            checkpoint='final epoch',loss='mean known-response MMSE surrogate; integer-forward straight-through',
            input_features=2513,output_controls=128,train_ids=[r['environment_id'] for r in rows],
            train_samples=58752,validation_or_test_selection=False,runtime_bundle=str(bundle),
            source_sha256=sources,artifact_sha256=artifacts,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
        if (output/'protocol.json').exists():assert read(output/'protocol.json')==json.loads(json.dumps(p))
        else:write_json(output/'protocol.json',p)
        for n,h in sources.items():
            path=output/'source_snapshot'/n;path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert sha256(path)==h
            else:shutil.copyfile(SOURCE/n,path)
        if not (output/'normalization.npz').exists():atomic_npz(output/'normalization.npz',**stats)
        else:
            with np.load(output/'normalization.npz') as a:
                for k in stats:np.testing.assert_array_equal(a[k],stats[k])
        preflight(output,raw,target,stats,bundle,public)
        if preflight_only:return
        x=torch.from_numpy(apply(raw,stats)).cuda();h=torch.from_numpy(target.astype(np.complex64)).cuda()
        frequency=torch.as_tensor(raw[:,1984]-4,device='cuda',dtype=torch.long)
        del target,raw
        objective=QualityObjective().cuda()
        for method,initialization in CONFIGS:
            name=method+'_'+initialization;folder=output/name;folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json');assert done['protocol_sha256']==fp(p)
                assert sha256(folder/'weights.pt')==done['weights_sha256'];continue
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            model=build(method).cuda()
            if initialization=='mse_init':
                old=OnlineController(bundle,method,public);model.load_state_dict(old.model.state_dict());old.verify();del old
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            shuffle=torch.Generator(device='cuda').manual_seed(0)
            history=[];elapsed=0.
            if (folder/'resume.pt').exists():
                saved=torch.load(folder/'resume.pt',map_location='cuda',weights_only=False)
                assert saved['protocol_sha256']==fp(p)
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                shuffle.set_state(saved['shuffle'].cpu());history=saved['history'];elapsed=saved['seconds']
                torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
            start=time.perf_counter()
            for epoch in range(len(history)+1,41):
                model.train();total=0.
                for index in torch.randperm(len(x),device='cuda',generator=shuffle).split(256):
                    loss=objective(model(x[index]),h[index],frequency[index]).mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite quality loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                    optimizer.step();total+=float(loss.detach())*len(index)
                history.append(dict(epoch=epoch,training_proxy=total/len(x)))
                seconds=elapsed+time.perf_counter()-start
                atomic_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),shuffle=shuffle.get_state(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,seconds=seconds,protocol_sha256=fp(p)),folder/'resume.pt')
                write_json(folder/'history.json',history)
                write_json(output/'progress.json',dict(status='training',method=name,epoch=epoch,epochs=40,loss=history[-1]['training_proxy'],at=now()))
                print(json.dumps(dict(method=name,**history[-1])),flush=True)
            atomic_torch(model.state_dict(),folder/'weights.pt')
            model.eval();clone=build(method).cuda();clone.load_state_dict(torch.load(folder/'weights.pt',map_location='cuda',weights_only=True));clone.eval()
            with torch.inference_mode():np.testing.assert_array_equal(model(x[:64]).cpu().numpy(),clone(x[:64]).cpu().numpy())
            write_json(folder/'complete.json',dict(status='complete',method=method,initialization=initialization,
                quality_epochs=40,total_training_epochs=40 if initialization=='scratch' else 80,seed=0,
                protocol_sha256=fp(p),training_seconds=seconds,parameters=sum(v.numel() for v in model.parameters()),
                weights_sha256=sha256(folder/'weights.pt'),history_sha256=sha256(folder/'history.json'),weight_replay_equal=True,at=now()))
            del model,clone,optimizer;torch.cuda.empty_cache()
        verify_sources(sources)
        for path,digest in artifacts.items():assert sha256(path)==digest
        write_json(output/'complete.json',dict(status='complete',models=[m+'_'+i for m,i in CONFIGS],
            file_sha256={n:sha256(output/n) for n in ['protocol.json','normalization.npz','preflight.json']},
            source_sha256=sources,at=now(),test_results_used_for_selection=False))
        write_json(output/'progress.json',dict(status='complete',at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');a=parser.parse_args()
    try:run(a.project.resolve(),a.preflight_only)
    except BaseException:
        output=a.project/'dataset_simulation/revision_af_20261029/C/training';output.mkdir(parents=True,exist_ok=True)
        write_json(output/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
