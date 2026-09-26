"""原七项直接控制学习基线：同一训练成员、教师标签，保留各自网络实现。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (SOURCE, DEEP_METHODS, LEVELS, require_host,
    public_data, observation, load_split, check_data, sha256, source_record,
    verify_sources, write_json, now)
from baseline_common.data import features
from baseline_mlp.train_compact import fit_stream
from baseline_mlp.method import load as load_mlp
from our_method_quality_rank.train import TrainingArray
from deep_common.preprocessing import fit, apply
from deep_common.layers import build
from run_deep_baselines import atomic_torch


def verify_labels(data, labels):
    manifest = check_data(data)
    protocol = json.loads((labels/'protocol.json').read_text())
    complete = json.loads((labels/'complete.json').read_text())
    expected_ids = [r['environment_id'] for r in manifest['environments'] if r['split']=='train']
    if protocol['train_ids'] != expected_ids or protocol['uses_test'] or protocol['uses_payload']:
        raise ValueError('教师标签训练来源不正确。')
    if complete['status'] != 'complete':
        raise ValueError('标签尚未完成。')
    for name, expected in complete['file_sha256'].items():
        if sha256(labels/name) != expected:
            raise ValueError('教师标签哈希不同。')
    return protocol


def run(data, labels, output):
    require_host()
    if not torch.cuda.is_available():
        raise RuntimeError('六网络必须在华硕CUDA运行，不回退CPU。')
    verify_labels(data, labels)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    train_x, _, _, train_rows = load_split(data, 'train', labels=False)
    test_x, _, _, test_rows = load_split(data, 'test', labels=False)
    if {r['seed'] for r in train_rows} & {r['seed'] for r in test_rows}:
        raise ValueError('训练测试环境交叠。')
    y = np.load(labels/'train_Y.npy')
    if y.shape != (len(train_x), 128):
        raise ValueError('标签数量不匹配。')
    files = ['study_full_baselines/train.py', 'study_full_baselines/common.py']
    files += [str(p.relative_to(SOURCE)) for folder in ['deep_common','baseline_mlp']
              for p in (SOURCE/folder).glob('*.py')]
    files += ['baseline_'+m+'/method.py' for m in DEEP_METHODS]
    protocol = dict(schema='full-baseline-unified-train-v1', epochs=40, seed=0,
        deep_batch=256, mlp_batch=64, learning_rate=.001, optimizer='Adam',
        loss='MSE of 128 normalized controls', checkpoint='final epoch',
        validation=False, test_tuning=False, train_samples=len(train_x), test_samples=len(test_x),
        train_ids=[r['environment_id'] for r in train_rows],
        test_ids=[r['environment_id'] for r in test_rows],
        data_manifest_sha256=sha256(data/'manifest.json'), labels_sha256=sha256(labels/'complete.json'),
        source_sha256=source_record(files), methods=['mlp']+DEEP_METHODS,
        mlp_scope='original NumPy float64 implementation and3537 public-derived features',
        deep_scope='original six architectures, same2513 public inputs, float32 TF32off',
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text()) != protocol:
            raise ValueError('训练协议改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in files:
            dest=output/'source_snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/name,dest)
    public=public_data(data)
    # 保留原MLP的特征与训练实现，不能把同名方法换成更弱的占位模型。
    folder=output/'mlp'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        old=json.loads((folder/'complete.json').read_text())
        if (old['fingerprint']!=fingerprint
                or sha256(folder/'predictions.npy')!=old['predictions_sha256']
                or sha256(folder/'model/weights.npz')!=old['weights_sha256']):
            raise ValueError('已完成MLP的来源、权重或预测改变。')
    if not (folder/'complete.json').exists():
        feature_train=np.asarray([features(observation(x,public)) for x in train_x])
        feature_test=np.asarray([features(observation(x,public)) for x in test_x])
        start=time.perf_counter()
        def report(item):
            write_json(output/'progress.json',dict(status='training',method='mlp',
                detail=item,at=now()))
        if (folder/'model/checkpoint.json').exists():
            model=load_mlp(folder/'model')
            if model.metadata.get('fingerprint')!=fingerprint:
                raise ValueError('MLP保存模型来源不同。')
        else:
            model=fit_stream(feature_train,y,epochs=40,batch_size=64,hidden=128,seed=0,
                learning_rate=.001,callback=report,checkpoint_dir=folder/'checkpoints',
                resume=(folder/'checkpoints/protocol.json').exists(),
                provenance=dict(fingerprint=fingerprint))
            model.save(folder/'model',metadata=dict(fingerprint=fingerprint))
        prediction=model.predict_features(feature_test)
        np.save(folder/'predictions.npy',prediction,allow_pickle=False)
        timings=[]
        for i in range(256):
            tick=time.perf_counter(); model.predict_features(features(observation(test_x[i],public)))
            timings.append(time.perf_counter()-tick)
        replay=load_mlp(folder/'model').predict_features(feature_test)
        if not np.array_equal(replay,prediction):
            raise ValueError('MLP权重回放不同。')
        write_json(folder/'complete.json',dict(status='complete',fingerprint=fingerprint,
            epochs=40,seed=0,training_and_inference_seconds=time.perf_counter()-start,
            inference_single_mean_seconds=float(np.mean(timings)),
            inference_single_p95_seconds=float(np.quantile(timings,.95)),
            latency_scope='CPU public feature preparation and forward; no physical measurement',
            predictions_sha256=sha256(folder/'predictions.npy'),
            weights_sha256=sha256(folder/'model/weights.npz'), full_replay_equal=True))
        del feature_train,feature_test,model,replay,prediction
    stats=fit(TrainingArray(train_x))
    np.savez(output/'normalization.npz',**stats)
    xtrain=torch.from_numpy(apply(train_x,stats)).cuda()
    xtest=torch.from_numpy(apply(test_x,stats)).cuda()
    ytrain=torch.from_numpy(y).cuda()
    for method in DEEP_METHODS:
        folder=output/method;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            old=json.loads((folder/'complete.json').read_text())
            if old['fingerprint']!=fingerprint or sha256(folder/'predictions.npy')!=old['predictions_sha256']:
                raise ValueError('已完成网络的来源或预测改变。')
            continue
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        model=build(method).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        shuffle=torch.Generator(device='cuda').manual_seed(0)
        history=[];elapsed=0.
        if (folder/'resume.pt').exists():
            saved=torch.load(folder/'resume.pt',map_location='cuda',weights_only=False)
            if saved['fingerprint']!=fingerprint:raise ValueError('断点来源不同。')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            shuffle.set_state(saved['shuffle'].cpu());history=saved['history'];elapsed=saved['seconds']
        start=time.perf_counter()
        for epoch in range(len(history)+1,41):
            model.train();loss_sum=0.
            for indices in torch.randperm(len(xtrain),device='cuda',generator=shuffle).split(256):
                loss=torch.nn.functional.mse_loss(model(xtrain[indices]),ytrain[indices])
                if not torch.isfinite(loss):raise ValueError('非有限训练损失。')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                loss_sum+=float(loss.detach())*len(indices)
            history.append(dict(epoch=epoch,mse=loss_sum/len(xtrain)))
            duration=elapsed+time.perf_counter()-start
            atomic_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                shuffle=shuffle.get_state(),history=history,seconds=duration,
                fingerprint=fingerprint),folder/'resume.pt')
            write_json(folder/'history.json',history)
            progress=dict(status='training',method=method,epoch=epoch,loss=history[-1]['mse'],at=now())
            write_json(output/'progress.json',progress)
            if epoch%10==0:print(json.dumps(progress),flush=True)
        model.eval();atomic_torch(model.state_dict(),folder/'weights.pt')
        with torch.inference_mode():
            prediction=np.concatenate([model(batch).cpu().numpy() for batch in xtest.split(512)])
            np.save(folder/'predictions.npy',prediction,allow_pickle=False)
            for _ in range(10):model(xtest[:1])
            torch.cuda.synchronize();timings=[]
            for i in range(256):
                tick=time.perf_counter();model(xtest[i:i+1]);torch.cuda.synchronize()
                timings.append(time.perf_counter()-tick)
            replay=build(method).cuda()
            replay.load_state_dict(torch.load(folder/'weights.pt',map_location='cuda',weights_only=True))
            replay.eval()
            for start_index in range(0,len(xtest),512):
                values=replay(xtest[start_index:start_index+512]).cpu().numpy()
                if not np.array_equal(values,prediction[start_index:start_index+512]):
                    raise ValueError('完整权重回放不同：'+method)
        if prediction.shape!=(len(test_x),128) or not np.isfinite(prediction).all():
            raise ValueError('预测无效。')
        write_json(folder/'complete.json',dict(status='complete',fingerprint=fingerprint,
            epochs=40,seed=0,training_seconds=duration,parameters=sum(p.numel() for p in model.parameters()),
            inference_single_mean_seconds=float(np.mean(timings)),
            inference_single_p95_seconds=float(np.quantile(timings,.95)),
            latency_scope='GPU standardized-resident-input forward, excludes preprocessing and transfer',
            weights_sha256=sha256(folder/'weights.pt'),predictions_sha256=sha256(folder/'predictions.npy'),
            full_replay_equal=True))
        del model,optimizer,replay;torch.cuda.empty_cache()
    verify_sources(protocol['source_sha256']);verify_labels(data,labels)
    write_json(output/'progress.json',dict(status='complete',methods=['mlp']+DEEP_METHODS,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','labels','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:run(a.data,a.labels,a.output)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
