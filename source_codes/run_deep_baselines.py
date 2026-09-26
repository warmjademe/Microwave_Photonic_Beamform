"""华硕GPU上的统一监督训练、推理与完整冻结接收器评测入口。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import numpy as np
from compact_dataset import CompactDataset, sha256
from generate_native_dataset import write_json, json_bytes, now
from deep_common import preprocessing

METHODS = ['dnn', 'cnn', 'rescnn', 'transformer', 'complex_cnn', 'jct']
SOURCE = Path(__file__).resolve().parent


def atomic_torch(obj, path):
    import torch
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def make_protocol(train, test):
    from native_sim.data import NativeDataset
    from run_native_baselines import verify_core
    if train.split != 'train' or test.split != 'test':
        raise ValueError('训练/测试身份错误。')
    if {e['environment_id'] for e in train.environments} & {e['environment_id'] for e in test.environments}:
        raise ValueError('环境划分交叠。')
    for key in ['generation_fingerprint', 'selection_fingerprint', 'signal_config']:
        if train.metadata[key] != test.metadata[key]:
            raise ValueError('训练/测试来源不同：' + key)
    native = NativeDataset(test.source_root)
    verify_core(native)
    files = [Path(__file__), SOURCE/'compact_dataset.py', SOURCE/'evaluate_deep_baselines.py',
             SOURCE/'summarize_deep_baselines.py']
    files += sorted((SOURCE/'deep_common').glob('*.py'))
    files += [SOURCE/('baseline_'+m)/'method.py' for m in METHODS]
    return dict(schema='mwp-supervised-six-v1', methods=METHODS, seed=0, epochs=40,
        batch_size=256, learning_rate=0.001, optimizer='Adam', weight_decay=0.,
        loss='mean squared error over all 128 normalized continuous controls',
        checkpoint_selection='final epoch 40; no validation and no test tuning',
        quantization='floor(clip(u,0,1)*levels+0.5); levels=76 delay,24 attenuation',
        initial_probes=16, additional_feedback=0, training_only_preprocessing=True,
        train_manifest_sha256=sha256(train.root/'manifest.json'),
        test_manifest_sha256=sha256(test.root/'manifest.json'),
        train_samples=len(train), test_samples=len(test), train_environments=len(train.environments),
        test_environments=len(test.environments), generation_fingerprint=train.metadata['generation_fingerprint'],
        selection_fingerprint=train.metadata['selection_fingerprint'],
        verified_native_core=native.manifest['core_source_sha256'],
        source_sha256={str(p.relative_to(SOURCE)):sha256(p) for p in files},
        evaluation_seed_rule='rng_for(environment_seed,carrier_GHz,0,510); same as previous nine baselines',
        statistics='1000 environment-cluster bootstrap draws, seed 20260924; paired NMSE difference vs DNN',
        architecture_scope='supervised architecture adaptations, not reproductions of original paper results')


def train_and_predict(train_root, test_root, output):
    import torch
    from deep_common.layers import build
    if not torch.cuda.is_available():
        raise RuntimeError('本实验要求华硕CUDA GPU，禁止回退CPU。')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    train = CompactDataset(train_root, verify_hashes=True)
    test = CompactDataset(test_root, verify_hashes=True)
    protocol = make_protocol(train, test)
    fp = hashlib.sha256(json_bytes(protocol)).hexdigest()
    path = root/'study_protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('实验协议已经改变，拒绝覆盖。')
    write_json(path, protocol)
    write_json(root/'runtime.json', dict(host=platform.node(), python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), deterministic=True, tf32=True))
    if (root/'normalization.npz').exists():
        with np.load(root/'normalization.npz') as f:
            if str(f['protocol_fingerprint']) != fp: raise ValueError('标准化版本不同。')
            stats = {k:f[k] for k in ['mean','scale']}
    else:
        stats = preprocessing.fit(train)
        np.savez(root/'normalization.npz', **stats, protocol_fingerprint=fp)
    # 整个数值训练集约1.2GB，一次放入GPU；不复制原始光学缓存。
    def to_gpu(data):
        result = torch.empty((len(data), 2513), device='cuda')
        maximum=0.
        for start in range(0, len(data), 4096):
            z=preprocessing.apply(data.X[start:start+4096],stats)
            maximum=max(maximum,float(abs(z).max()))
            result[start:start+4096] = torch.from_numpy(z).cuda()
        # 训练集按自身方差标准化后，任意值不应超出sqrt(2N)（I/Q共尺度）。
        if data.split=='train' and maximum>np.sqrt(2*len(data))*1.01:
            raise ValueError('标准化尺度无效，拒绝开始训练。')
        write_json(root/('preprocessing_'+data.split+'_audit.json'),dict(max_abs=maximum,samples=len(data),
            fitted_on='train',normalization_sha256=sha256(root/'normalization.npz')))
        return result
    xtrain = to_gpu(train)
    ytrain = torch.from_numpy(np.array(train.Y)).cuda()
    xtest = to_gpu(test)
    for method in METHODS:
        folder = root/method; folder.mkdir(exist_ok=True)
        marker = folder/'complete.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            if record['protocol_fingerprint'] != fp: raise ValueError('模型版本不同。')
            for filename, digest in record['file_sha256'].items():
                if sha256(folder/filename) != digest: raise ValueError('模型文件改变。')
            continue
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = build(method).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
        shuffle = torch.Generator(device='cuda').manual_seed(0)
        history = []; elapsed_before = 0.
        if (folder/'resume.pt').exists():
            old = torch.load(folder/'resume.pt', map_location='cuda', weights_only=False)
            if old['protocol_fingerprint'] != fp: raise ValueError('断点协议不同。')
            model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
            shuffle.set_state(old['shuffle'].cpu()); history = old['history']
            elapsed_before = old['training_seconds']
        started = time.perf_counter(); model.train()
        for epoch in range(len(history)+1, 41):
            order = torch.randperm(len(train), generator=shuffle, device='cuda')
            loss_sum = torch.zeros((), device='cuda'); epoch_start = time.perf_counter()
            for indices in order.split(protocol['batch_size']):
                prediction = model(xtrain[indices])
                loss = torch.nn.functional.mse_loss(prediction, ytrain[indices])
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                loss_sum += loss.detach() * len(indices)
            torch.cuda.synchronize()
            mse = float(loss_sum.item()/len(train))
            if not np.isfinite(mse): raise ValueError('训练损失非有限。')
            history.append(dict(epoch=epoch,mse=mse,seconds=time.perf_counter()-epoch_start))
            duration = elapsed_before + time.perf_counter()-started
            atomic_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                shuffle=shuffle.get_state(), training_seconds=duration, protocol_fingerprint=fp),folder/'resume.pt')
            write_json(folder/'training_loss.json', history)
            write_json(root/'training_progress.json', dict(status='training',method=method,epoch=epoch,
                epochs=40,train_mse=mse,updated_at_utc=now(),protocol_fingerprint=fp))
            print(json.dumps(dict(method=method,**history[-1])),flush=True)
        model.eval()
        atomic_torch(model.state_dict(), folder/'weights.pt')
        with torch.inference_mode():
            # 批量推理保存连续输出；统一接收器随后进行硬件档位量化。
            torch.cuda.synchronize(); inference_start = time.perf_counter()
            outputs = [model(batch).cpu().numpy() for batch in xtest.split(512)]
            torch.cuda.synchronize(); inference_seconds = time.perf_counter()-inference_start
            prediction = np.concatenate(outputs)
            if prediction.shape != (len(test),128) or not np.isfinite(prediction).all(): raise ValueError('预测无效。')
            np.save(folder/'predictions.npy',prediction,allow_pickle=False)
            # 固定取前256条公开观测测单条GPU延迟；不据接收结果挑选样例。
            for _ in range(10): model(xtest[:1])
            torch.cuda.synchronize(); timings=[]
            for i in range(256):
                t=time.perf_counter(); model(xtest[i:i+1]); torch.cuda.synchronize()
                timings.append(time.perf_counter()-t)
        metadata = dict(method=method,seed=0,epochs=40,parameters=sum(p.numel() for p in model.parameters()),
            training_seconds=duration,inference_batch512_seconds=inference_seconds,
            inference_single_mean_seconds=float(np.mean(timings)),inference_single_p95_seconds=float(np.quantile(timings,.95)),
            latency_scope='GPU-resident standardized input to continuous output; excludes transfer/preprocessing',
            protocol_fingerprint=fp,normalization_sha256=sha256(root/'normalization.npz'),
            finished_at_utc=now())
        write_json(folder/'metadata.json',metadata)
        filenames=['weights.pt','predictions.npy','metadata.json','training_loss.json']
        write_json(marker,dict(protocol_fingerprint=fp,file_sha256={f:sha256(folder/f) for f in filenames}))
        del optimizer,model; torch.cuda.empty_cache()
    write_json(root/'training_progress.json',dict(status='complete',methods=METHODS,epochs=40,
        protocol_fingerprint=fp,finished_at_utc=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--test',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try: train_and_predict(args.train,args.test,args.output)
    except BaseException:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/('failure_'+str(int(time.time()))+'.json'),dict(traceback=traceback.format_exc(),time=now()))
        raise


if __name__=='__main__': main()
