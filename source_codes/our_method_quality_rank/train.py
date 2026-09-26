"""华硕GPU监督训练：训练器仅加载train成员，固定40轮后保存最后模型。"""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time
import traceback
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_quality_rank.common import METHODS, make_model, loss_for
from compact_dataset import sha256
from generate_native_dataset import write_json, now, hashes
from deep_common.preprocessing import fit, apply
from run_deep_baselines import atomic_torch


def check_data(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('数据生成尚未完成。')
    if hashes() != manifest['core_source_sha256']:
        raise ValueError('冻结核心改变。')
    for name, digest in manifest['source_sha256'].items():
        if name.endswith('.py') and sha256(SOURCE/name) != digest:
            raise ValueError('源码改变：'+name)
    if sha256(root/'public.npz') != manifest['public_sha256']:
        raise ValueError('公开配置改变。')
    if sha256(root/'records.json') != manifest['records_sha256']:
        raise ValueError('文件索引改变。')
    return manifest


def load_split(root, split, labels=False):
    if split == 'test' and labels:
        raise ValueError('此读取器禁止加载测试监督标签。')
    manifest = check_data(root); records = {
        r['environment_id']: r for r in json.loads((root/'records.json').read_text())}
    rows = [r for r in manifest['environments'] if r['split'] == split]
    x, single, robust = [], [], []
    for row in rows:
        folder = root/row['path']
        for name, digest in records[row['environment_id']]['file_sha256'].items():
            if sha256(folder/name) != digest:
                raise ValueError('数据SHA不一致：'+str(folder/name))
        with np.load(folder/'data.npz') as f:
            x.append(f['X'])
            if labels:
                single.append(f['single_nmse']); robust.append(f['robust_nmse'])
    return (np.concatenate(x), np.concatenate(single) if labels else None,
            np.concatenate(robust) if labels else None, rows)


class TrainingArray:
    split = 'train'
    def __init__(self, x):
        self.X = x
    def __len__(self):
        return len(self.X)


def run(data, output):
    if platform.node() != 'qyb-HuaShuo' or not torch.cuda.is_available():
        raise RuntimeError('需华硕GPU。')
    manifest = check_data(data)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    protocol = dict(created_at_utc=now(), seed=0, epochs=40, batch_size=256,
        learning_rate=.001, optimizer='Adam', checkpoint='final epoch', methods=METHODS,
        data_manifest_sha256=sha256(data/'manifest.json'), train_samples=14688,
        test_labels_loaded=False, validation=False, temperature=.05,
        host=platform.node(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        python=platform.python_version(), cuda=torch.version.cuda,
        source_sha256=manifest['source_sha256'])
    write_json(output/'protocol.json', protocol)
    x, single, robust, rows = load_split(data, 'train', labels=True)
    stats = fit(TrainingArray(x)); np.savez(output/'normalization.npz', **stats)
    x = torch.from_numpy(apply(x, stats)).cuda()
    single = torch.from_numpy(single).cuda(); robust = torch.from_numpy(robust).cuda()
    with np.load(data/'public.npz') as f:
        controls = torch.as_tensor(f['catalog_controls'], device='cuda', dtype=torch.float32)
    for method in METHODS:
        folder = output/method; folder.mkdir()
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = make_model(method).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        shuffle = torch.Generator(device='cuda').manual_seed(0)
        started = time.perf_counter(); history = []
        for epoch in range(1, 41):
            model.train(); loss_sum = 0.
            for indices in torch.randperm(len(x), device='cuda', generator=shuffle).split(256):
                loss = loss_for(method, model(x[indices]), single[indices], robust[indices], controls)
                if not torch.isfinite(loss):
                    raise ValueError('损失非有限。')
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                loss_sum += float(loss.detach())*len(indices)
            history.append(dict(epoch=epoch, loss=loss_sum/len(x)))
            write_json(folder/'history.json', history)
            write_json(output/'progress.json', dict(status='training', method=method,
                epoch=epoch, loss=history[-1]['loss'], updated_at_utc=now()))
            if epoch % 5 == 0:
                print(json.dumps(dict(method=method, **history[-1])), flush=True)
        torch.cuda.synchronize()
        atomic_torch(model.state_dict(), folder/'weights.pt')
        write_json(folder/'complete.json', dict(method=method, seed=0, epochs=40,
            parameters=sum(p.numel() for p in model.parameters()),
            training_seconds=time.perf_counter()-started,
            weights_sha256=sha256(folder/'weights.pt'), history_sha256=sha256(folder/'history.json')))
    check_data(data)
    write_json(output/'progress.json', dict(status='complete', methods=METHODS, finished_at_utc=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.data, a.output)
    except BaseException:
        if a.output.exists():
            write_json(a.output/'failure.json', dict(traceback=traceback.format_exc(), at=now()))
        raise
