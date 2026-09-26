"""固定监督复响应训练；只在完整train输入准备好后启动。"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_quality_rank.train import load_split, check_data
from our_method_response_control.physics import ridge_estimate
from our_method_response_control.model import Model, conditions
from compact_dataset import sha256
from generate_native_dataset import write_json, now
from run_deep_baselines import atomic_torch


def verify_targets(data, targets):
    manifest = json.loads((data/'manifest.json').read_text())
    protocol = json.loads((targets/'protocol.json').read_text())
    complete = json.loads((targets/'complete.json').read_text())
    if complete['status'] != 'complete':
        raise ValueError('响应标签尚未完成。')
    if sha256(targets/'train_response.npy') != complete['label_sha256']:
        raise ValueError('训练响应标签改变。')
    if sha256(targets/'training_covariance.npy') != complete['covariance_sha256']:
        raise ValueError('协方差文件改变。')
    ids = [r['environment_id'] for r in manifest['environments'] if r['split'] == 'train']
    if ids != protocol['train_environment_ids']:
        raise ValueError('训练环境身份或顺序不同。')
    if hashlib.sha256(json.dumps(manifest['environments'], sort_keys=True).encode()).hexdigest() != protocol['source_environment_plan_sha256']:
        raise ValueError('传播环境计划改变。')
    for f in ['our_method_response_control/physics.py', 'diagnostics/linear_centered.py',
              'diagnostics/carrier_reference.py']:
        if sha256(SOURCE/f) != protocol['source_sha256'][f]:
            raise ValueError('标签所用物理源码改变。')


def precision():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def run(data, targets, output):
    if platform.node() != 'qyb-HuaShuo' or not torch.cuda.is_available():
        raise RuntimeError('仅华硕GPU执行。')
    check_data(data); verify_targets(data, targets); precision()
    output.mkdir(exist_ok=False, parents=True)
    files = [p for p in (SOURCE/'our_method_response_control').glob('*')
             if p.suffix in ['.py', '.md']]
    protocol = dict(created_at_utc=now(), epochs=40, seed=0, batch_size=64,
        learning_rate=.001, optimizer='Adam', gradient_clip_norm=5., loss='mean per-sample relative complex-response MSE',
        checkpoint='final epoch', validation=False, test_labels_loaded=False,
        precision='float32/complex64; TF32 disabled',
        data_manifest_sha256=sha256(data/'manifest.json'), targets_complete_sha256=sha256(targets/'complete.json'),
        source_sha256={str(p.relative_to(SOURCE)): sha256(p) for p in files},
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        host=platform.node(), train_samples=14688)
    write_json(output/'protocol.json', protocol)
    for p in files:
        destination = output/'source_snapshot'/p.name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, destination)
    raw, _, _, _ = load_split(data, 'train', labels=False)
    with np.load(data/'public.npz') as f:
        pilots = f['pilot_qpsk']
    initial = ridge_estimate(raw, pilots).astype(np.complex64)
    cond = torch.from_numpy(conditions(raw, initial, pilots)).cuda()
    initial = torch.from_numpy(initial).cuda()
    target = torch.from_numpy(np.load(targets/'train_response.npy')).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = Model().cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    shuffle = torch.Generator(device='cuda').manual_seed(0)
    history = []; started = time.perf_counter()
    for epoch in range(1, 41):
        model.train(); total = 0.
        for ix in torch.randperm(len(target), device='cuda', generator=shuffle).split(64):
            prediction = model(initial[ix], cond[ix])
            loss = ((prediction-target[ix]).abs().square().mean((-1, -2))
                /target[ix].abs().square().mean((-1, -2)).clamp_min(1e-24)).mean()
            if not torch.isfinite(loss):
                raise ValueError('非有限监督损失。')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += float(loss.detach())*len(ix)
        history.append(dict(epoch=epoch, loss=total/len(target)))
        write_json(output/'history.json', history)
        write_json(output/'progress.json', dict(status='training', epoch=epoch,
            loss=history[-1]['loss'], seconds=time.perf_counter()-started))
        if epoch % 5 == 0:
            atomic_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                shuffle=shuffle.get_state()), output/'resume.pt')
            print(json.dumps(history[-1]), flush=True)
    atomic_torch(model.state_dict(), output/'weights.pt')
    check_data(data); verify_targets(data, targets)
    write_json(output/'complete.json', dict(status='complete', epochs=40, seed=0,
        parameters=sum(p.numel() for p in model.parameters()), training_seconds=time.perf_counter()-started,
        weights_sha256=sha256(output/'weights.pt'), history_sha256=sha256(output/'history.json'),
        finished_at_utc=now()))
    write_json(output/'progress.json', dict(status='complete', epoch=40, seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'targets', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.data, a.targets, a.output)
    except BaseException:
        if a.output.exists():
            write_json(a.output/'failure.json', dict(traceback=traceback.format_exc(), at=now()))
        raise
