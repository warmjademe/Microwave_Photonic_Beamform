"""核查可变规模入口保留原数值算法，并在华硕GPU检查实际训练样本。"""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.fair_training_common import cohort_metadata, check_members
from our_method_response_control.train import precision, verify_targets
from our_method_response_control.physics import ridge_estimate, pilot_gain
from our_method_response_control.model import conditions
from deep_common.preprocessing import fit, apply
from our_method_quality_rank.train import TrainingArray
from baseline_response_realcnn import train_scaled as real
from baseline_joint_response_linear import train_scaled as linear
from our_method_quality_rank import train_scaled as quality


def numerical_ast(folder, target):
    def node(filename):
        tree = ast.parse((SOURCE/folder/filename).read_text())
        run = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'run')
        loops = [x for x in run.body if isinstance(x, ast.For) and isinstance(x.target, ast.Name)
                 and x.target.id == target]
        if not loops: raise ValueError('找不到应保持不变的训练/拟合循环。')
        return [ast.dump(x, include_attributes=False) for x in loops]
    if node('train.py') != node('train_scaled.py'):
        raise ValueError('数值训练或拟合循环与原实现不同：'+folder)


def run(project, data, targets, output):
    require_host(); precision()
    if output.exists(): raise FileExistsError('不覆盖入口核查记录。')
    if not torch.cuda.is_available(): raise RuntimeError('必须使用华硕GPU。')
    cohort = cohort_metadata(data); verify_targets(data, targets)
    manifest = check_data(data)
    identity = json.loads((data/'view_identity.json').read_text())
    original = check_data(project/'dataset_simulation/outputs/quality_rank_hybrid_20260925')
    checked = []
    for folder, target in [('baseline_response_realcnn', 'epoch'),
                           ('baseline_joint_response_linear', 'method'),
                           ('our_method_quality_rank', 'method')]:
        numerical_ast(folder, target); checked.append(folder)
    # 旧、新线性方程及推理本体也逐项比较，防止仅循环相同但公式改变。
    trees = [ast.parse((SOURCE/'baseline_joint_response_linear'/n).read_text())
             for n in ['train.py', 'train_scaled.py']]
    for name in ['fit_one', 'predict']:
        bodies = [ast.dump(next(x for x in t.body if isinstance(x, ast.FunctionDef) and x.name == name),
                          include_attributes=False) for t in trees]
        if bodies[0] != bodies[1]: raise ValueError('线性公式改变。')
    negative = []
    for kind in ['test_seed_overlap', 'test_member_change', 'training_order', 'sample_count', 'stratum_balance']:
        changed = copy.deepcopy(manifest)
        train = [r for r in changed['environments'] if r['split'] == 'train']
        test = [r for r in changed['environments'] if r['split'] == 'test']
        if kind == 'test_seed_overlap': train[0]['seed'] = test[0]['seed']
        if kind == 'test_member_change': test[0]['seed'] += 1
        if kind == 'training_order': train[0]['index'] = 1
        if kind == 'sample_count': changed['train_samples'] += 1
        if kind == 'stratum_balance': train[0]['joint_stratum_id'] = train[1]['joint_stratum_id']
        try: check_members(changed, identity, original)
        except ValueError: negative.append(kind)
        else: raise ValueError('错误成员没有拒绝：'+kind)
    x, single, robust, rows = load_split(data, 'train', labels=True)
    if len(x) != cohort['train_samples'] or len(rows) != cohort['train_environments']:
        raise ValueError('实际数据数量不符。')
    labels = np.load(targets/'train_response.npy', mmap_mode='r')
    if labels.shape != (len(x), 64, 31): raise ValueError('响应标签形状不符。')
    public = public_data(data); index = np.arange(64)
    initial = ridge_estimate(x[index], public['pilot_qpsk']).astype(np.complex64)
    condition = conditions(x[index], initial, public['pilot_qpsk'])
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = real.Model().cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    initial = torch.from_numpy(initial).cuda(); condition = torch.from_numpy(condition).cuda()
    target = torch.from_numpy(np.array(labels[index])).cuda(); prediction = model(initial, condition)
    loss = ((prediction-target).abs().square().mean((-1,-2))
            /target.abs().square().mean((-1,-2)).clamp_min(1e-24)).mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
    assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in model.parameters())
    real_check = dict(batch=64, parameters=sum(p.numel() for p in model.parameters()), loss=float(loss.detach()))
    assert real_check['parameters'] == 17524
    del model, optimizer, prediction
    measurements = pilot_gain(x, public['pilot_qpsk']); ix = np.flatnonzero(x[:,1984] == 4)[:64]
    y = torch.from_numpy(measurements[ix]).cuda(); h = torch.from_numpy(np.array(labels[ix])).cuda()
    linear_checks = []
    for method in linear.METHODS:
        w, residual = linear.fit_one(y, h, method)
        predicted = linear.predict(y[:2], w, method)
        assert predicted.shape == (2,64,31) and torch.isfinite(predicted).all()
        linear_checks.append(dict(method=method, training_rows=len(ix), relative_equation_residual=residual))
    stats = fit(TrainingArray(x)); normalized = torch.from_numpy(apply(x[:256], stats)).cuda()
    a = torch.from_numpy(single[:256]).cuda(); b = torch.from_numpy(robust[:256]).cuda()
    with np.load(data/'public.npz') as f:
        controls = torch.as_tensor(f['catalog_controls'], device='cuda', dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    quality_checks = []
    for method in quality.METHODS:
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = quality.make_model(method).cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss = quality.loss_for(method, model(normalized), a, b, controls)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in model.parameters())
        quality_checks.append(dict(method=method, batch=256, loss=float(loss.detach()),
                                   parameters=sum(p.numel() for p in model.parameters())))
    check_data(data); verify_targets(data, targets)
    output.mkdir(parents=True)
    source_names = ['study_full_baselines/'+n for n in ['check_scaled_trainers.py',
        'fair_training_common.py', 'SCALED_FAIR_TRAINING_PROTOCOL.md']]
    source_names += [folder+'/'+n for folder in checked for n in ['train.py', 'train_scaled.py']]
    result = dict(status='passed_scaled_training_entry_preflight', cohort=cohort,
        unchanged_numerical_loops=checked, unchanged_linear_equations=True,
        negative_checks=negative, real_gpu_batch=real_check, linear_gpu_fits=linear_checks,
        quality_gpu_batches=quality_checks, gpu=torch.cuda.get_device_name(),
        new_formal_models=0, test_supervision_loaded=False,
        scope='Actual864 view and GPU batches; does not replace full training at selected larger cohort',
        source_sha256=source_record(source_names), at=now())
    import shutil
    for name in source_names:
        destination = output/'source_snapshot'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, destination)
    write_json(output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cohort','source_sha256']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project','data','targets','output']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); run(a.project.resolve(),a.data.resolve(),a.targets.resolve(),a.output.resolve())
