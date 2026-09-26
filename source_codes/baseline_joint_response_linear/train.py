"""使用相同训练响应目标，建立联合频率及相对损失匹配的线性对照。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from our_method_response_control.physics import pilot_gain
from our_method_response_control.train import precision, verify_targets
from study_full_baselines.scale_large import freeze, fingerprint

METHODS = ['per_tone_relative', 'joint_relative', 'joint_absolute']


def fit_one(y, h, method):
    """张量维度：环境×16探测×31频点，环境×64路径×31频点。"""
    y = y.to(torch.complex128); h = h.to(torch.complex128)
    if method.endswith('relative'):
        scale = h.abs().square().mean((-1, -2)).sqrt().clamp_min(1e-12)
    else:
        # 共同尺度只改善数值范围，不改变未加权最小二乘及相对加载。
        scale = h.abs().square().mean().sqrt().clamp_min(1e-12).expand(len(h))
    yy = y/scale[:, None, None]; hh = h/scale[:, None, None]
    if method.startswith('joint'):
        yy = yy.reshape(len(y), -1); hh = hh.reshape(len(h), -1)
    else:
        yy = yy.permute(2, 0, 1); hh = hh.permute(2, 0, 1)
    gram = yy.mH @ yy; cross = yy.mH @ hh
    loading = .01*gram.diagonal(dim1=-2, dim2=-1).real.mean(-1)
    matrix = gram+loading[..., None, None]*torch.eye(gram.shape[-1], device='cuda')
    w = torch.linalg.solve(matrix, cross)
    residual = (matrix@w-cross).abs().square().sum().sqrt()/cross.abs().square().sum().sqrt().clamp_min(1e-30)
    if not torch.isfinite(w).all() or float(residual)>1e-9:
        raise ValueError('线性估计方程残差未通过。')
    return w, float(residual)


def predict(y, w, method):
    y = y.to(torch.complex128)
    if method.startswith('joint'):
        return (y.reshape(len(y), -1)@w).reshape(-1, 64, 31)
    return (y.permute(2, 0, 1)@w).permute(1, 2, 0)


def run(data, targets, output):
    require_host(); precision()
    if not torch.cuda.is_available():
        raise RuntimeError('线性矩阵求解和推理仅在华硕GPU。')
    manifest = check_data(data); verify_targets(data, targets)
    raw, _, _, rows = load_split(data, 'train', labels=False)
    if len(rows)!=864 or len([r for r in manifest['environments'] if r['split']=='test'])!=216:
        raise ValueError('本入口固定原864/216探索性划分。')
    public = public_data(data); measurement = pilot_gain(raw, public['pilot_qpsk'])
    target = np.load(targets/'train_response.npy', mmap_mode='r')
    source_names = ['baseline_joint_response_linear/train.py','baseline_joint_response_linear/README.md',
        'study_full_baselines/common.py', 'study_full_baselines/scale_large.py',
        'our_method_response_control/physics.py']
    protocol = dict(methods=METHODS, relative_loading=.01, train_environments=864,
        train_samples=len(raw), test_scope='old216 exploratory only', no_test_labels=True,
        data_sha256=sha256(data/'manifest.json'), targets_sha256=sha256(targets/'complete.json'),
        source_sha256=source_record(source_names), precision='complex128 GPU')
    freeze(output, protocol); fp = fingerprint(protocol)
    if (output/'complete.json').exists():
        raise FileExistsError('已完成线性基线不重复运行。')
    weights = {}; residuals = {}; fitting_seconds = {}; start = time.perf_counter()
    for method in METHODS:
        folder = output/method; folder.mkdir(exist_ok=True); fitted = []; errors = []
        tick = time.perf_counter()
        for fc in range(4, 21):
            path = folder/('weights_%02d.npy'%fc); marker = path.with_suffix('.json')
            if marker.exists():
                meta = json.loads(marker.read_text())
                if meta['fingerprint']!=fp or sha256(path)!=meta['sha256']:
                    raise ValueError('已保存线性模型来源不同。')
                fitted.append(np.load(path)); errors.append(meta['relative_residual']); continue
            if path.exists():
                raise ValueError('未提交线性模型须先检查。')
            ix = np.flatnonzero(raw[:, 1984]==fc)
            y = torch.from_numpy(measurement[ix]).cuda()
            h = torch.from_numpy(np.array(target[ix])).cuda()
            w, residual = fit_one(y, h, method)
            # 同相位旋转输入应只旋转输出，不能改变相对幅度。
            z = np.exp(.7j)
            assert torch.allclose(predict(y[:2]*z, w, method), predict(y[:2], w, method)*z,
                                  rtol=1e-10, atol=1e-12)
            arr = w.cpu().numpy(); np.save(path, arr, allow_pickle=False)
            write_json(marker, dict(fingerprint=fp, sha256=sha256(path), relative_residual=residual))
            fitted.append(arr); errors.append(residual)
            state = dict(status='fitting', method=method, carrier_ghz=fc, pid=os.getpid(), at=now())
            write_json(output/'progress.json', state)
        weights[method] = fitted; residuals[method] = errors
        fitting_seconds[method] = time.perf_counter()-tick
    # 全部模型拟合完成后，才打开旧测试公开输入。
    x, _, _, test_rows = load_split(data, 'test', labels=False)
    ytest = pilot_gain(x, public['pilot_qpsk']); records = []
    for method in METHODS:
        predictions = np.empty((len(x), 64, 31), np.complex128)
        for fc in range(4, 21):
            ix = np.flatnonzero(x[:, 1984]==fc); y = torch.from_numpy(ytest[ix]).cuda()
            w = torch.from_numpy(weights[method][fc-4]).cuda()
            predictions[ix] = predict(y, w, method).cpu().numpy()
            stored = torch.from_numpy(np.load(output/method/('weights_%02d.npy'%fc))).cuda()
            if not np.array_equal(predict(y, stored, method).cpu().numpy(), predictions[ix]):
                raise ValueError('线性权重回放不同。')
        path = output/method/'predicted_response.npy'; np.save(path, predictions, allow_pickle=False)
        record = dict(method=method, prediction_sha256=sha256(path),
            weights_sha256={str(fc):sha256(output/method/('weights_%02d.npy'%fc)) for fc in range(4,21)},
            maximum_relative_equation_residual=max(residuals[method]), full_prediction_replay=True,
            fit_wall_seconds_this_invocation=fitting_seconds[method],
            real_parameter_count=int(sum(2*w.size for w in weights[method])))
        write_json(output/method/'complete.json', record); records.append(record)
    verify_sources(protocol['source_sha256']); verify_targets(data, targets)
    write_json(output/'complete.json', dict(status='complete', fingerprint=fp, records=records,
        wall_seconds_this_invocation=time.perf_counter()-start, at=now()))
    write_json(output/'progress.json', dict(status='complete', methods=METHODS, at=now()))
    print(json.dumps([dict(method=r['method'], max_residual=r['maximum_relative_equation_residual'],
        parameters=r['real_parameter_count']) for r in records]), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'targets', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.data, a.targets, a.output)
    except BaseException:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
