"""任意完整嵌套训练规模的响应标签、CNN训练及旧测试公开预测。"""
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
from study_full_baselines.common import *
from our_method_response_control.physics import training_target, ridge_estimate
from our_method_response_control.model import Model, conditions
from our_method_response_control.train import precision, verify_targets
from run_deep_baselines import atomic_torch

SOURCES = ['study_full_baselines/scale_large.py', 'study_full_baselines/SCALE_LARGE_PROTOCOL.md',
    'study_full_baselines/common.py', 'our_method_response_control/model.py',
    'our_method_response_control/physics.py', 'our_method_response_control/train.py',
    'diagnostics/linear_centered.py', 'diagnostics/carrier_reference.py']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze(folder, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        if json.loads((folder/'protocol.json').read_text()) != protocol:
            raise ValueError('规模协议发生变化。')
    else:
        write_json(folder/'protocol.json', protocol)
        for name in protocol['source_sha256']:
            dst = folder/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE/name, dst)


def prepare_targets(data, output):
    """仅从完整训练成员生成标签；逐环境提交可安全续跑。"""
    manifest = check_data(data)
    rows = [r for r in manifest['environments'] if r['split']=='train']; count = len(rows)
    if count % 216 or [r['index'] for r in rows] != list(range(count)):
        raise ValueError('训练成员顺序或规模不满足冻结分层。')
    groups = {}
    for row in rows:
        groups.setdefault(row['joint_stratum_id'], []).append(row['stratum_repeat_index'])
    if len(groups) != 216 or any(sorted(v)!=list(range(count//216)) for v in groups.values()):
        raise ValueError('训练集合未保持联合分层。')
    protocol = dict(samples=count*17, train_environment_ids=[r['environment_id'] for r in rows],
        source_environment_plan_sha256=fingerprint(manifest['environments']),
        data_manifest_sha256=sha256(data/'manifest.json'), source_sha256=source_record(SOURCES),
        target_definition='unchanged effective linear response; original physical parameters',
        no_test_response_labels=True)
    freeze(output, protocol); fp = fingerprint(protocol)
    if (output/'complete.json').exists():
        verify_targets(data, output); return rows
    records = output/'records'; records.mkdir(exist_ok=True)
    for row in rows:
        path = records/('%05d.npz'%row['index']); marker = path.with_suffix('.json')
        if marker.exists():
            meta = json.loads(marker.read_text())
            if meta['fingerprint']!=fp or meta['sha256']!=sha256(path):
                raise ValueError('响应训练标签身份发生变化。')
            continue
        if path.exists():
            raise ValueError('存在未提交的响应标签。')
        env = json.loads((data/row['path']/'environment.json').read_text())
        h = np.asarray([training_target(env, fc) for fc in range(4, 21)], dtype=np.complex64)
        if h.shape != (17, 64, 31) or not np.isfinite(h).all():
            raise ValueError('响应标签非法。')
        atomic_npz(path, response=h)
        write_json(marker, dict(fingerprint=fp, sha256=sha256(path),
            environment_id=row['environment_id']))
        if (row['index']+1)%100 == 0:
            write_json(output/'progress.json', dict(status='preparing', completed=row['index']+1,
                total=count, pid=os.getpid(), at=now()))
    target = np.lib.format.open_memmap(output/'train_response.npy', mode='w+',
                                     dtype=np.complex64, shape=(count*17, 64, 31))
    covariance = np.zeros((17, 64, 64), complex)
    for row in rows:
        path = records/('%05d.npz'%row['index'])
        if sha256(path) != json.loads(path.with_suffix('.json').read_text())['sha256']:
            raise ValueError('汇总前标签哈希变化。')
        with np.load(path) as f:
            h = f['response']; target[row['index']*17:(row['index']+1)*17] = h
        # 使用保存的complex64监督目标估计二阶矩，与规模子集实验相同。
        high = h.astype(np.complex128)
        covariance += high @ high.conj().transpose(0, 2, 1)
    target.flush(); del target
    covariance /= count*31
    if np.linalg.eigvalsh(covariance).min() < -1e-12*abs(covariance).max():
        raise ValueError('协方差非正半定。')
    np.save(output/'training_covariance.npy', covariance, allow_pickle=False)
    verify_sources(protocol['source_sha256']); check_data(data)
    write_json(output/'complete.json', dict(status='complete', samples=count*17,
        label_sha256=sha256(output/'train_response.npy'),
        covariance_sha256=sha256(output/'training_covariance.npy'), at=now()))
    verify_targets(data, output)
    write_json(output/'progress.json', dict(status='complete', completed=count, total=count, at=now()))
    return rows


def run(data, exploratory_data, output):
    require_host(); precision()
    if not torch.cuda.is_available():
        raise RuntimeError('规模训练仅在华硕GPU。')
    rows = prepare_targets(data, output/'targets'); count = len(rows)
    manifest = check_data(exploratory_data)
    if len([r for r in manifest['environments'] if r['split']=='test']) != 216:
        raise ValueError('本入口只允许旧216环境探索性测试。')
    train_seeds = {r['seed'] for r in rows}
    if train_seeds & {r['seed'] for r in manifest['environments'] if r['split']=='test'}:
        raise ValueError('训练与旧测试环境重叠。')
    if sha256(data/'public.npz') != sha256(exploratory_data/'public.npz'):
        raise ValueError('训练与测试公开测量配置不同。')
    public = public_data(data); raw, _, _, _ = load_split(data, 'train', labels=False)
    initial_array = ridge_estimate(raw, public['pilot_qpsk']).astype(np.complex64)
    cond = torch.from_numpy(conditions(raw, initial_array, public['pilot_qpsk'])).cuda()
    initial = torch.from_numpy(initial_array).cuda()
    target = torch.from_numpy(np.load(output/'targets/train_response.npy')).cuda()
    for schedule in ['fixed_epochs', 'equal_updates']:
        folder = output/('response_n%04d_%s'%(count, schedule))
        draws = len(raw) if schedule=='fixed_epochs' else 14688
        if draws > len(raw):
            raise ValueError('此入口用于864及以上训练环境。')
        protocol = dict(train_environments=count, train_samples=len(raw), epochs=40,
            schedule=schedule, sample_draws_per_epoch=draws, updates_per_epoch=int(np.ceil(draws/64)),
            effective_data_passes=40*draws/len(raw), seed=0, batch_size=64,
            learning_rate=.001, gradient_clip=5., optimizer='Adam',
            loss='relative complex response MSE', checkpoint='last epoch', validation=False,
            data_manifest_sha256=sha256(data/'manifest.json'),
            test_manifest_sha256=sha256(exploratory_data/'manifest.json'),
            targets_complete_sha256=sha256(output/'targets/complete.json'),
            source_sha256=source_record(SOURCES), scope='old216 exploratory size curve')
        freeze(folder, protocol); fp = fingerprint(protocol)
        if (folder/'complete.json').exists():
            meta = json.loads((folder/'complete.json').read_text())
            if (meta['fingerprint']!=fp or sha256(folder/'weights.pt')!=meta['weights_sha256']
                    or sha256(folder/'predicted_response.npy')!=meta['prediction_sha256']):
                raise ValueError('已完成规模产物改变。')
            continue
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = Model().cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        shuffle = torch.Generator(device='cuda').manual_seed(0)
        history = []; elapsed = 0.; used = torch.zeros(len(raw), dtype=torch.bool, device='cuda')
        if (folder/'resume.pt').exists():
            old = torch.load(folder/'resume.pt', map_location='cuda', weights_only=False)
            if old['fingerprint']!=fp:
                raise ValueError('规模断点身份不同。')
            model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
            shuffle.set_state(old['shuffle'].cpu()); used = old['used']
            history = old['history']; elapsed = old['seconds']
        started = time.perf_counter(); seconds = elapsed
        for epoch in range(len(history)+1, 41):
            model.train(); order = torch.randperm(len(raw), device='cuda', generator=shuffle)[:draws]
            used[order] = True; loss_sum = 0.
            for ix in order.split(64):
                predicted = model(initial[ix], cond[ix])
                loss = ((predicted-target[ix]).abs().square().mean((-1, -2))
                    /target[ix].abs().square().mean((-1, -2)).clamp_min(1e-24)).mean()
                if not torch.isfinite(loss):
                    raise ValueError('大规模训练损失非有限。')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                loss_sum += float(loss.detach())*len(ix)
            seconds = elapsed+time.perf_counter()-started
            history.append(dict(epoch=epoch, loss=loss_sum/draws))
            atomic_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                shuffle=shuffle.get_state(), used=used, history=history, seconds=seconds,
                fingerprint=fp), folder/'resume.pt')
            write_json(folder/'history.json', history)
            state = dict(status='training', method=folder.name, epoch=epoch, total_epochs=40,
                loss=history[-1]['loss'], pid=os.getpid(), at=now())
            write_json(output/'progress.json', state)
            if epoch%10==0:
                print(json.dumps(state), flush=True)
        atomic_torch(model.state_dict(), folder/'weights.pt'); model.eval()
        # 权重固定后，仅加载已看过的旧测试公开观测。
        x, _, _, _ = load_split(exploratory_data, 'test', labels=False)
        base = ridge_estimate(x, public['pilot_qpsk']).astype(np.complex64)
        test_cond = torch.from_numpy(conditions(x, base, public['pilot_qpsk'])).cuda()
        test_initial = torch.from_numpy(base).cuda()
        with torch.inference_mode():
            predictions = np.concatenate([model(test_initial[i:i+128], test_cond[i:i+128]).cpu().numpy()
                for i in range(0, len(x), 128)])
            np.save(folder/'predicted_response.npy', predictions, allow_pickle=False)
            replay = Model().cuda()
            replay.load_state_dict(torch.load(folder/'weights.pt', map_location='cuda', weights_only=True))
            replay.eval()
            for i in range(0, len(x), 128):
                if not np.array_equal(replay(test_initial[i:i+128], test_cond[i:i+128]).cpu().numpy(),
                                      predictions[i:i+128]):
                    raise ValueError('大规模检查点回放不同。')
        used_array = used.cpu().numpy()
        np.save(folder/'used_training_samples.npy', used_array, allow_pickle=False)
        write_json(folder/'complete.json', dict(status='complete', fingerprint=fp,
            parameters=sum(p.numel() for p in model.parameters()), epochs=40, seed=0,
            training_seconds=seconds, weights_sha256=sha256(folder/'weights.pt'),
            prediction_sha256=sha256(folder/'predicted_response.npy'),
            used_samples=int(used_array.sum()), used_environments=int(used_array.reshape(-1,17).any(axis=1).sum()),
            used_samples_sha256=sha256(folder/'used_training_samples.npy'),
            full_prediction_replay=True, at=now()))
        del model, optimizer, replay, test_cond, test_initial; torch.cuda.empty_cache()
    verify_targets(data, output/'targets')
    # 核对协议内记录的来源，而不是重新生成一份即时哈希与自身比较。
    for folder in [output/'targets']+list(output.glob('response_n*')):
        verify_sources(json.loads((folder/'protocol.json').read_text())['source_sha256'])
    write_json(output/'progress.json', dict(status='complete', train_environments=count, at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'exploratory-data', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.data, a.exploratory_data, a.output)
    except BaseException:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
