"""修订E2：原训练集、40轮最终检查点；独立种子1/2及各自训练残差统计。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import (require_host, sha256, write_json, now,
    source_record, verify_sources, SOURCE, load_split, public_data)
from study_full_baselines.scale_large import fingerprint
from our_method_response_control.physics import ridge_estimate
from our_method_response_control.model import Model, conditions
from our_method_response_control.train import precision, verify_targets
from run_deep_baselines import atomic_torch


def read(p):
    return json.loads(Path(p).read_text())


def fit_statistics(folder, raw, initial, cond, target, model):
    """每个种子仅用训练误差拟合B；与原实现的空间/频率矩定义一致。"""
    spatial = np.zeros((17, 64, 64), np.complex128)
    frequency = np.zeros((17, 31, 31), np.complex128)
    counts = np.zeros(17, int)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(raw), 64):
            end = min(begin + 64, len(raw))
            pred = model(initial[begin:end], cond[begin:end]).cpu().numpy()
            scale2 = np.maximum(np.mean(abs(pred.astype(complex))**2, axis=(1, 2)), 1e-24)
            truth = target[begin:end].cpu().numpy().astype(complex)
            error = (truth - pred) / np.sqrt(scale2)[:, None, None]
            for fc in np.unique(raw[begin:end, 1984].astype(int)):
                e = error[raw[begin:end, 1984] == fc]
                spatial[fc-4] += np.sum(e @ e.conj().transpose(0, 2, 1), axis=0)
                frequency[fc-4] += np.sum(e.transpose(0, 2, 1) @ e.conj(), axis=0)
                counts[fc-4] += len(e)
    if np.any(counts != 3456):
        raise ValueError('Residual statistics do not cover all training environments.')
    spatial /= counts[:, None, None] * 31
    spatial = (spatial + spatial.conj().transpose(0, 2, 1))/2
    frequency /= counts[:, None, None] * 64
    frequency = (frequency + frequency.conj().transpose(0, 2, 1))/2
    frequency /= np.trace(frequency, axis1=1, axis2=2).real[:, None, None]/31
    for name, value in [('cnn_residual_covariance.npy', spatial), ('cnn_frequency_covariance.npy', frequency)]:
        if not np.isfinite(value).all() or np.linalg.eigvalsh(value).min() < -1e-10 * max(1., abs(value).max()):
            raise ValueError('Invalid residual statistics.')
        tmp = folder/(name+'.tmp')
        with tmp.open('wb') as f:
            np.save(f, value, allow_pickle=False)
        tmp.replace(folder/name)


def run(project, output, seeds, verify_seed0=False):
    require_host(); precision()
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5*1024**3:
        raise RuntimeError('At least 5 GiB of free CUDA memory is required.')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = project/'dataset_simulation'
    data = root/'outputs/scaling_train_3456_20260925'
    original = root/'baseline_results/20260925_full_baselines/scale_3456'
    targets = original/'targets'
    original_model = original/'response_n3456_fixed_epochs'
    verify_targets(data, targets)
    old = read(original_model/'complete.json'); old_protocol = read(original_model/'protocol.json')
    if sha256(original_model/'weights.pt') != old['weights_sha256']:
        raise ValueError('Original seed-0 model changed.')
    sources = source_record(['study_revision_20261029/train_e2.py',
        'our_method_response_control/model.py', 'our_method_response_control/physics.py',
        'our_method_response_control/train.py', 'study_full_baselines/common.py',
        'run_deep_baselines.py', 'diagnostics/linear_centered.py', 'diagnostics/carrier_reference.py'])
    raw, _, _, rows = load_split(data, 'train', labels=False)
    if len(rows) != 3456 or len(raw) != 58752:
        raise ValueError('Wrong training membership.')
    public = public_data(data)
    base = ridge_estimate(raw, public['pilot_qpsk']).astype(np.complex64)
    initial = torch.from_numpy(base).cuda()
    cond = torch.from_numpy(conditions(raw, base, public['pilot_qpsk'])).cuda()
    target = torch.from_numpy(np.load(targets/'train_response.npy')).cuda()
    del base
    write_json(output/'seed0_reference.json', dict(path=str(original_model),
        complete_sha256=sha256(original_model/'complete.json'), weights_sha256=old['weights_sha256'],
        scope='unchanged original seed 0; not retrained and not selected from multiple seeds'))
    if verify_seed0:
        # Check the new fitting implementation against archived seed-0 B statistics.
        verify_dir = output/'seed0_statistics_check'; verify_dir.mkdir(exist_ok=True)
        model = Model().cuda()
        model.load_state_dict(torch.load(original_model/'weights.pt', map_location='cuda', weights_only=True))
        fit_statistics(verify_dir, raw, initial, cond, target, model)
        references = {
            'cnn_residual_covariance.npy': root/'diagnostics/20260926_measurement_refinement_fit_3456',
            'cnn_frequency_covariance.npy': root/'diagnostics/20260926_joint_refinement_train3456'}
        errors = {}
        for name, parent in references.items():
            a = np.load(verify_dir/name); b = np.load(parent/name)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)
            errors[name] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
        write_json(verify_dir/'complete.json', dict(status='passed', relative_errors=errors, at=now()))
        del model
    for seed in seeds:
        if seed not in (1, 2):
            raise ValueError('Only preregistered new seeds 1 and 2 are permitted.')
        folder = output/('seed_%d'%seed); folder.mkdir(exist_ok=True)
        protocol = dict(seed=seed, epochs=40, batch_size=64, learning_rate=.001,
            optimizer='Adam', gradient_clip=5., checkpoint='last epoch',
            loss='relative complex response MSE', train_environments=3456, train_samples=58752,
            train_environment_ids=[r['environment_id'] for r in rows], validation_used=False,
            test_used=False, data_manifest_sha256=sha256(data/'manifest.json'),
            targets_complete_sha256=sha256(targets/'complete.json'),
            original_seed0_protocol_sha256=sha256(original_model/'protocol.json'), source_sha256=sources)
        for k in ['epochs','batch_size','learning_rate','gradient_clip','checkpoint','loss','train_environments','train_samples']:
            if protocol[k] != old_protocol[k]:
                raise ValueError('Original training hyperparameter mismatch: '+k)
        fp = fingerprint(protocol)
        if (folder/'protocol.json').exists():
            if read(folder/'protocol.json') != protocol:
                raise ValueError('Frozen E2 training protocol changed.')
        else:
            write_json(folder/'protocol.json', protocol)
            for name in sources:
                dest = folder/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE/name, dest)
        if (folder/'complete.json').exists():
            record = read(folder/'complete.json')
            if record['fingerprint'] != fp:
                raise ValueError('Complete-model identity changed.')
            for name, digest in record['file_sha256'].items():
                if sha256(folder/name) != digest:
                    raise ValueError('Complete-model artifact changed: '+name)
            continue
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = Model().cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        shuffle = torch.Generator(device='cuda').manual_seed(seed)
        history = []; elapsed = 0.; used = torch.zeros(len(raw), dtype=torch.bool, device='cuda')
        if (folder/'resume.pt').exists():
            state = torch.load(folder/'resume.pt', map_location='cuda', weights_only=False)
            if state['fingerprint'] != fp:
                raise ValueError('Checkpoint identity mismatch.')
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            shuffle.set_state(state['shuffle'].cpu()); history = state['history']
            elapsed = state['seconds']; used = state['used']
        started = time.perf_counter(); seconds = elapsed
        for epoch in range(len(history)+1, 41):
            model.train(); order = torch.randperm(len(raw), generator=shuffle, device='cuda')
            total = 0.
            for begin in range(0, len(raw), 64):
                ix = order[begin:begin+64]; used[ix] = True
                pred = model(initial[ix], cond[ix])
                loss = ((pred-target[ix]).abs().square().mean((-1,-2))/target[ix].abs().square().mean((-1,-2)).clamp_min(1e-24)).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite E2 training loss.')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                total += float(loss.detach())*len(ix)
            seconds = elapsed + time.perf_counter() - started
            history.append(dict(epoch=epoch, loss=total/len(raw)))
            atomic_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                shuffle=shuffle.get_state(), used=used, history=history,
                seconds=seconds, fingerprint=fp), folder/'resume.pt')
            write_json(folder/'history.json', history)
            state = dict(status='training', seed=seed, epoch=epoch, total_epochs=40,
                loss=history[-1]['loss'], pid=os.getpid(), at=now())
            write_json(output/'progress.json', state); print(json.dumps(state), flush=True)
        if not bool(used.all()):
            raise ValueError('Not all training samples were used.')
        atomic_torch(model.state_dict(), folder/'weights.pt')
        model.eval()
        with torch.inference_mode():
            check = model(initial[:64], cond[:64]).cpu().numpy()
            replay = Model().cuda()
            replay.load_state_dict(torch.load(folder/'weights.pt', map_location='cuda', weights_only=True)); replay.eval()
            np.testing.assert_array_equal(check, replay(initial[:64], cond[:64]).cpu().numpy())
            del replay
        write_json(output/'progress.json', dict(status='fitting_training_statistics', seed=seed, at=now()))
        fit_statistics(folder, raw, initial, cond, target, model)
        verify_sources(sources); verify_targets(data, targets)
        files = ['weights.pt','cnn_residual_covariance.npy','cnn_frequency_covariance.npy','history.json']
        write_json(folder/'complete.json', dict(status='complete', fingerprint=fp, seed=seed,
            epochs=40, training_seconds=seconds, train_environments=3456, train_samples=58752,
            test_used=False, parameters=sum(p.numel() for p in model.parameters()),
            file_sha256={name:sha256(folder/name) for name in files}, at=now()))
        del model, optimizer; torch.cuda.empty_cache()
    write_json(output/'complete.json', dict(status='training_and_statistics_complete', seeds=seeds,
        evaluation_complete=False, at=now()))
    write_json(output/'progress.json', dict(status='training_and_statistics_complete', seeds=seeds, at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1,2])
    p.add_argument('--verify-seed0-statistics', action='store_true')
    a = p.parse_args()
    try:
        run(a.project.resolve(), a.output.resolve(), a.seeds, a.verify_seed0_statistics)
    except BaseException:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()), dict(at=now(), traceback=traceback.format_exc()))
        raise
