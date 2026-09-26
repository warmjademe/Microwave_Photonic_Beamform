"""只为训练成员生成额外复响应数字标签；不生成波形，不读取测试标签。"""
import argparse
import json
import platform
from pathlib import Path
import shutil
import sys
import time
import numpy as np
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from baseline_common.channel import make_environment, serialize_environment
from compact_dataset import sha256
from generate_native_dataset import write_json, now, hashes
from our_method_response_control.physics import training_target


def run(data, output):
    if platform.node() != 'qyb-HuaShuo':
        raise RuntimeError('仅华硕执行。')
    m = json.loads((data/'manifest.json').read_text())
    if hashes() != m['core_source_sha256']:
        raise ValueError('冻结核心改变。')
    rows = [r for r in m['environments'] if r['split'] == 'train']
    if len(rows) != 864 or [r['index'] for r in rows] != list(range(864)):
        raise ValueError('训练身份/顺序与预设不同。')
    output.mkdir(parents=True, exist_ok=False)
    target = np.lib.format.open_memmap(output/'train_response.npy', mode='w+',
        dtype=np.complex64, shape=(864*17, 64, 31))
    covariance = np.zeros((17, 64, 64), complex)
    files = ['our_method_response_control/physics.py', 'our_method_response_control/model.py',
        'our_method_response_control/prepare.py', 'our_method_response_control/test_model.py',
        'our_method_response_control/PROTOCOL.md', 'diagnostics/linear_centered.py',
        'diagnostics/carrier_reference.py']
    protocol = dict(created_at_utc=now(), role='training labels only', samples=864*17,
        train_environment_ids=[r['environment_id'] for r in rows],
        source_environment_plan_sha256=__import__('hashlib').sha256(
            json.dumps(m['environments'], sort_keys=True).encode()).hexdigest(),
        source_sha256={f: sha256(SOURCE/f) for f in files},
        core_source_sha256=hashes(), no_test_response_labels=True,
        target_definition='linearized effective per-route response, population mean frame-power normalization',
        covariance='uncentered theoretical zero-mean fading second moment, pooled over training environments and31tones per carrier',
        host=platform.node(), numpy=np.__version__)
    write_json(output/'protocol.json', protocol)
    for f in files:
        path = output/'source_snapshot'/f; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/f, path)
    started = time.perf_counter()
    for row in rows:
        environment = serialize_environment(make_environment(row['seed'], row['factors']))
        for ci, fc in enumerate(range(4, 21)):
            h = training_target(environment, fc)
            if h.shape != (64, 31) or not np.all(np.isfinite(h)):
                raise ValueError('非法响应目标。')
            target[row['index']*17+ci] = h
            covariance[ci] += h@h.conj().T
        if (row['index']+1) % 100 == 0:
            progress = dict(status='running', completed=row['index']+1, total=864,
                seconds=time.perf_counter()-started)
            write_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
    target.flush(); del target
    covariance /= (864*31)
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-14*np.max(abs(covariance)):
        raise ValueError('训练协方差非正半定。')
    np.save(output/'training_covariance.npy', covariance, allow_pickle=False)
    write_json(output/'complete.json', dict(status='complete', samples=864*17,
        label_sha256=sha256(output/'train_response.npy'), covariance_sha256=sha256(output/'training_covariance.npy'),
        seconds=time.perf_counter()-started, finished_at_utc=now()))
    write_json(output/'progress.json', dict(status='complete', completed=864, total=864,
        seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.data, a.output)
