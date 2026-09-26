"""正式采样前在一个已有训练环境上检验17频率全流程及五项GPU梯度。"""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_quality_rank.common import METHODS, catalog, make_model, loss_for, controls_from_output
from our_method_quality_rank.generate import one_environment
from our_method_quality_rank.test_common import (
    test_vectorized_metrics_match_independent_iq_fft, test_tied_labels_are_not_forced_to_one_arbitrary_control)
from our_method_quality_rank.train import TrainingArray
from deep_common.preprocessing import fit, apply
from native_sim.config import NativeConfig
from native_sim.data import NativeDataset
from compact_dataset import CompactDataset, sha256
from generate_native_dataset import write_json, now
from baseline_common.config import rng_for, qpsk


def run(project, output):
    if platform.node() != 'qyb-HuaShuo' or not torch.cuda.is_available():
        raise RuntimeError('需华硕GPU。')
    output.mkdir(exist_ok=False, parents=True)
    test_vectorized_metrics_match_independent_iq_fft()
    test_tied_labels_are_not_forced_to_one_arbitrary_control()
    d = CompactDataset(project/'dataset_simulation/dataset_train')
    native = NativeDataset(d.source_root)
    row = native.environments('train')[0]
    controls, angles = catalog(NativeConfig())
    np.savez(output/'public.npz', catalog_controls=controls, catalog_angles_deg=angles,
        pilot_qpsk=qpsk(rng_for(2026092503, 2), (31, 2)))
    started = time.perf_counter()
    record = one_environment(str(output), row)
    with np.load(output/row['path']/'data.npz') as f:
        x = f['X']; single = f['single_nmse']; robust = f['robust_nmse']
    if (x.shape != (17, 2513) or single.shape != (17, 64)
        or robust.shape != (17, 64) or not np.all(np.isfinite(robust))
        or not np.all(np.isfinite(single)) or np.any(robust < 0) or np.any(single < 0)):
        raise ValueError('完整帧生成输出无效。')
    stats = fit(TrainingArray(x)); z = torch.from_numpy(apply(x, stats)).cuda()
    s = torch.from_numpy(single).cuda(); r = torch.from_numpy(robust).cuda()
    u = torch.as_tensor(controls, device='cuda', dtype=torch.float32)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    results = {}
    for method in METHODS:
        torch.manual_seed(0); model = make_model(method).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss = loss_for(method, model(z), s, r, u)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None):
            raise ValueError('非有限损失或梯度。')
        optimizer.step()
        with torch.no_grad():
            control = controls_from_output(method, model(z), u)
        levels = torch.tensor([76]*64+[24]*64, device='cuda')
        torch.testing.assert_close(control*levels, (control*levels).round())
        results[method] = dict(first_loss=float(loss.detach()), parameters=sum(p.numel() for p in model.parameters()))
    result = dict(status='passed', host=platform.node(), at=now(),
        existing_training_environment=row['environment_id'], full_carriers=17,
        generation_seconds=record['seconds'], all_seconds=time.perf_counter()-started,
        models=results, vectorized_evaluation_matches_iq_fft=True,
        tied_quality_loss_has_no_arbitrary_preferred_candidate=True,
        source_sha256=sha256(__file__), failure_before_test='pytest unavailable; direct function runner used, no package installation')
    write_json(output/'summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.project, a.output)
