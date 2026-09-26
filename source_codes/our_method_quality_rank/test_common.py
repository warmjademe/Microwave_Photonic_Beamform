"""仅在华硕执行；核对向量化评分的物理含义和不同损失的监督边界。"""
import platform
from pathlib import Path
import sys
import numpy as np
import torch
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_quality_rank.common import candidate_metrics, catalog, loss_for, controls_from_output
from native_sim.config import NativeConfig
from native_sim.control_engine import NativeControlEngine
from native_sim.evaluation import evaluate_record
from baseline_common.config import qpsk, rng_for


def test_vectorized_metrics_match_independent_iq_fft():
    assert platform.node() == 'qyb-HuaShuo'
    cfg = NativeConfig(); rng = np.random.default_rng(71)
    pilots = qpsk(rng, (31, 2)); payload = qpsk(rng, (31,))
    engine = NativeControlEngine(cfg, 1e-8*(rng.normal(size=(64, 255))
        +1j*rng.normal(size=(64, 255))), np.full(64, .001), 12e9, pilots)
    controls = rng.uniform(size=(4, 128))
    result = candidate_metrics(engine, controls, payload, 81, 12, 410, 5, draws=3)
    for index, u in enumerate(controls):
        independent = [evaluate_record(engine, u, payload, rng_for(81, 12, 410, 5, j))
                       for j in range(3)]
        np.testing.assert_allclose(result['nmse'][index],
            np.mean([v['payload_nmse'] for v in independent]), rtol=1e-11, atol=1e-12)
        assert result['bit_errors'][index] == sum(v['bit_errors'] for v in independent)


def test_tied_labels_are_not_forced_to_one_arbitrary_control():
    assert torch.cuda.is_available()
    u = torch.as_tensor(catalog(NativeConfig())[0], dtype=torch.float32, device='cuda')
    target = torch.ones((2, 64), device='cuda')
    raw = torch.zeros((2, 64), device='cuda', requires_grad=True)
    loss = loss_for('candidate_soft_robust', raw, target, target, u)
    loss.backward()
    assert float(raw.grad.abs().max()) < 1e-7
    for kind in ['candidate_soft_robust', 'absolute_robust']:
        r = torch.zeros((2, 64 if kind.startswith('candidate') else 128), device='cuda')
        actual = controls_from_output(kind, r, u)
        levels = torch.tensor([76]*64+[24]*64, device='cuda')
        torch.testing.assert_close(actual*levels, (actual*levels).round())


if __name__ == '__main__':
    test_vectorized_metrics_match_independent_iq_fft()
    test_tied_labels_are_not_forced_to_one_arbitrary_control()
    print('{"status":"passed","tests":2,"host":"qyb-HuaShuo"}', flush=True)
