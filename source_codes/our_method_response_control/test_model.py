"""华硕上的最小机制验证；不读取新的测试环境或成绩。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import platform
import json
import numpy as np
import torch
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_response_control.physics import design, decode, LEVELS
from our_method_response_control.model import Model


def test_regularized_inverse():
    rng = np.random.default_rng(923)
    for fc in [4, 12, 20]:
        m = design(fc); a = m['sensing']
        y = rng.normal(size=(31, 16))+1j*rng.normal(size=(31, 16))
        proposed = np.einsum('knp,kp->kn', m['lift'], y)
        ah = a.conj().transpose(0, 2, 1)
        loading = .01*np.sum(abs(a)**2, axis=(1, 2))/16
        independent = np.linalg.solve(ah@a+loading[:, None, None]*np.eye(64), (ah@y[..., None]))[..., 0]
        np.testing.assert_allclose(proposed, independent, rtol=1e-10, atol=1e-10)


def test_decoder_scope_noise_and_invariance():
    rng = np.random.default_rng(924)
    h = (rng.normal(size=(64, 31))+1j*rng.normal(size=(64, 31)))*1e-4
    m = design(12); initial = m['probes'][3]
    u, meta = decode(h, 12, initial)
    rotated, _ = decode(h*np.exp(.713j), 12, initial)
    np.testing.assert_array_equal(np.rint(u*LEVELS), np.rint(rotated*LEVELS))
    assert meta['final_proxy'] >= meta['initial_proxy']
    np.testing.assert_allclose(u*LEVELS, np.rint(u*LEVELS), atol=1e-10)
    assert np.all((u >= 0) & (u <= 1))
    zero, z = decode(np.zeros_like(h), 12, initial)
    np.testing.assert_array_equal(zero, initial)
    assert z['initial_proxy'] == z['final_proxy'] == -1.
    # 仅一路有信号时，抑制其余路的噪声和光DC应有帮助，信号路保留0dB。
    one = np.zeros_like(h); one[0] = 1e-3
    selected, s = decode(one, 12, np.zeros(128))
    assert selected[64] == 0 and np.all(selected[65:] == 1)
    assert s['final_proxy'] > s['initial_proxy']


def test_complex_equivariance_and_gradients():
    assert torch.cuda.is_available()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); torch.set_num_threads(1)
    model = Model().cuda()
    # 激活完整残差支路，避免仅测试初始化时的恒等映射。
    with torch.no_grad():
        model.output.real.weight.normal_(std=.03); model.output.imag.weight.normal_(std=.03)
    initial = torch.complex(torch.randn(3, 64, 31, device='cuda'),
                            torch.randn(3, 64, 31, device='cuda'))*1e-3
    condition = torch.randn(3, 3, device='cuda')
    rotation = torch.exp(1j*torch.tensor([.731, -.84, 2.19], device='cuda'))[:, None, None]
    out = model(initial, condition)
    torch.testing.assert_close(model(initial*rotation, condition), out*rotation, atol=2e-8, rtol=3e-5)
    target = initial*.4
    loss = ((out-target).abs().square().mean((-1, -2))
            /target.abs().square().mean((-1, -2))).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert float(model.condition[-1].weight.grad.abs().sum()) > 0


if __name__ == '__main__':
    assert platform.node() == 'qyb-HuaShuo'
    test_regularized_inverse()
    test_decoder_scope_noise_and_invariance()
    test_complex_equivariance_and_gradients()
    print(json.dumps(dict(status='passed', tests=3, host=platform.node())), flush=True)
