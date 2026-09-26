"""质量监督候选排序：仅使用16次公开探测，输出合法的128个器件控制数。

候选目录在采样前固定，包含原16个探测设置和48个额外几何设置。
离线可计算每个候选的真实接收误差；在线网络只能读公开测量X。
"""
from pathlib import Path
import sys
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'diagnostics'))
from baseline_common.controls import probe_codebook, geometric_control
from baseline_common.config import rng_for

METHODS = ['absolute_single', 'absolute_robust', 'candidate_ce_robust',
           'candidate_soft_single', 'candidate_soft_robust']
LEVELS = np.r_[np.full(64, 76), np.full(64, 24)]


def catalog(cfg):
    """按固定种子0遍历原代码本的方向网格，去重后补足64项。"""
    initial, angles = probe_codebook(cfg)
    controls, directions = list(initial), list(angles)
    seen = {tuple(np.rint(u * LEVELS).astype(int)) for u in controls}
    candidates = [(az, el) for az in np.linspace(-35, 35, 41)
                  for el in np.linspace(-15, 15, 21)]
    rng_for(0, 2401).shuffle(candidates)
    for az, el in candidates:
        u = geometric_control(cfg, az, el)
        key = tuple(np.rint(u * LEVELS).astype(int))
        if key in seen:
            continue
        controls.append(u); directions.append((az, el)); seen.add(key)
        if len(controls) == 64:
            break
    return np.asarray(controls), np.asarray(directions)


def candidate_metrics(engine, controls, payload, seed, carrier, stream, frame, draws=8):
    """矢量化完整两导频均衡及第三块评价；与evaluate_record逐项一致。

    每个APD抽样的底层随机数跨候选配对；DC不同导致噪声幅度不同。
    本函数只用于离线监督/独立评分，不作为在线网络输入。
    """
    codes = np.stack([engine.codes(u) for u in controls])
    trans = engine.transmissions[codes[:, 64:]]
    coeff = engine.current_factor * np.sum(
        engine.branch_band_w[None, :, :] * engine.phase[codes[:, :64]]
        * trans[:, :, None], axis=1)
    dc = trans @ engine.branch_dc_w
    _, variance = engine._noise(dc)
    signal = np.einsum('btk,ck->cbt', engine.extract, coeff)
    nmse, errors = [], []
    for draw in range(draws):
        rng = rng_for(seed, carrier, stream, frame, draw)
        z = (rng.standard_normal(255) + 1j*rng.standard_normal(255)) / np.sqrt(2)
        noise = np.einsum('btk,k->bt', engine.extract, z)
        symbols = signal + np.sqrt(variance)[:, None, None]*noise[None, :, :]
        gain = np.mean(symbols[:, :2, :]*engine.pilots.T[None, :, :].conj(), axis=1)
        noise_var = variance[:, None]*engine.symbol_noise_row_norm[2][None, :]
        w = gain.conj()/np.maximum(abs(gain)**2+noise_var, np.finfo(float).tiny)
        received = w*symbols[:, 2, :]
        nmse.append(np.mean(abs(received-payload[None, :])**2, axis=1)
                    / np.mean(abs(payload)**2))
        errors.append(np.sum((received.real >= 0) != (payload.real[None, :] >= 0), axis=1)
                      + np.sum((received.imag >= 0) != (payload.imag[None, :] >= 0), axis=1))
    return dict(nmse=np.mean(nmse, axis=0), bit_errors=np.sum(errors, axis=0),
                bits_tested=62*draws)


def make_model(kind):
    """保持原DNN编码器，只在候选方法中把末层改为64个未归一化分数。"""
    import torch
    from baseline_dnn.method import Model
    model = Model()
    if kind.startswith('candidate_'):
        model.readout.head = torch.nn.Sequential(
            torch.nn.Linear(320, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))
    return model


def loss_for(kind, raw, single, robust, controls):
    """固定温度0.05；接收效果接近的候选可以同时获得监督权重。"""
    import torch
    target = single if kind.endswith('_single') else robust
    if kind.startswith('absolute_'):
        return torch.mean((raw-controls[target.argmin(-1)])**2)
    if kind == 'candidate_ce_robust':
        return torch.nn.functional.cross_entropy(raw, target.argmin(-1))
    probability = torch.softmax(-(target-target.amin(-1, keepdim=True))/.05, dim=-1)
    return -(probability*torch.log_softmax(raw, dim=-1)).sum(-1).mean()


def controls_from_output(kind, raw, controls):
    import torch
    if kind.startswith('candidate_'):
        return controls[raw.argmax(-1)]
    levels = torch.as_tensor(LEVELS, device=raw.device, dtype=raw.dtype)
    return torch.floor(raw.clamp(0, 1)*levels+.5)/levels
