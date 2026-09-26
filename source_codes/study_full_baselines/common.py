"""统一公开观测、完整接收指标和原子结果保存，不更改已冻结物理核。"""
from pathlib import Path
import json
import os
import platform
import sys
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'diagnostics'))
from compact_dataset import sha256, unpack_observation
from native_sim.config import NativeConfig
from native_sim.control_engine import NativeControlEngine
from native_sim.waveforms import transmit_coefficients, channel_at_offsets
from baseline_common.config import rng_for
from our_method_quality_rank.generate import frame_engine
from our_method_quality_rank.train import check_data, load_split
from generate_native_dataset import write_json, now
from hybrid_centered import hybrid_from_coefficients

LEVELS = np.r_[np.full(64, 76), np.full(64, 24)]
ONLINE_CLASSIC = ['ttd_das', 'codebook', 'coordinate', 'spsa', 'done', 'de']
DEEP_METHODS = ['dnn', 'cnn', 'rescnn', 'transformer', 'complex_cnn', 'jct']
QUALITY_METRICS = ['bit_errors', 'bits_tested', 'payload_nmse',
                   'symbol_errors', 'symbols_tested', 'block_errors', 'blocks_tested',
                   'clean_output_power_a2', 'noise_output_power_a2', 'optical_dc_w']


def require_host():
    if platform.node() != 'qyb-HuaShuo':
        raise RuntimeError('本项目数值计算仅允许在华硕运行。')


def public_data(data):
    with np.load(Path(data) / 'public.npz') as f:
        return {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls']}


def observation(x, public):
    return unpack_observation(x, public, NativeConfig())


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def clean_frame_engine(environment, carrier, pilots, payload):
    """仅供评分：相同波形与传播的无天线噪声反事实，仍含器件非线性。"""
    cfg = NativeConfig()
    coeff = transmit_coefficients(pilots, payload, cfg)
    channel = channel_at_offsets(environment, cfg, carrier * 1e9)
    power = 1e-3 * 10 ** (float(environment['power_dbm']) / 10)
    band, dc, _ = hybrid_from_coefficients(
        500 * np.sqrt(power) * channel * coeff[None, :], carrier * 1e9, cfg)
    return NativeControlEngine(cfg, band, dc, carrier * 1e9, pilots)


def reception_metrics(engine, clean_engine, controls, payload, seed, carrier, draws=8):
    """所有控制共用逐抽样底噪，评价器不把未知payload用于均衡器拟合。"""
    controls = np.asarray(controls, float)
    codes = np.stack([engine.codes(u) for u in controls])
    trans = engine.transmissions[codes[:, 64:]]
    def coefficients(current):
        return current.current_factor * np.sum(current.branch_band_w[None, :, :]
            * current.phase[codes[:, :64]] * trans[:, :, None], axis=1)
    coeff, clean = coefficients(engine), coefficients(clean_engine)
    dc = trans @ engine.branch_dc_w
    _, variance = engine._noise(dc)
    signal = np.einsum('btk,ck->cbt', engine.extract, coeff)
    errors = np.zeros(len(controls), np.int64)
    symbols_wrong = errors.copy(); blocks_wrong = errors.copy()
    nmse = np.zeros(len(controls))
    for draw in range(draws):
        rng = rng_for(seed, carrier, 105, 5, draw)
        z = (rng.standard_normal(255) + 1j * rng.standard_normal(255)) / np.sqrt(2)
        noise = np.einsum('btk,k->bt', engine.extract, z)
        received_blocks = signal + np.sqrt(variance)[:, None, None] * noise[None, :, :]
        gain = np.mean(received_blocks[:, :2] * engine.pilots.T[None].conj(), axis=1)
        noise_var = variance[:, None] * engine.symbol_noise_row_norm[2]
        weight = gain.conj() / np.maximum(abs(gain)**2 + noise_var, np.finfo(float).tiny)
        received = weight * received_blocks[:, 2]
        wrong_i = (received.real >= 0) != (payload.real[None] >= 0)
        wrong_q = (received.imag >= 0) != (payload.imag[None] >= 0)
        error_count = wrong_i.sum(1) + wrong_q.sum(1)
        errors += error_count
        symbols_wrong += (wrong_i | wrong_q).sum(1)
        blocks_wrong += error_count > 0
        nmse += np.mean(abs(received - payload[None])**2, axis=1) / np.mean(abs(payload)**2)
    # Parseval: 512点IQ的平均电流功率等于255个Fourier系数的模平方和。
    signal_power = np.sum(abs(clean)**2, axis=1)
    noise_power = np.sum(abs(coeff - clean)**2, axis=1) + 255 * variance
    values = np.column_stack([errors, np.full(len(controls), 62 * draws), nmse / draws,
        symbols_wrong, np.full(len(controls), 31 * draws), blocks_wrong,
        np.full(len(controls), draws), signal_power, noise_power, dc])
    if not np.isfinite(values).all() or np.any(values[:, 8] <= 0):
        raise ValueError('接收质量指标非有限或噪声功率非正。')
    return values, codes


def source_record(names):
    return {name: sha256(SOURCE / name) for name in sorted(set(names))}


def verify_sources(record):
    for name, expected in record.items():
        if sha256(SOURCE / name) != expected:
            raise ValueError('运行所用源码已经改变：' + name)
