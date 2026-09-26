"""只用已有训练环境检查估计、GPU、128维控制、独立接收评分的接口。"""
import argparse
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from our_method_response_control.train import precision, verify_targets
from our_method_response_control.physics import design, ridge_estimate, decode, LEVELS
from our_method_response_control.model import Model, conditions
from our_method_response_control.evaluate import choose_controls
from our_method_quality_rank.generate import frame_engine
from our_method_quality_rank.common import candidate_metrics
from native_sim.control_engine import THERMAL_PSD_A2_HZ, RESPONSIVITY_A_PER_W, DARK_CURRENT_A, DEMODULATOR_GAIN
from compact_dataset import sha256
from generate_native_dataset import write_json, now, hashes
import torch


def scalar_reference(h, fc, initial):
    """每个候选都从64路完整重算，核对增量更新解码器；仅用于少量前置检查。"""
    model = design(fc)
    code = np.rint(initial*LEVELS).astype(int)
    def objective(candidate):
        trans = model['transmission'][candidate[64:]]
        signal = np.sum(h*trans[:, None]*model['phase'][candidate[:64]], axis=0)
        optical_dc = float(trans@model['dc'])
        noise = np.sum(model['antenna_payload_variance']*trans[:, None]**2, axis=0)
        psd = THERMAL_PSD_A2_HZ+model['shot_factor']*(RESPONSIVITY_A_PER_W*optical_dc+DARK_CURRENT_A)
        noise += DEMODULATOR_GAIN**2*model['cfg'].df_hz*psd*model['apd_row_norm']
        return float(-np.mean(noise/(abs(signal)**2+noise)))
    score = objective(code)
    for _ in range(2):
        for index in range(128):
            scores = []
            for level in range(int(LEVELS[index])+1):
                proposal = code.copy(); proposal[index] = level
                scores.append(objective(proposal))
            best = int(np.argmax(scores))
            if scores[best] > score+1e-14:
                code[index] = best; score = scores[best]
    return code, score


def run(project, data, targets, output):
    if platform.node() != 'qyb-HuaShuo' or not torch.cuda.is_available():
        raise RuntimeError('仅华硕GPU执行。')
    verify_targets(data, targets); precision()
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(p.relative_to(SOURCE)): sha256(p)
        for p in (SOURCE/'our_method_response_control').glob('*') if p.suffix in ['.py', '.md']}
    write_json(output/'protocol.json', dict(created_at_utc=now(), source_sha256=source_hashes,
        core_source_sha256=hashes(), role='existing training environment only; no new test reads'))
    old = project/'dataset_simulation/diagnostics/20260925_quality_rank_hybrid_preflight'
    paths = list(old.glob('**/data.npz'))
    if len(paths) != 1:
        raise ValueError('已有训练前置检查应只有一个环境。')
    with np.load(paths[0]) as f:
        x = f['X']
    environment = json.loads((paths[0].parent/'environment.json').read_text())
    with np.load(old/'public.npz') as f:
        pilots, probes = f['pilot_qpsk'], f['catalog_controls'][:16]
    covariance = np.load(targets/'training_covariance.npy')
    torch.manual_seed(0); model = Model().cuda().eval()
    started = time.perf_counter(); records = []
    for fc in [4, 12, 20]:
        raw = x[fc-4:fc-3]
        initial = ridge_estimate(raw, pilots).astype(np.complex64)
        with torch.inference_mode():
            estimated = model(torch.from_numpy(initial).cuda(),
                torch.from_numpy(conditions(raw, initial, pilots)).cuda()).cpu().numpy()[0]
        np.testing.assert_array_equal(estimated, initial[0])  # 零初始化残差。
        controls, responses, seconds, proxy = choose_controls(raw[0], pilots, probes, covariance, estimated)
        codes = np.rint(controls*LEVELS).astype(int)
        if not np.all(np.isfinite(responses)) or np.any(codes < 0) or np.any(codes > LEVELS):
            raise ValueError('非法响应或控制码。')
        engine, payload = frame_engine(environment, fc, pilots, 5)
        result = candidate_metrics(engine, controls, payload, environment['seed'], fc, 105, 5, draws=8)
        if not np.all(np.isfinite(result['nmse'])) or np.any(result['nmse'] < 0):
            raise ValueError('接收评分无效。')
        if fc == 12:
            base = probes[int(raw[0, 1985:2001].argmax())]
            independent_codes, independent_proxy = scalar_reference(responses[1], fc, base)
            np.testing.assert_array_equal(independent_codes, codes[1])
            np.testing.assert_allclose(independent_proxy, proxy[1, 1], atol=1e-12, rtol=1e-12)
        records.append(dict(carrier_ghz=fc, legal_codes=int(codes.size),
            all_proxy_nondecreasing=bool(np.all(proxy[:3, 1] >= proxy[:3, 0]-1e-12)),
            scores_finite=True, control_cpu_seconds=seconds.tolist(),
            scalar_full_recompute_checked=fc == 12))
    result = dict(status='passed', at=now(), host=platform.node(), records=records,
        old_training_environment_seed=environment['seed'], source_sha256=source_hashes,
        seconds=time.perf_counter()-started, new_test_read=False,
        interpretation='implementation preflight, untrained CNN; no method effectiveness result')
    write_json(output/'summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'data', 'targets', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.project, a.data, a.targets, a.output)
