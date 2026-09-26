"""只用第一个训练环境核对指标、反馈预算、教师和原生IQ评分一致性。"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from baseline_common.feedback import FeedbackSession
from native_sim.control_engine import optimize_teacher
from native_sim.evaluation import evaluate_record
from our_method_quality_rank.common import candidate_metrics


def run(data, output):
    require_host(); manifest = check_data(data)
    row = next(r for r in manifest['environments'] if r['split']=='train')
    env = json.loads((data/row['path']/'environment.json').read_text())
    with np.load(data/row['path']/'data.npz') as f:
        x = f['X']
    public = public_data(data); evidence = []
    for fc in [4, 12, 20]:
        start = time.perf_counter(); obs = observation(x[fc-4], public)
        original, _ = frame_engine(env, fc, public['pilot_qpsk'], 0)
        chosen, counts = [], []
        for method in ONLINE_CLASSIC:
            def measure(u, call):
                return original.measure_detailed(u, rng_for(row['seed'], fc, 620, call))['score']
            session = FeedbackSession(original.cfg, obs, measure, 64)
            u = importlib.import_module('baseline_'+method+'.method').optimize(
                session, rng_for(0, row['seed'], fc, 630))
            assert session.calls <= 64
            assert any(np.array_equal(session.project(u), v) for v in session.controls)
            chosen.append(u); counts.append(session.calls)
        teacher, info = optimize_teacher(original, public['probe_controls'],
            rng_for(0, row['seed'], fc, 610), starts=2, sweeps=1)
        chosen.extend([teacher, chosen[0]])
        engine, payload = frame_engine(env, fc, public['pilot_qpsk'], 5)
        clean = clean_frame_engine(env, fc, public['pilot_qpsk'], payload)
        metrics, codes = reception_metrics(engine, clean, chosen, payload, row['seed'], fc)
        ref = candidate_metrics(engine, chosen, payload, row['seed'], fc, 105, 5, draws=8)
        assert np.array_equal(metrics[:, 0], ref['bit_errors'])
        assert np.allclose(metrics[:, 2], ref['nmse'], rtol=1e-12, atol=1e-13)
        assert np.array_equal(metrics[0], metrics[-1])
        assert np.all((metrics[:, 3] <= metrics[:, 0]) & (metrics[:, 0] <= 2*metrics[:, 3]))
        assert np.all((codes >= 0) & (codes <= LEVELS))
        # 独立慢路径：真实512点IQ，再FFT，再导频均衡。
        slow = [evaluate_record(engine, chosen[0], payload,
            rng_for(row['seed'], fc, 105, 5, draw)) for draw in range(8)]
        assert sum(v['bit_errors'] for v in slow) == metrics[0, 0]
        assert np.isclose(np.mean([v['payload_nmse'] for v in slow]), metrics[0, 2], rtol=1e-12)
        clean_iq = clean.iq(chosen[0]); noisy_mean_iq = engine.iq(chosen[0])
        _, variance = engine._noise(engine.state(chosen[0])['optical_dc_w'])
        assert np.isclose(np.mean(abs(clean_iq)**2), metrics[0, 7], rtol=1e-12)
        assert np.isclose(np.mean(abs(noisy_mean_iq-clean_iq)**2)+255*variance, metrics[0, 8], rtol=1e-12)
        evidence.append(dict(carrier_ghz=fc, feedback_calls=counts,
            candidate_metric_match=True, slow_IQ_metric_match=True, parseval_power_match=True,
            teacher_objective_nondecreasing=info['objective']>=info['initial_best_objective'],
            seconds=time.perf_counter()-start))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('前置检查结果不覆盖。')
    write_json(output, dict(status='passed', training_environment_id=row['environment_id'],
        evidence=evidence, data_sha256=sha256(data/'manifest.json'), at=now(),
        sources=source_record(['study_full_baselines/common.py','study_full_baselines/preflight.py',
                               'study_full_baselines/prepare_labels.py'])))
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.data, a.output)
