"""Monte Carlo核对共同天线噪声与独立APD噪声的公开导频协方差。"""
from pathlib import Path
import sys
import platform
import json
import numpy as np
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from baseline_common.config import KB, qpsk, cn
from native_sim.control_engine import NativeControlEngine, BAND_OFFSETS
from our_method_response_control.physics import design, measurement_noise


def run():
    assert platform.node() == 'qyb-HuaShuo'
    rng = np.random.default_rng(2026092504); pilots = qpsk(rng, (31, 2)); records = []
    for fc in [4, 20]:
        model = design(fc); cfg = model['cfg']; tone = 15
        engine = NativeControlEngine(cfg, np.zeros((64, 255), complex), model['dc'], fc*1e9, pilots)
        state = engine.state(model['probes'][0]); _, var = engine._noise(state['optical_dc_w'])
        x = np.zeros(2513); x[1984] = fc
        x[2001:2497] = np.tile(var*engine.symbol_noise_row_norm[0], 16)
        expected = measurement_noise(model, x, pilots)[tone]
        delays = np.rint(model['probes'][:, :64]*76)/cfg.sample_rate_hz
        phase = np.exp(-2j*np.pi*delays[:, :, None]*(fc*1e9+BAND_OFFSETS*cfg.df_hz))
        average = (engine.extract[0, tone]*pilots[tone, 0].conj()
                  +engine.extract[1, tone]*pilots[tone, 1].conj())/2
        projection = (model['transmission'][0]*phase*model['gain'][None, :, :]*average[None, None, :]).reshape(16, -1)
        samples = []
        for _ in range(256):
            # 每次独立噪声实现中，16个探测共享同一64路天线噪声。
            antenna = np.sqrt(KB*cfg.temperature_k*cfg.df_hz)*cn(rng, (32, 64*255))
            shared = antenna@projection.T
            apd = np.sqrt(var)*cn(rng, (32, 16, 255))
            samples.append(shared+np.einsum('bpk,k->bp', apd, average))
        samples = np.concatenate(samples)
        actual = samples.T@samples.conj()/len(samples)
        error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        pilot_actual = samples[:1024].T@samples[:1024].conj()/1024
        pilot_error = float(np.linalg.norm(pilot_actual-expected)/np.linalg.norm(expected))
        # proper复高斯样本协方差的相对Frobenius均方根采样误差。
        # 1024次的预期精度未必足够支持10%门限；增加至8192次，门限不变。
        expected_rms = float(np.trace(expected).real/np.linalg.norm(expected)/np.sqrt(len(samples)))
        record = dict(carrier_ghz=fc, draws=len(samples), relative_frobenius_error=error,
            first_1024_relative_error=pilot_error, expected_sampling_rms=expected_rms,
            first_1024_expected_sampling_rms=expected_rms*np.sqrt(8),
            relative_trace_error=float(abs(np.trace(actual)-np.trace(expected))/np.trace(expected).real))
        records.append(record)
        print(json.dumps(record), flush=True)
        assert expected_rms < .05, (fc, expected_rms)
        assert error < .10, (fc, error)
        assert np.linalg.eigvalsh(expected).min() > 0
    print(json.dumps(dict(status='passed', records=records, threshold=.10)), flush=True)


if __name__ == '__main__':
    run()
