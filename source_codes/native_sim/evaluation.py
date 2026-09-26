"""只用公开导频设置均衡器，再在第三块独立QPSK数据上评价。"""
import numpy as np


def evaluate_record(engine, control, payload, rng):
    """返回同一次APD噪声抽样的IQ和指标；不拿payload拟合相位/增益/时序。"""
    target = np.asarray(payload, complex)
    if target.shape != (31,) or not np.all(np.isfinite(target)):
        raise ValueError('独立数据块必须是31个有限QPSK符号。')
    iq = engine.iq(control, rng)
    tones = np.arange(-15, 16) % 64
    symbols = np.stack([np.fft.fft(iq[start:start+64])[tones]/64
                        for start in (60, 184, 308)], axis=-1)
    # 两个已知块的LS均值；依据已标定APD方差作固定MMSE正则化。
    gain = np.mean(symbols[:, :2]*np.conj(engine.pilots), axis=-1)
    state = engine.state(control)
    _, variance = engine._noise(state['optical_dc_w'])
    noise_var = variance*engine.symbol_noise_row_norm[2]
    weight = np.conj(gain)/np.maximum(abs(gain)**2+noise_var, np.finfo(float).tiny)
    received = weight*symbols[:, 2]
    nmse = float(np.mean(abs(received-target)**2)/np.mean(abs(target)**2))
    bit_errors = int(np.sum((received.real >= 0) != (target.real >= 0))
                     + np.sum((received.imag >= 0) != (target.imag >= 0)))
    return dict(evm_percent=float(100*np.sqrt(nmse)), bit_errors=bit_errors,
                bits_tested=62, ber=bit_errors/62, payload_nmse=nmse,
                snr_db=float(-10*np.log10(max(nmse, 1e-30))),
                snr_db_definition='-10*log10(payload NMSE); includes distortion and estimation error',
                iq_a=iq, received_qpsk=received, raw_payload_a=symbols[:, 2],
                pilot_estimated_gain_a=gain, payload_noise_var_a2=noise_var,
                equalizer_uses_payload=False)
