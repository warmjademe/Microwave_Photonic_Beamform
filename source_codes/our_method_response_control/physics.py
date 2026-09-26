"""由公共探测反演复响应，并在全部128个合法器件控制量上计算。

这里使用可解析的小信号控制代理；最终优劣必须经独立混合接收器评分。
真实传播参数只允许training_target调用，估计与decode不接受该参数。
"""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE)); sys.path.insert(0, str(SOURCE/'diagnostics'))
from native_sim.config import NativeConfig
from native_sim.control_engine import (extraction_matrix, ACTIVE_TONES, BAND_OFFSETS,
    FIXED_LOSS_DB, APD_GAIN, DEMODULATOR_GAIN, RESPONSIVITY_A_PER_W,
    THERMAL_PSD_A2_HZ, ELECTRON_CHARGE_C, IONIZATION_RATIO, DARK_CURRENT_A)
from native_sim.laser import load_profile
from baseline_common.config import KB
from baseline_common.controls import probe_codebook
from baseline_common.channel import deserialize_environment, response
from linear_centered import transfers
from carrier_reference import bias_equilibrium

LEVELS = np.r_[np.full(64, 76), np.full(64, 24)]


@lru_cache(maxsize=17)
def design(carrier_ghz):
    cfg = NativeConfig(); fc = float(carrier_ghz)*1e9
    probes, _ = probe_codebook(cfg)
    frequency = fc+cfg.offsets_hz
    phase = np.exp(-2j*np.pi*np.arange(77)[:, None]/cfg.sample_rate_hz*frequency[None, :])
    transmission = 10**(-(FIXED_LOSS_DB+np.arange(25)*.5)/10)
    probe_code = np.rint(probes[:, :64]*76).astype(int)
    sensing = (transmission[0]*phase[probe_code]).transpose(2, 0, 1)  # 31×16×64
    gram = sensing @ sensing.conj().transpose(0, 2, 1)
    loading = .01*np.trace(gram, axis1=1, axis2=2).real/16
    inverse = np.linalg.solve(gram+loading[:, None, None]*np.eye(16), np.broadcast_to(np.eye(16), gram.shape))
    lift = sensing.conj().transpose(0, 2, 1) @ inverse
    models = transfers(fc, 2); profile = load_profile()
    # RF实通带/前端尺度500、APD/解调尺度60，与统一接收器保持一致。
    gain = np.stack([500*p['modulation_peak_current_a']*DEMODULATOR_GAIN*APD_GAIN
        *RESPONSIVITY_A_PER_W*m['response'][:, 1]*m['power_per_scaled_photon']
        for p, m in zip(profile, models)])
    extract = extraction_matrix(); noise_density = KB*cfg.temperature_k*cfg.df_hz
    antenna_payload_variance = noise_density*(abs(gain)**2 @ (abs(extract[2])**2).T)
    dc = np.array([bias_equilibrium(p)['optical_dc_w'] for p in profile])
    excess = IONIZATION_RATIO*APD_GAIN+(1-IONIZATION_RATIO)*(2-1/APD_GAIN)
    shot_factor = 2*ELECTRON_CHARGE_C*APD_GAIN**2*excess
    return dict(cfg=cfg, carrier_ghz=int(carrier_ghz), probes=probes, phase=phase, transmission=transmission,
        sensing=sensing, lift=lift, gain=gain, extract=extract, dc=dc,
        antenna_payload_variance=antenna_payload_variance, shot_factor=shot_factor,
        apd_row_norm=np.sum(abs(extract[2])**2, axis=-1))


def pilot_gain(x, pilots):
    x = np.asarray(x)
    iq = x[..., :1984].reshape(x.shape[:-1]+(16, 31, 2, 2))
    y = iq[..., 0]+1j*iq[..., 1]
    return np.mean(y*pilots.conj(), axis=-1)  # ...×16×31


def ridge_estimate(x, pilots):
    x = np.atleast_2d(x); result = np.empty((len(x), 64, 31), complex)
    y = pilot_gain(x, pilots)
    for fc in np.unique(x[:, 1984]):
        indices = np.flatnonzero(x[:, 1984] == fc)
        result[indices] = np.einsum('knp,bpk->bnk', design(int(fc))['lift'], y[indices])
    return result


def training_target(environment, carrier_ghz):
    """仅用于离线监督的响应代理，不生成测试控制。

    平均帧功率=3块×124样点×31个单位方差音调/512；不读取未来payload。
    矩形带限边界及非线性残差由最终混合接收器独立检验。
    """
    model = design(carrier_ghz); cfg = model['cfg']
    h = response(deserialize_environment(environment), cfg, carrier_ghz*1e9)
    norm = np.sqrt(3*(cfg.useful_samples+cfg.cp_samples)*31/cfg.iq_samples)
    unit_gain = model['gain'][:, ACTIVE_TONES*8+127]
    return unit_gain*h*np.sqrt(1e-3*10**(environment['power_dbm']/10))/norm


@lru_cache(maxsize=34)
def _shared_antenna_noise(carrier_ghz, pilot_bytes):
    """按全部255频格精确投影线性噪声，保留探测间相关性。"""
    model = design(carrier_ghz)
    pilots = np.frombuffer(pilot_bytes, np.complex128).reshape(31, 2)
    extract = model['extract']
    average = (extract[0]*pilots[:, 0].conj()[:, None]
               +extract[1]*pilots[:, 1].conj()[:, None])/2
    frequencies = carrier_ghz*1e9+BAND_OFFSETS*model['cfg'].df_hz
    delay = np.rint(model['probes'][:, :64]*76)/model['cfg'].sample_rate_hz
    b = (model['transmission'][0]*np.exp(-2j*np.pi*delay[:, :, None]*frequencies[None, None, :])
         *model['gain'][None, :, :])
    gram_by_frequency = np.einsum('pnb,qnb->bpq', b, b.conj())
    common = (KB*model['cfg'].temperature_k*model['cfg'].df_hz
              *np.einsum('kb,bpq->kpq', abs(average)**2, gram_by_frequency))
    ratio = np.sum(abs(average)**2, axis=-1)/np.sum(abs(extract[0])**2, axis=-1)
    return common, ratio


def measurement_noise(model, x, pilots):
    """16次测量共用天线噪声，APD抽样独立；包含两导频平均的相关项。"""
    common, ratio = _shared_antenna_noise(model['carrier_ghz'], np.asarray(pilots, np.complex128).tobytes())
    apd = x[2001:2497].reshape(16, 31)*ratio[None, :]
    return common+np.einsum('pk,pq->kpq', apd, np.eye(16))


def covariance_estimate(x, pilots, covariance):
    x = np.asarray(x); model = design(int(x[1984])); a = model['sensing']
    c = covariance[int(x[1984])-4]
    ca = c[None, :, :] @ a.conj().transpose(0, 2, 1)
    matrix = a @ ca + measurement_noise(model, x, pilots)
    # 只为数值正定加入相对1e-12的加载，不用测试结果选择。
    matrix += np.trace(matrix, axis1=1, axis2=2).real[:, None, None]/16*1e-12*np.eye(16)
    y = pilot_gain(x, pilots).T
    return np.einsum('knp,kp->nk', ca, np.linalg.solve(matrix, y[..., None])[..., 0])


def decode(response_estimate, carrier_ghz, initial_control, sweeps=2):
    """64路延时77档、衰减25档，固定2轮；没有真实信道或测量回调参数。"""
    model = design(carrier_ghz); h = np.asarray(response_estimate, complex)
    if h.shape != (64, 31) or not np.all(np.isfinite(h)):
        raise ValueError('复响应必须是64×31有限数组。')
    code = np.rint(np.clip(initial_control, 0, 1)*LEVELS).astype(int)
    trans = model['transmission'][code[64:]]
    phase = model['phase']; noise_table = model['antenna_payload_variance']
    branch = h*phase[code[:64]]*trans[:, None]
    signal = branch.sum(0); antenna_noise = np.sum(noise_table*trans[:, None]**2, axis=0)
    dc = float(trans @ model['dc'])
    def objective(s, n, power):
        psd = THERMAL_PSD_A2_HZ+model['shot_factor']*(RESPONSIVITY_A_PER_W*np.asarray(power)+DARK_CURRENT_A)
        noise = n+(DEMODULATOR_GAIN**2*model['cfg'].df_hz*psd)[..., None]*model['apd_row_norm']
        return -np.mean(noise/np.maximum(abs(s)**2+noise, np.finfo(float).tiny), axis=-1)
    initial = float(objective(signal, antenna_noise, dc)); score = initial
    for _ in range(sweeps):
        for index in range(128):
            n = index % 64
            if index < 64:
                replacement = h[n][None, :]*phase*trans[n]
                nn = np.broadcast_to(antenna_noise, replacement.shape)
                pp = np.full(77, dc)
            else:
                candidate_trans = model['transmission']
                replacement = h[n][None, :]*phase[code[n]][None, :]*candidate_trans[:, None]
                nn = antenna_noise[None, :]+noise_table[n][None, :]*(candidate_trans[:, None]**2-trans[n]**2)
                pp = dc+(candidate_trans-trans[n])*model['dc'][n]
            ss = signal[None, :]-branch[n][None, :]+replacement
            values = objective(ss, nn, pp); best = int(values.argmax())
            if values[best] > score+1e-14:
                code[index] = best; signal = ss[best]; antenna_noise = nn[best].copy()
                dc = float(pp[best]); branch[n] = replacement[best]
                if index >= 64:
                    trans[n] = model['transmission'][best]
                score = float(values[best])
    return code/LEVELS, dict(initial_proxy=initial, final_proxy=score,
        coordinate_sweeps=sweeps, extra_feedback=0,
        definition='mean known-response MMSE proxy including antenna and APD noise; not measured payload error')
