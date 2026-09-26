"""期望信号的统计多径信道；不依赖光子探测方式。"""
from itertools import product
import numpy as np
from .config import C0, rng_for, cn
from .controls import direction


FACTOR_LEVELS = {'rays': [4, 8, 16, 24], 'max_delay_ns': [50, 100, 200],
                 'angular_std_deg': [1, 3, 6], 'power_bin': list(range(6))}
JOINT_STRATA = 216


def balanced_factors(count, seed):
    """逐因子均衡后独立打乱；不把所有条件组合穷举。"""
    rng = rng_for(seed, 120)
    choices = {'rays': [4, 8, 16, 24], 'max_delay_ns': [50, 100, 200],
               'angular_std_deg': [1, 3, 6], 'power_bin': list(range(6))}
    out = {}
    for name, values in choices.items():
        a = np.resize(values, count)
        rng.shuffle(a)
        out[name] = a
    return [{name: float(a[i]) for name, a in out.items()} for i in range(count)]


def joint_stratified_factors(count, seed):
    """每个四因素联合格重复同样次数，再打乱；不截断不完整的216格。"""
    if int(count) != count or count <= 0 or count % JOINT_STRATA:
        raise ValueError('joint_stratified 的每个划分数量必须为正的 216 倍数。')
    count = int(count)
    grid = list(product(*FACTOR_LEVELS.values()))
    rows = [dict(zip(FACTOR_LEVELS, map(float, values)))
            for _ in range(count // JOINT_STRATA) for values in grid]
    order = rng_for(seed, 121).permutation(count)
    return [rows[int(i)] for i in order]


def sampling_plan(count, seed, method='marginal_balanced'):
    """只规划分层因素并给出可审计元数据，不生成信道或接收数据。"""
    if int(count) != count or count <= 0:
        raise ValueError('环境数量必须为正整数。')
    if method == 'marginal_balanced':
        factors = balanced_factors(int(count), seed)
        stream, repetitions = 120, None
    elif method == 'joint_stratified':
        factors = joint_stratified_factors(count, seed)
        stream, repetitions = 121, int(count) // JOINT_STRATA
    else:
        raise ValueError('未知采样方法：' + str(method))
    return factors, dict(method=method, environment_count=int(count),
                         factor_levels={k: list(v) for k, v in FACTOR_LEVELS.items()},
                         joint_strata=JOINT_STRATA, repeats_per_joint_stratum=repetitions,
                         exact_joint_balance=(method == 'joint_stratified'),
                         ordering_seed=int(seed), ordering_seed_sequence=[int(seed), stream])


def make_environment(seed, factors):
    rng = rng_for(seed, 1)
    count = int(factors['rays'])
    limit = float(factors['max_delay_ns'])
    spread = float(factors['angular_std_deg'])
    az, el = rng.uniform(-35, 35), rng.uniform(-15, 15)
    delays_ns = np.r_[0., np.sort(rng.uniform(0, limit, count-1))]
    pdp = np.exp(-delays_ns/(limit/3))
    pdp /= pdp.sum()
    alpha = cn(rng, count)*np.sqrt(pdp)
    angles = np.column_stack([az+rng.normal(0, spread, count),
                              el+rng.normal(0, spread/2, count)])
    # 相位以 4 GHz 为参考；同一环境的 17 档共享这些路径。
    avg = float(np.dot(pdp, delays_ns))
    realized = abs(alpha)**2
    realized /= realized.sum()
    rmean = float(np.dot(realized, delays_ns))
    return dict(seed=int(seed), factors=factors, mean_az_deg=float(az), mean_el_deg=float(el),
                power_dbm=float(-105+5*int(factors['power_bin'])+rng.uniform(0, 5)),
                delays_s=delays_ns*1e-9, angles_deg=angles, pdp=pdp, alpha=alpha,
                actual_max_delay_ns=float(delays_ns.max()), pdp_rms_delay_ns=float(np.sqrt(np.dot(pdp, (delays_ns-avg)**2))),
                realized_rms_delay_ns=float(np.sqrt(np.dot(realized, (delays_ns-rmean)**2))),
                reference_hz=4e9, doppler_hz=0., shadow_std_db=0., interference=False)


def response(environment, cfg, carrier_hz, return_paths=False):
    e = environment
    angles = np.asarray(e['angles_deg'])
    geo = cfg.positions@direction(angles[:, 0], angles[:, 1]).T/C0
    freq = carrier_hz+cfg.offsets_hz
    phase = (2*np.pi*geo[:, :, None]*freq[None, None, :]
             -2*np.pi*np.asarray(e['delays_s'])[None, :, None]*(freq-e['reference_hz'])[None, None, :])
    paths = np.asarray(e['alpha'])[None, :, None]*np.exp(1j*phase)
    h = paths.sum(axis=1)
    return (h, paths) if return_paths else h


def serialize_environment(e):
    out = {}
    for key, value in e.items():
        if isinstance(value, np.ndarray):
            out[key] = ({'real': value.real.tolist(), 'imag': value.imag.tolist()}
                        if np.iscomplexobj(value) else value.tolist())
        else:
            out[key] = value
    return out


def deserialize_environment(e):
    out = dict(e)
    for key in ('delays_s', 'angles_deg', 'pdp'):
        out[key] = np.asarray(out[key], float)
    value = out['alpha']
    out['alpha'] = np.asarray(value['real'])+1j*np.asarray(value['imag'])
    return out
