"""双边同时扰动，仅使用计费的标量反馈。"""
import numpy as np


PERTURBATION = 0.08
LEARNING_RATE = 0.20
PERTURBATION_DECAY = 0.101
LEARNING_RATE_DECAY = 0.602


def optimize(session, rng):
    """SPSA 的量化/边界适配版，不保证连续原算法的收敛性质。"""
    n = int(session.cfg.n)
    levels = np.r_[np.full(n, session.cfg.delay_levels),
                   np.full(n, session.cfg.attenuation_levels)].astype(np.float64)
    if np.any(levels < 2) or not session.controls:
        raise ValueError("双边离散扰动要求每维至少三个码，且会话已有预探测。")
    current = np.asarray(session.best_control, np.float64).copy()
    iteration = 0
    while session.remaining >= 2:
        ck = PERTURBATION / (iteration + 1) ** PERTURBATION_DECAY
        # 以整数码构建严格对称的正负扰动；至少移动一个码。
        step_codes = np.clip(np.rint(ck * levels), 1, np.floor(levels / 2))
        center_codes = np.clip(np.rint(session.project(current) * levels),
                               step_codes, levels - step_codes)
        center = center_codes / levels
        delta = rng.choice(np.array([-1.0, 1.0]), size=2 * n)
        steps = step_codes / levels
        plus = np.asarray(session.project(center + steps * delta), np.float64)
        minus = np.asarray(session.project(center - steps * delta), np.float64)
        # 不把有偏的一侧裁剪差分冒充对称差分。
        score_plus = float(session.evaluate(plus))
        score_minus = float(session.evaluate(minus))
        gradient = (score_plus - score_minus) / (2 * steps * delta)
        # 限制更新幅度，降低目标分数单位/尺度对默认步长的影响。
        gradient /= max(1.0, float(np.max(np.abs(gradient))))
        ak = LEARNING_RATE / (iteration + 1) ** LEARNING_RATE_DECAY
        current = np.asarray(session.project(center + ak * gradient), np.float64)
        # 更新点不是免费观测；已测点/零更新无需额外查询。
        already_measured = any(np.array_equal(current, old) for old in session.controls)
        if session.remaining > 0 and not already_measured:
            session.evaluate(current)
        iteration += 1
    if session.remaining == 1:
        # 单次余额不能形成双边梯度，用一次合法单码邻域试探收尾。
        trial = np.asarray(session.best_control, np.float64).copy()
        coordinate = int(rng.integers(2 * n))
        sign = 1.0 if trial[coordinate] <= 0.5 else -1.0
        trial[coordinate] += sign / levels[coordinate]
        session.evaluate(session.project(trial))
    return np.asarray(session.best_control, np.float64).copy()
