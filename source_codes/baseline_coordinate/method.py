"""多尺度、带反馈预算的离散坐标搜索；不访问信道真值。"""
import numpy as np


# 每个元组分别是延时码和衰减码的步长，而不是物理单位。
STEP_CODES = ((32, 4), (8, 2), (1, 1))


def optimize(session, rng):
    """用已查询的最好设置作为起点，返回已实际评价的最好设置。"""
    n = int(session.cfg.n)
    levels = np.r_[np.full(n, session.cfg.delay_levels),
                   np.full(n, session.cfg.attenuation_levels)].astype(np.float64)
    if np.any(levels < 1) or not session.controls:
        raise ValueError("控制码级数必须为正，并且 session 必须包含共同的预探测。")
    visited = {np.asarray(session.project(u), np.float64).tobytes()
               for u in session.controls}
    while session.remaining > 0:
        queried_this_round = 0
        for scale_index, (delay_step, attenuation_step) in enumerate(STEP_CODES):
            # 把剩余预算分配给后续尺度，低预算下也保留细调机会。
            allowance = int(np.ceil(session.remaining / (len(STEP_CODES) - scale_index)))
            steps = np.r_[np.full(n, delay_step), np.full(n, attenuation_step)] / levels
            used = 0
            for coordinate in rng.permutation(2 * n):
                center = np.asarray(session.best_control, np.float64).copy()
                for sign in rng.permutation(np.array([-1.0, 1.0])):
                    if session.remaining <= 0 or used >= allowance:
                        break
                    trial = center.copy()
                    trial[coordinate] += sign * steps[coordinate]
                    trial = np.asarray(session.project(trial), np.float64)
                    key = trial.tobytes()
                    # 限幅后的零移动、已经测量的点都不再消耗反馈。
                    if key in visited or np.array_equal(trial, center):
                        continue
                    session.evaluate(trial)
                    visited.add(key)
                    used += 1
                    queried_this_round += 1
                if session.remaining <= 0 or used >= allowance:
                    break
        # 离散邻域已走完时允许提前结束，不能无限重复同一个控制。
        if queried_this_round == 0:
            break
    return np.asarray(session.best_control, np.float64).copy()
