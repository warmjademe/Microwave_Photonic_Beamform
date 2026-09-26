"""在剩余反馈预算内扫描更密方向网格，所有候选都是合法几何时延。"""
import numpy as np
from baseline_common.controls import geometric_control


def optimize(session, rng):
    candidates = [(az, el) for az in np.linspace(-35, 35, 41)
                  for el in np.linspace(-15, 15, 21)]
    # 固定种子随机遍历，避免小预算只覆盖一侧方向；不按真方向排序。
    rng.shuffle(candidates)
    seen = {tuple(u) for u in session.controls}
    for az, el in candidates:
        if session.remaining <= 0:
            break
        u = geometric_control(session.cfg, az, el)
        if tuple(u) in seen:
            continue
        seen.add(tuple(u))
        session.evaluate(u)
    return session.best_control
