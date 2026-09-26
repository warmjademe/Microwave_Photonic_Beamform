"""由共同初始扫描估计粗方向，使用该方向的几何时延与等衰减。"""
import numpy as np


def optimize(session, rng):
    del rng
    count = session.cfg.probes
    # 只使用公共扫描得到的粗方向，不读取仿真真实方向。
    index = int(np.argmax(session.scores[:count]))
    return session.controls[index].copy()
