"""DONE 思路适配：随机傅里叶特征、正则在线回归、有界候选搜索。"""
import numpy as np


FEATURES = 64
RIDGE = 0.01
FREQUENCY_SCALE = 4.0
GLOBAL_CANDIDATES = 24
LOCAL_CANDIDATES = 24
LOCAL_STARTS = 4
LOCAL_STEPS = 12
LOCAL_RATE = 0.08
EXPLORATION_PROBABILITY = 0.10


class _FourierSurrogate:
    """固定随机特征，用递推最小二乘精确累计同一正则回归目标。"""
    def __init__(self, dimension, rng, offset):
        self.frequencies = rng.normal(0, FREQUENCY_SCALE / np.sqrt(dimension),
                                      (FEATURES, dimension))
        self.phases = rng.uniform(0, 2 * np.pi, FEATURES)
        self.factor = np.sqrt(2.0 / FEATURES)
        self.coefficients = np.zeros(FEATURES + 1)
        self.inverse = np.eye(FEATURES + 1) / RIDGE
        self.offset = float(offset)

    def features(self, points):
        x = np.atleast_2d(np.asarray(points, np.float64))
        return np.column_stack((np.ones(len(x)), self.factor * np.cos(
            x @ self.frequencies.T + self.phases)))

    def update(self, point, score):
        phi = self.features(point)[0]
        p_phi = self.inverse @ phi
        denominator = 1.0 + float(phi @ p_phi)
        error = float(score) - self.offset - float(phi @ self.coefficients)
        self.coefficients += p_phi * error / denominator
        self.inverse -= np.outer(p_phi, p_phi) / denominator
        # 抑制递推运算产生的反对称浮点舍入量。
        self.inverse = (self.inverse + self.inverse.T) / 2

    def predict(self, points):
        return self.features(points) @ self.coefficients + self.offset

    def gradient(self, points):
        x = np.atleast_2d(points)
        weighted = -self.factor * np.sin(x @ self.frequencies.T + self.phases)
        return (weighted * self.coefficients[1:]) @ self.frequencies


def _candidate_pool(model, best, rng):
    dimension = len(best)
    global_points = rng.uniform(0, 1, (GLOBAL_CANDIDATES, dimension))
    radii = np.resize(np.array([0.03, 0.10, 0.25]), LOCAL_CANDIDATES)
    local_points = np.clip(best + rng.normal(size=(LOCAL_CANDIDATES, dimension))
                           * radii[:, None], 0, 1)
    candidates = np.vstack((best, global_points, local_points))
    points = candidates[np.argsort(model.predict(candidates))[-LOCAL_STARTS:]].copy()
    for step in range(LOCAL_STEPS):
        gradients = model.gradient(points)
        gradients /= np.maximum(1.0, np.max(np.abs(gradients), axis=1))[:, None]
        points = np.clip(points + LOCAL_RATE / np.sqrt(step + 1) * gradients, 0, 1)
    return np.vstack((candidates, points))


def optimize(session, rng):
    """在线拟合只接收已经付费获得的反馈，代理预测不作真实评分。"""
    if not session.controls or len(session.controls) != len(session.scores):
        raise ValueError("session 需要成对保存共同预探测控制与分数。")
    dimension = 2 * int(session.cfg.n)
    model = _FourierSurrogate(dimension, rng, np.mean(session.scores))
    visited = set()
    for control, score in zip(session.controls, session.scores):
        model.update(control, score)
        visited.add(np.asarray(session.project(control), np.float64).tobytes())
    while session.remaining > 0:
        best = np.asarray(session.best_control, np.float64)
        candidates = np.asarray([session.project(u) for u in _candidate_pool(model, best, rng)],
                                np.float64)
        if rng.random() < EXPLORATION_PROBABILITY:
            order = rng.permutation(len(candidates))
        else:
            order = np.argsort(model.predict(candidates))[::-1]
        chosen = None
        for index in order:
            candidate = np.asarray(session.project(candidates[index]), np.float64)
            if candidate.tobytes() not in visited:
                chosen = candidate
                break
        if chosen is None:
            # 离散网格中局部池重复时作有限次数的全域探索。
            for _ in range(32):
                candidate = np.asarray(session.project(rng.uniform(0, 1, dimension)), np.float64)
                if candidate.tobytes() not in visited:
                    chosen = candidate
                    break
        if chosen is None:
            break
        score = float(session.evaluate(chosen))
        model.update(chosen, score)
        visited.add(chosen.tobytes())
    return np.asarray(session.best_control, np.float64).copy()
