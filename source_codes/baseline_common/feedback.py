"""对所有在线算法强制执行共同测量预算。"""
import numpy as np
import time
from .controls import project


class FeedbackSession:
    def __init__(self, cfg, observation, measure_callback, budget):
        self.cfg = cfg
        self.observation = observation
        self.budget = int(budget)
        self._measure = measure_callback
        self.simulation_feedback_seconds = 0.0
        self.controls = [project(u, cfg) for u in observation['probe_controls']]
        self.scores = list(map(float, observation['quality']))
        if self.budget < len(self.controls) or len(self.scores) != len(self.controls):
            raise ValueError('总预算必须覆盖共同初始探测。')
        if not np.all(np.isfinite(self.scores)):
            raise ValueError('初始质量数据包含非有限值。')

    @property
    def calls(self):
        return len(self.scores)

    @property
    def remaining(self):
        return self.budget-self.calls

    @property
    def best_control(self):
        return self.controls[int(np.argmax(self.scores))].copy()

    @property
    def best_score(self):
        return float(np.max(self.scores))

    def project(self, u):
        return project(u, self.cfg)

    def evaluate(self, u):
        if self.remaining <= 0:
            raise RuntimeError('已用完全部反馈预算。')
        u = self.project(u)
        start = time.perf_counter()
        score = float(self._measure(u, self.calls))
        self.simulation_feedback_seconds += time.perf_counter()-start
        if not np.isfinite(score):
            raise FloatingPointError('模拟器返回非有限质量值。')
        self.controls.append(u.copy())
        self.scores.append(score)
        return score
