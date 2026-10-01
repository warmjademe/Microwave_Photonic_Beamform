"""保留冻结模型，以共同反馈包装实现真正的 64 次控制探测。"""
import copy
import time
import numpy as np
from baseline_common.feedback import FeedbackSession
from baseline_common.controls import codes
from baseline_codebook.method import optimize as scan_codebook
from study_full_baselines.common import NativeConfig, LEVELS, observation

DIRECT = ['mlp', 'dnn', 'cnn', 'rescnn', 'transformer', 'complex_cnn', 'jct']
NEW_METHODS = ['initial_select64'] + [name + '_feedback64' for name in DIRECT]
REUSED_METHODS = ['codebook', 'coordinate', 'spsa', 'done', 'de', 'cnn_warm64']
BASELINES64 = NEW_METHODS + REUSED_METHODS[:-1]


class Feedback64:
    """输入仍为 16 次公开观测；新增反馈仅供候选评价，不进入模型。"""
    warm = True
    kind = 'feedback64'

    def __init__(self, base, public, fixed_catalog=False):
        self.base = base
        self.public = public
        self.fixed_catalog = fixed_catalog
        self.cfg = NativeConfig()
        self.artifacts = base.artifacts

    def decide(self, raw, rng, measure_callback=None):
        if measure_callback is None:
            raise ValueError('64 次反馈必须提供测量接口')
        start = time.perf_counter()
        session = FeedbackSession(self.cfg, observation(raw, self.public), measure_callback, 64)
        fallback_rng = copy.deepcopy(rng)
        key = lambda u: tuple(codes(u, self.cfg))
        seen = {key(u) for u in session.controls}
        proposed = None

        def measure_new(control):
            k = key(control)
            if k not in seen:
                session.evaluate(control)
                seen.add(k)

        if self.fixed_catalog:
            catalog = self.public['catalog_controls']
            if catalog.shape != (64, 128):
                raise ValueError('固定扫描必须使用原公开 64 套控制')
            np.testing.assert_array_equal(catalog[:16], self.public['probe_controls'])
            for control in catalog[16:]:
                measure_new(control)
        else:
            predicted = self.base.decide(raw.copy(), copy.deepcopy(rng), None)
            proposed = predicted['control_code']
            measure_new(proposed / LEVELS)
            scan_codebook(session, fallback_rng)
        if session.calls != 64 or len({key(u) for u in session.controls}) != 64:
            raise ValueError('应实际测量 64 个去重控制')
        selected = codes(session.best_control, self.cfg)
        result = dict(control_code=selected, control=selected / LEVELS,
                      feedback_calls=session.calls,
                      software_seconds=max(0., time.perf_counter()-start-session.simulation_feedback_seconds),
                      feedback_simulator_seconds=session.simulation_feedback_seconds,
                      trace_control_code=np.asarray([codes(u, self.cfg) for u in session.controls]),
                      trace_scores=np.asarray(session.scores))
        if proposed is not None:
            result['proposal_control_code'] = proposed
        return result


def audit(arrays, methods, public, carrier):
    """核对逐次测量及最终选择，而不只相信汇总测量计数。"""
    raw, control, metrics = (arrays[k] for k in ['public_X', 'control_code', 'metrics'])
    if raw.shape != (2513,) or raw[1984] != carrier or not np.isfinite(raw).all():
        raise ValueError('公开观测错误')
    if control.shape != (len(methods), 128) or metrics.shape != (len(methods), 13):
        raise ValueError('方法或指标覆盖错误')
    if not np.isfinite(metrics).all() or np.any(metrics < 0):
        raise ValueError('指标或时间非法')
    np.testing.assert_array_equal(metrics[:, [1, 4, 6, 10]],
                                  np.tile([496, 248, 8, 64], (len(methods), 1)))
    if (np.any(metrics[:, 3] > metrics[:, 0]) or np.any(metrics[:, 0] > 2*metrics[:, 3])
            or np.any(metrics[:, 0] > 496) or np.any(metrics[:, 3] > 248)
            or np.any(metrics[:, 5] > 8) or np.any(metrics[:, 8] <= 0)):
        raise ValueError('错误计数或信噪功率关系错误')
    np.testing.assert_array_equal(metrics[:, [0, 1, 3, 4, 5, 6]],
                                  np.rint(metrics[:, [0, 1, 3, 4, 5, 6]]))
    initial = np.rint(public['probe_controls'] * LEVELS).astype(np.int16)
    allowed = {'public_X', 'control_code', 'metrics'}
    for i, name in enumerate(methods):
        ck, sk = name+'__trace_control_code', name+'__trace_scores'
        allowed.update([ck, sk])
        trace, scores = arrays[ck], arrays[sk]
        if trace.shape != (64, 128) or scores.shape != (64,) or not np.isfinite(scores).all():
            raise ValueError('反馈轨迹不完整')
        if (np.any(trace < 0) or np.any(trace > LEVELS)
                or not np.array_equal(trace, np.rint(trace)) or len(np.unique(trace, axis=0)) != 64):
            raise ValueError('控制非法或重复计费')
        np.testing.assert_array_equal(trace[:16], initial)
        np.testing.assert_array_equal(scores[:16], raw[1985:2001])
        np.testing.assert_array_equal(control[i], trace[int(scores.argmax())])
        if name != 'initial_select64':
            pk = name+'__proposal_control_code'; allowed.add(pk)
            if not any(np.array_equal(arrays[pk], u) for u in trace):
                raise ValueError('模型候选未实际测量或复用')
    if set(arrays) != allowed:
        raise ValueError('出现未声明字段或丢失信息')
    return len(methods) * 64
