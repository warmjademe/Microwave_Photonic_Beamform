"""正式执行候选控制时只计算所需五个起点，不重复计算诊断对照。"""
import numpy as np
from our_method_response_control.physics import decode
from our_method_two_stage.control import phase_starts, proxy_quality


def decode_multistart(response_estimate, carrier_ghz, initial_control):
    starts = np.vstack([initial_control, phase_starts(response_estimate, carrier_ghz)])
    candidates = np.asarray([decode(response_estimate, carrier_ghz, u, sweeps=2)[0] for u in starts])
    qualities = proxy_quality(response_estimate, carrier_ghz, candidates)
    best = int(qualities[:, 1].argmin())
    return candidates[best], dict(selected_start=best, starts=5,
        coordinate_sweeps=10, extra_feedback=0, candidate_proxy=qualities.tolist())
