"""只用估计响应生成候选，再用同预算导频反馈选择实际测过的设置。"""
import copy
import numpy as np
from baseline_codebook.method import optimize as scan_codebook
from our_method_response_control.physics import decode,LEVELS
from our_method_two_stage.control import phase_starts


def optimize(session,rng,response_estimate,carrier_ghz,mode):
    if mode not in ['warm_codebook','response_candidates']:
        raise ValueError('未知的候选生成方式。')
    if session.calls!=16 or session.budget!=64:
        raise ValueError('本轮比较固定16次初始探测、总计64次反馈。')
    h=np.asarray(response_estimate,complex)
    if h.shape!=(64,31) or not np.isfinite(h).all():
        raise ValueError('估计响应应为64×31有限复数。')
    initial=session.best_control
    # 候选起点消耗的随机数不能改变补齐码本的遍历顺序。
    fallback_rng=copy.deepcopy(rng)
    def key(u):return tuple(np.rint(session.project(u)*LEVELS).astype(int))
    seen={key(u) for u in session.controls}
    proposed=unique=0
    def measure_new(u):
        nonlocal unique
        k=key(u)
        if k in seen:return
        session.evaluate(u);seen.add(k);unique+=1
    if mode=='warm_codebook':
        candidate,_=decode(h,carrier_ghz,initial,sweeps=2)
        proposed=1;measure_new(candidate)
    else:
        # 开始观测新反馈前固定48个起点；不使用真实传播参数或数据符号。
        starts=[initial,*phase_starts(h,carrier_ghz)]
        for _ in range(48-len(starts)):
            starts.append(np.r_[rng.integers(0,77,size=64)/76,np.zeros(64)])
        for start in starts:
            if session.remaining==0:break
            candidate,_=decode(h,carrier_ghz,start,sweeps=2)
            proposed+=1;measure_new(candidate)
    before_fallback=session.calls
    # 去重产生空余预算时按原几何码本补足，两种版本都遵守总64次。
    if session.remaining:
        scan_codebook(session,fallback_rng)
    if session.calls!=64:raise ValueError('没有完成预定64次反馈预算。')
    result=session.best_control
    if key(result) not in {key(u) for u in session.controls}:
        raise ValueError('返回了未经测量的控制。')
    return result,dict(mode=mode,model_optimized_candidates=proposed,
        unique_response_candidates=unique,geometric_fallback_queries=session.calls-before_fallback,
        total_feedback_calls=session.calls,extra_feedback_calls=session.calls-16,
        selection='maximum observed pilot score; unknown payload not used')
