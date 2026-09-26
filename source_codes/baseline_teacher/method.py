"""用真信道评价的高预算、多起点离线教师；不作为公平在线控制器。"""
import numpy as np
from baseline_common.controls import project


def optimize(engine, initial_controls, rng, starts=2, sweeps=1):
    cfg = engine.cfg
    initial_scores = np.array([engine.evaluate(u)['objective'] for u in initial_controls])
    evaluations = len(initial_controls)
    best_index = int(np.argmax(initial_scores))
    best, best_score = initial_controls[best_index].copy(), float(initial_scores[best_index])
    # 起点均来自公共码本；真值只用于离线排序和优化目标。
    for start in np.argsort(initial_scores)[-starts:]:
        state = engine.state(initial_controls[start])
        for _ in range(sweeps):
            for n in rng.permutation(cfg.n):
                # 粗扫整个时延范围，再在粗扫结果附近逐码细化。
                candidates = np.unique(np.r_[np.arange(0, cfg.delay_levels+1, 5)/cfg.delay_levels,
                                               state['control'][n]])
                u, score, calls = engine.scan_coordinate(state, int(n), candidates)
                evaluations += calls
                state = engine.state(u)
                candidates = np.unique(np.clip(u[n]+np.arange(-4, 5)/cfg.delay_levels, 0, 1))
                u, score, calls = engine.scan_coordinate(state, int(n), candidates)
                evaluations += calls
                state = engine.state(u)
                attenuation = np.arange(cfg.attenuation_levels+1)/cfg.attenuation_levels
                u, score, calls = engine.scan_coordinate(state, cfg.n+int(n), attenuation)
                evaluations += calls
                state = engine.state(u)
                if score > best_score:
                    best, best_score = u.copy(), score
    # 去掉所有支路共同的时延，消除一类等效标签；不改相对时延。
    best[:cfg.n] -= best[:cfg.n].min()
    best = project(best, cfg)
    final = engine.evaluate(best)
    evaluations += 1
    if final['objective'] < float(initial_scores.max())-1e-10:
        raise AssertionError('教师不应劣于已经评价的最好初始控制。')
    return best, dict(objective=final['objective'], snr_db=final['snr_db'],
                      initial_best_objective=float(initial_scores.max()),
                      objective_evaluations=evaluations, starts=starts, sweeps=sweeps,
                      privileged_true_channel=True, globally_optimal=False)
