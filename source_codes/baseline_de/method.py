"""DE/rand/1/bin 的有界离散适配；所有种群评分均来自实际反馈。"""
import numpy as np


POPULATION_SIZE = 16
MUTATION_FACTOR = 0.65
CROSSOVER_RATE = 0.80


def optimize(session, rng):
    n = int(session.cfg.n)
    dimension = 2 * n
    levels = np.r_[np.full(n, session.cfg.delay_levels),
                   np.full(n, session.cfg.attenuation_levels)].astype(np.float64)
    if np.any(levels < 1) or not session.controls:
        raise ValueError("需要正的控制码级数和共同的预探测数据。")
    # 共同预探测已经付费，允许复用；同一个控制只占一个种群位置。
    population, scores, used = [], [], set()
    for index in np.argsort(session.scores)[::-1]:
        control = np.asarray(session.project(session.controls[index]), np.float64)
        if control.tobytes() in used:
            continue
        population.append(control.copy())
        scores.append(float(session.scores[index]))
        used.add(control.tobytes())
        if len(population) == POPULATION_SIZE:
            break
    duplicate_attempts = 0
    while len(population) < POPULATION_SIZE and session.remaining > 0:
        candidate = np.asarray(session.project(rng.uniform(0, 1, dimension)), np.float64)
        if candidate.tobytes() in used:
            duplicate_attempts += 1
            if duplicate_attempts >= 32:
                break
            continue
        score = float(session.evaluate(candidate))
        population.append(candidate)
        scores.append(score)
        used.add(candidate.tobytes())
    # rand/1/bin 至少需要目标以外的三个不同种群成员。
    if len(population) < 4:
        return np.asarray(session.best_control, np.float64).copy()
    population = np.asarray(population, np.float64)
    scores = np.asarray(scores, np.float64)
    while session.remaining > 0:
        for target in rng.permutation(len(population)):
            if session.remaining <= 0:
                break
            eligible = np.delete(np.arange(len(population)), target)
            a, b, c = rng.choice(eligible, 3, replace=False)
            mutant = population[a] + MUTATION_FACTOR * (population[b] - population[c])
            crossover = rng.random(dimension) < CROSSOVER_RATE
            crossover[int(rng.integers(dimension))] = True
            trial = np.where(crossover, mutant, population[target])
            trial = np.asarray(session.project(trial), np.float64)
            if np.array_equal(trial, population[target]):
                # 量化吞掉差异时，至少引入一个合法控制码的变化。
                coordinate = int(rng.integers(dimension))
                sign = 1.0 if trial[coordinate] <= 0.5 else -1.0
                trial[coordinate] += sign / levels[coordinate]
                trial = np.asarray(session.project(trial), np.float64)
            score = float(session.evaluate(trial))
            if score >= scores[target]:
                population[target] = trial
                scores[target] = score
    return np.asarray(session.best_control, np.float64).copy()
