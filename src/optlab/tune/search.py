"""Hyperparameter search (TRD 7, FR-T1/T2) with an exact, equal trial budget.

Every method proposes points in the unit cube [0, 1]^d; `from_unit` maps them to
hyperparameters (log scale for log_uniform; a `p_zero` dimension sends u < p_zero to 0).
Each method calls `objective` exactly `budget` times, so all optimizers get the same budget.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from optlab.config import SpaceDim


@dataclass
class Trial:
    index: int
    hparams: dict[str, Any]
    score: float  # validation AUC; -inf for diverged trials (still recorded)


Objective = Callable[[int, dict[str, Any]], float]


def from_unit(space: Mapping[str, SpaceDim], u: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (name, dim), v in zip(space.items(), np.clip(u, 0.0, 1.0), strict=True):
        if dim.type == "choice":
            out[name] = dim.values[min(int(v * len(dim.values)), len(dim.values) - 1)]
            continue
        if dim.p_zero and v < dim.p_zero:
            out[name] = 0.0
            continue
        r = (v - dim.p_zero) / (1 - dim.p_zero) if dim.p_zero else v
        if dim.type == "log_uniform":  # clamped: exp(log(high)) can overshoot by 1 ulp
            out[name] = min(
                dim.high,
                float(math.exp(math.log(dim.low) + r * (math.log(dim.high) - math.log(dim.low)))),
            )
        else:
            out[name] = float(dim.low + r * (dim.high - dim.low))
    return out


def _run(space, objective: Objective, points: list[np.ndarray]) -> list[Trial]:
    trials = []
    for i, u in enumerate(points):
        hp = from_unit(space, u)
        trials.append(Trial(i, hp, objective(i, hp)))
    return trials


def random_search(space, objective: Objective, budget: int, seed: int) -> list[Trial]:
    rng = np.random.default_rng(seed)
    return _run(space, objective, [rng.random(len(space)) for _ in range(budget)])


def grid_search(space, objective: Objective, budget: int, seed: int) -> list[Trial]:
    """Cell-centred grid with ceil(budget^(1/d)) points per dimension; exactly `budget` of
    its points are evaluated, in a seeded random order, to keep the budget equal."""
    d = len(space)
    k = math.ceil(budget ** (1 / d) - 1e-9)
    axis = (np.arange(k) + 0.5) / k
    grid = [np.array(p) for p in itertools.product(axis, repeat=d)]
    order = np.random.default_rng(seed).permutation(len(grid))[:budget]
    return _run(space, objective, [grid[i] for i in sorted(order)])


def pso(
    space,
    objective: Objective,
    budget: int,
    seed: int,
    particles: int = 5,
    inertia: float = 0.7,
    c_personal: float = 1.5,
    c_global: float = 1.5,
) -> list[Trial]:
    """Particle swarm in the unit cube; `particles` evaluations per iteration."""
    rng = np.random.default_rng(seed)
    d = len(space)
    pos = rng.random((particles, d))
    vel = rng.uniform(-0.2, 0.2, (particles, d))
    best_pos, best_val = pos.copy(), np.full(particles, -np.inf)
    trials: list[Trial] = []
    while len(trials) < budget:
        for p in range(particles):
            if len(trials) == budget:
                break
            hp = from_unit(space, pos[p])
            score = objective(len(trials), hp)
            trials.append(Trial(len(trials), hp, score))
            if score > best_val[p]:
                best_val[p], best_pos[p] = score, pos[p].copy()
        g = best_pos[int(np.argmax(best_val))]
        r1, r2 = rng.random((particles, d)), rng.random((particles, d))
        vel = inertia * vel + c_personal * r1 * (best_pos - pos) + c_global * r2 * (g - pos)
        pos = np.clip(pos + vel, 0.0, 1.0)
    return trials


def _tournament(rng: np.random.Generator, pop: list[np.ndarray], scores: list[float]) -> np.ndarray:
    i, j = rng.integers(len(pop), size=2)
    return pop[i] if scores[i] >= scores[j] else pop[j]


def ga(
    space,
    objective: Objective,
    budget: int,
    seed: int,
    population: int = 5,
    mutation_sd: float = 0.1,
) -> list[Trial]:
    """Genetic algorithm in the unit cube: tournament selection, uniform crossover, Gaussian
    mutation, elitism (the elite carries over without being re-evaluated)."""
    rng = np.random.default_rng(seed)
    d = len(space)
    pop = [rng.random(d) for _ in range(min(population, budget))]
    trials: list[Trial] = []
    scores: list[float] = []
    for u in pop:
        hp = from_unit(space, u)
        scores.append(objective(len(trials), hp))
        trials.append(Trial(len(trials), hp, scores[-1]))
    while len(trials) < budget:
        elite = int(np.argmax(scores))
        new_pop, new_scores = [pop[elite]], [scores[elite]]
        while len(new_pop) < population and len(trials) < budget:
            a, b = _tournament(rng, pop, scores), _tournament(rng, pop, scores)
            child = np.where(rng.random(d) < 0.5, a, b) + rng.normal(0, mutation_sd, d)
            child = np.clip(child, 0.0, 1.0)
            hp = from_unit(space, child)
            score = objective(len(trials), hp)
            trials.append(Trial(len(trials), hp, score))
            new_pop.append(child)
            new_scores.append(score)
        pop, scores = new_pop, new_scores
    return trials


METHODS = {"random": random_search, "grid": grid_search, "pso": pso, "ga": ga}


def search(
    method: str, space: Mapping[str, SpaceDim], objective: Objective, budget: int, seed: int
) -> list[Trial]:
    trials = METHODS[method](space, objective, budget, seed)
    if len(trials) != budget:  # the fairness guarantee: enforce it, don't assume it
        raise AssertionError(f"{method} used {len(trials)} trials, budget was {budget}")
    return trials
