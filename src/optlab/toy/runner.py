"""Run optimizers on toy benchmarks and record results in the DB (TRD 10, FR-B1..B3)."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optlab.config import OptimizerConfig, ToyBenchmarkConfig, ToyExperimentConfig
from optlab.db.repository import git_commit, id_of, insert, insert_many, upsert_experiment
from optlab.optim import build_optimizer
from optlab.toy.functions import Fn, make_function, start_point

NEEDS_CLOSURE = {"sam_sgd_momentum", "lbfgs"}


@dataclass
class ToyResult:
    steps_to_tol: int | None
    final_f: float | None  # None when the run diverged
    final_grad_norm: float | None
    curve: list[tuple[int, float, float]] = field(default_factory=list)  # (step, f, |grad|)
    trajectory: np.ndarray | None = None  # (steps + 1, dim)


def run_one(
    fn: Fn,
    x0: np.ndarray,
    optimizer: str,
    hparams: dict[str, Any],
    max_steps: int,
    tolerance: float,
    f_star: float = 0.0,
    divergence_threshold: float = 1e12,
    curve_dense_steps: int = 100,
    curve_stride: int = 10,
) -> ToyResult:
    """Iterate until |f - f*| <= tolerance, divergence, or max_steps.

    One step is one `optimizer.step()` (for L-BFGS: one outer iteration with max_iter=1, its
    line-search evaluations not counted; for Newton: one x <- x - H^-1 g). The gradient
    evaluated for logging at x_t is the one the step uses, except for closure-based
    optimizers (SAM, L-BFGS), which re-evaluate inside `step`.
    """
    x = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float64))
    opt = None if optimizer == "newton" else build_optimizer(optimizer, [x], hparams)

    def closure() -> torch.Tensor:
        x.grad = None
        value = fn(x)
        value.backward()
        return value

    path = [x0.copy()]
    curve: list[tuple[int, float, float]] = []
    steps_to_tol = None
    for t in range(max_steps + 1):
        f = float(closure().detach())
        gnorm = float(x.grad.norm())
        if not (math.isfinite(f) and math.isfinite(gnorm)) or abs(f) > divergence_threshold:
            return ToyResult(None, None, None, curve, np.array(path[:-1]))
        if t <= curve_dense_steps or t % curve_stride == 0:
            curve.append((t, f, gnorm))
        if abs(f - f_star) <= tolerance:
            steps_to_tol = t
            break
        if t == max_steps:
            break
        try:
            with torch.no_grad():
                if opt is None:
                    hess = torch.autograd.functional.hessian(fn, x.detach())
                    x -= torch.linalg.solve(hess, x.grad)
                elif optimizer in NEEDS_CLOSURE:
                    opt.step(closure)
                else:
                    opt.step()
        except torch.linalg.LinAlgError:  # singular Hessian: Newton step undefined
            return ToyResult(None, None, None, curve, np.array(path))
        path.append(x.detach().numpy().copy())
    if curve[-1][0] != t:
        curve.append((t, f, gnorm))
    return ToyResult(steps_to_tol, f, gnorm, curve, np.array(path))


def _rank(result: ToyResult, lr: float) -> tuple[float, float, float]:
    """Fewest steps to tolerance; then lowest final f; diverged last; then smaller lr."""
    steps = result.steps_to_tol if result.steps_to_tol is not None else math.inf
    final = result.final_f if result.final_f is not None else math.inf
    return (steps, final, lr)


def run_benchmark_optimizer(
    exp: ToyExperimentConfig,
    bench: ToyBenchmarkConfig,
    opt_cfg: OptimizerConfig,
) -> list[tuple[int, dict[str, Any], ToyResult]]:
    """All seeds for one (benchmark, optimizer). First-order optimizers get their learning rate
    from `exp.lr_grid`, chosen on `exp.tuning_seed`; references use their fixed settings."""
    fn = make_function(bench)
    base = {**opt_cfg.fixed, **exp.hparam_overrides.get(opt_cfg.name, {})}
    kwargs = dict(
        max_steps=bench.max_steps,
        tolerance=bench.tolerance,
        divergence_threshold=exp.divergence_threshold,
        curve_dense_steps=exp.curve_dense_steps,
        curve_stride=exp.curve_stride,
    )

    def run(seed: int, hp: dict[str, Any]) -> ToyResult:
        return run_one(fn, start_point(bench, seed), opt_cfg.name, hp, **kwargs)

    done: dict[int, ToyResult] = {}
    if opt_cfg.family == "second_order":
        hparams = base
    else:
        trials = {lr: run(exp.tuning_seed, {**base, "lr": lr}) for lr in exp.lr_grid}
        best_lr = min(trials, key=lambda lr: _rank(trials[lr], lr))
        hparams = {**base, "lr": best_lr}
        done[exp.tuning_seed] = trials[best_lr]
    return [(s, hparams, done[s] if s in done else run(s, hparams)) for s in exp.seeds]


def record(
    conn: sqlite3.Connection,
    bench: ToyBenchmarkConfig,
    optimizer: str,
    seed: int,
    hparams: dict[str, Any],
    result: ToyResult,
    trajectory_dir: Path,
) -> int:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    traj_path = trajectory_dir / f"{bench.name}__{optimizer}__seed{seed}.npz"
    np.savez_compressed(traj_path, x=result.trajectory)
    toy_run_id = insert(
        conn,
        "toy_runs",
        {
            "benchmark_id": id_of(conn, "toy_benchmarks", bench.name),
            "optimizer_id": id_of(conn, "optimizers", optimizer),
            "seed": seed,
            "hyperparams_json": json.dumps(hparams, sort_keys=True),
            "max_steps": bench.max_steps,
            "tolerance": bench.tolerance,
            "steps_to_tol": result.steps_to_tol,
            "final_f": result.final_f,
            "final_grad_norm": result.final_grad_norm,
            "trajectory_path": traj_path.as_posix(),
        },
    )
    insert_many(
        conn,
        "toy_run_curves",
        ["toy_run_id", "step", "f_value", "grad_norm"],
        [(toy_run_id, *row) for row in result.curve],
    )
    return toy_run_id


def run_suite(
    conn: sqlite3.Connection,
    exp: ToyExperimentConfig,
    optimizers: dict[str, OptimizerConfig],
    force: bool = False,
    only_benchmarks: set[str] | None = None,
    only_optimizers: set[str] | None = None,
    log: Any = print,
) -> None:
    """Resumable: a (benchmark, optimizer) whose seeds are all recorded is skipped unless
    `force`, which deletes and recomputes it."""
    torch.set_num_threads(1)  # 2-D problems: thread overhead only slows things down
    upsert_experiment(conn, exp.name, exp.phase, exp.description, exp.hypothesis, git_commit())
    conn.commit()
    for bench in exp.benchmarks:
        if only_benchmarks and bench.name not in only_benchmarks:
            continue
        names = exp.optimizers + (exp.reference_optimizers if bench.references else [])
        for name in names:
            if only_optimizers and name not in only_optimizers:
                continue
            key = (id_of(conn, "toy_benchmarks", bench.name), id_of(conn, "optimizers", name))
            existing = conn.execute(
                "SELECT COUNT(*) FROM toy_runs WHERE benchmark_id = ? AND optimizer_id = ?", key
            ).fetchone()[0]
            if existing == len(exp.seeds) and not force:
                continue
            conn.execute("DELETE FROM toy_runs WHERE benchmark_id = ? AND optimizer_id = ?", key)
            for seed, hparams, result in run_benchmark_optimizer(exp, bench, optimizers[name]):
                record(conn, bench, name, seed, hparams, result, Path(exp.trajectory_dir))
            conn.commit()
            reached = sum(
                1
                for r in conn.execute(
                    "SELECT steps_to_tol FROM toy_runs WHERE benchmark_id = ? AND optimizer_id = ?",
                    key,
                )
                if r[0] is not None
            )
            log(f"{bench.name:<16} {name:<17} reached tol {reached}/{len(exp.seeds)}")
