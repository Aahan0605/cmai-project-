from pathlib import Path

import numpy as np
import pytest
import torch

from optlab.config import ToyBenchmarkConfig, ToyExperimentConfig, load_optimizers
from optlab.db.connection import connect, init_schema
from optlab.db.repository import seed_catalogues
from optlab.toy.functions import make_function, minimizer, quadratic_matrix
from optlab.toy.report import write_report
from optlab.toy.runner import run_one, run_suite


def bench(kind: str, **kw) -> ToyBenchmarkConfig:
    defaults = dict(
        name=kind,
        kind=kind,
        dim=2,
        max_steps=2000,
        tolerance=1e-8,
        start_low=-1.0,
        start_high=1.0,
    )
    if kind == "quadratic":
        defaults["condition_number"] = 1000.0
    return ToyBenchmarkConfig(**(defaults | kw))


def test_functions_have_known_minimum() -> None:
    for kind in ["quadratic", "rosenbrock", "rastrigin"]:
        cfg = bench(kind, dim=5)
        assert float(make_function(cfg)(torch.tensor(minimizer(cfg)))) == pytest.approx(0.0)
    eig = torch.linalg.eigvalsh(quadratic_matrix(4, 1000.0, rotation_seed=3))
    assert float(eig.max() / eig.min()) == pytest.approx(1000.0)


def test_newton_solves_quadratic_in_one_step() -> None:
    cfg = bench("quadratic")
    result = run_one(make_function(cfg), np.array([0.8, -0.6]), "newton", {}, 50, 1e-12)
    assert result.steps_to_tol == 1
    assert result.trajectory.shape == (2, 2)


def test_gd_converges_and_logs_curve() -> None:
    cfg = bench("quadratic", condition_number=10.0)
    result = run_one(make_function(cfg), np.array([0.8, -0.6]), "sgd", {"lr": 0.1}, 2000, 1e-8)
    assert result.steps_to_tol is not None and result.final_f <= 1e-8
    steps = [s for s, _, _ in result.curve]
    assert steps[0] == 0 and steps[-1] == result.steps_to_tol
    assert len(result.trajectory) == result.steps_to_tol + 1


def test_divergence_is_recorded_not_raised() -> None:
    # lr * lambda_max = 1000 >> 2: plain GD must blow up on the kappa = 1000 quadratic.
    cfg = bench("quadratic")
    result = run_one(make_function(cfg), np.array([0.8, -0.6]), "sgd", {"lr": 1.0}, 2000, 1e-8)
    assert result.steps_to_tol is None and result.final_f is None
    assert all(np.isfinite(f) for _, f, _ in result.curve)


def test_suite_writes_db_and_report(root: Path, tmp_path: Path) -> None:
    exp = ToyExperimentConfig(
        name="toy_test",
        hypothesis="h",
        description="d",
        seeds=[0, 1],
        lr_grid=[0.01, 0.1],
        optimizers=["sgd", "adam"],
        reference_optimizers=["newton"],
        trajectory_dir=str(tmp_path / "traj"),
        benchmarks=[
            bench("quadratic", name="q", condition_number=10.0, references=True),
            bench("quadratic", name="q100", condition_number=100.0),
        ],
    )
    optimizers = load_optimizers(root / "configs")
    conn = connect(tmp_path / "r.db")
    init_schema(conn, root / "backend" / "schema.sql")
    seed_catalogues(conn, optimizers.values(), [], [], exp.benchmarks)
    run_suite(conn, exp, optimizers, log=lambda _: None)
    assert conn.execute("SELECT COUNT(*) FROM toy_runs").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM toy_run_curves").fetchone()[0] > 6
    paths = [r[0] for r in conn.execute("SELECT trajectory_path FROM toy_runs")]
    assert all(Path(p).exists() for p in paths)
    # Resumable: a second call recomputes nothing (same row ids).
    ids = [r[0] for r in conn.execute("SELECT id FROM toy_runs ORDER BY id")]
    run_suite(conn, exp, optimizers, log=lambda _: None)
    assert [r[0] for r in conn.execute("SELECT id FROM toy_runs ORDER BY id")] == ids

    unrelated = exp.model_copy(
        update={"name": "other", "benchmarks": [bench("quadratic", name="unrelated", max_steps=2)]}
    )
    seed_catalogues(conn, [], [], [], unrelated.benchmarks)
    run_suite(conn, unrelated, optimizers, log=lambda _: None)
    write_report(conn, exp, tmp_path / "out", tmp_path / "summary.md")
    summary = (tmp_path / "summary.md").read_text()
    assert "unrelated" not in summary
    assert "| newton | 1 [1–1] (2/2) |" in summary
    assert (tmp_path / "out" / "q_trajectories.png").exists()
    assert (tmp_path / "out" / "quadratic_scaling.png").exists()
