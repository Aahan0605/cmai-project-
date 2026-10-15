"""Contract tests: the read-only API and the static snapshot return the same objects."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import ROOT  # tests/ is on sys.path under pytest
from optlab import config
from optlab.api import create_app
from optlab.config import ExperimentConfig, SplitConfig, ToyBenchmarkConfig, ToyExperimentConfig
from optlab.data.manifest import VideoRow, assign_splits
from optlab.db.connection import connect, init_schema
from optlab.db.repository import seed_catalogues
from optlab.export.snapshot import export_snapshot
from optlab.pipeline import Experiment
from optlab.toy.runner import run_suite
from test_pipeline import synthetic


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, Path, ToyExperimentConfig]:
    root = ROOT
    tmp = tmp_path_factory.mktemp("api")
    ff = [VideoRow(f"v{i}", "", int(i % 3 != 0), "m", f"id{i}") for i in range(60)]
    synthetic(tmp, "faceforensics_pp", assign_splits(ff, SplitConfig(seed=0)), 0.6, 0)
    db = tmp / "r.db"
    conn = connect(db)
    init_schema(conn, root / "backend" / "schema.sql")
    cfgs = root / "configs"
    optimizers, models = config.load_optimizers(cfgs), config.load_models(cfgs)
    toy_cfg = ToyExperimentConfig(
        name="toy_api",
        hypothesis="h",
        seeds=[0],
        lr_grid=[0.1],
        optimizers=["sgd"],
        trajectory_dir=str(tmp / "traj"),
        benchmarks=[
            ToyBenchmarkConfig(
                name="q",
                kind="quadratic",
                dim=2,
                condition_number=10,
                max_steps=200,
                tolerance=1e-6,
                start_low=-1,
                start_high=1,
            )
        ],
    )
    seed_catalogues(
        conn,
        optimizers.values(),
        models.values(),
        config.load_datasets(cfgs).values(),
        toy_cfg.benchmarks,
    )
    run_suite(conn, toy_cfg, optimizers, log=lambda _: None)
    cfg = ExperimentConfig(
        name="api_test",
        phase="baseline",
        hypothesis="h",
        train_dataset="faceforensics_pp",
        eval={"in_distribution": "faceforensics_pp"},
        model="clip_vit_b16__freq__weighted_sum",
        optimizers=["adam", "sgd_momentum"],
        tuning={"trials": 2, "epochs": 2},
        training={"max_epochs": 3, "batch_size": 64, "patience": 2},
        seeds=[0, 1],
        fusion="simplex_temperature",
        manifests_dir=str(tmp / "manifests"),
        features_dir=str(tmp / "features"),
        artifacts_dir=str(tmp / "artifacts"),
    )
    Experiment(conn, cfg, tmp / "cfg.yaml", optimizers, models, log=lambda _: None).sweep()
    conn.close()
    return db, tmp, toy_cfg


@pytest.fixture(scope="module")
def client(built, tmp_path_factory) -> TestClient:
    db, tmp, toy_cfg = built
    path = tmp / "toy.yaml"
    path.write_text(json.dumps(toy_cfg.model_dump(mode="json")))  # JSON is valid YAML
    return TestClient(create_app(db, path))


def test_health_and_catalogue(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "schema_version": 4}
    exps = client.get("/api/experiments").json()
    assert [e["name"] for e in exps] == ["toy_api", "api_test"]
    assert {o["name"] for o in client.get("/api/optimizers").json()} >= {"adam", "sgd"}


def test_experiment_views(client: TestClient) -> None:
    exp_id = next(e["id"] for e in client.get("/api/experiments").json() if e["name"] == "api_test")
    lb = client.get(f"/api/experiments/{exp_id}/leaderboard").json()
    table = next(t for t in lb["tables"] if t["granularity"] == "video")
    assert {r["optimizer"] for r in table["rows"]} == {"adam", "sgd_momentum"}
    assert all(r["n_seeds"] == 2 and len(r["values"]) == 2 for r in table["rows"])
    assert len(table["pairwise"]) == 1
    conv = client.get(f"/api/experiments/{exp_id}/convergence").json()["optimizers"]
    assert set(conv["adam"]["metrics"]) == {"val_auc", "val_loss", "train_loss", "grad_norm"}
    tuning = client.get(f"/api/experiments/{exp_id}/tuning").json()["optimizers"]
    assert sum(t["selected"] for t in tuning["adam"]["trials"]) == 1
    fusion = client.get(f"/api/experiments/{exp_id}/fusion").json()["runs"]
    assert all(
        f["formulation"] == "simplex_temperature" and f["kkt_residual"] < 1e-6 for f in fusion
    )
    assert abs(sum(fusion[0]["weights"].values()) - 1) < 1e-9
    runs = client.get(f"/api/experiments/{exp_id}/runs").json()
    detail = client.get(f"/api/runs/{runs[0]['id']}").json()
    assert detail["epochs"] and detail["evaluations"] and detail["fusion"]["weights"]


def test_toy_view(client: TestClient) -> None:
    toy = client.get("/api/toy").json()
    bench = toy["benchmarks"][0]
    assert bench["grid"] and len(bench["grid"]["z"]) == 80
    assert toy["trajectories"]["q"]["sgd"] and toy["curves"]["q"]["sgd"]


def test_errors_and_read_only(client: TestClient) -> None:
    assert client.get("/api/runs/999999").status_code == 404
    assert client.get("/api/experiments/999/leaderboard").status_code == 404
    assert client.get("/api/experiments/1/nonsense").status_code == 404
    assert client.get("/api/cv/nothing").status_code == 404
    assert client.post("/api/experiments").status_code == 405


def test_snapshot_matches_api(client: TestClient, built) -> None:
    db, tmp, toy_cfg = built
    out = tmp / "snapshot"
    conn = connect(db, readonly=True)
    n = export_snapshot(conn, toy_cfg, out)
    conn.close()
    assert n == 4
    exp_id = next(e["id"] for e in client.get("/api/experiments").json() if e["name"] == "api_test")
    for path in [
        "experiments",
        "optimizers",
        "toy",
        f"experiments/{exp_id}/leaderboard",
        f"experiments/{exp_id}/fusion",
    ]:
        assert (
            json.loads((out / f"{path}.json").read_text()) == client.get(f"/api/{path}").json()
        ), path
    assert json.loads((out / "meta.json").read_text())["mode"] == "snapshot"
    assert np.isclose(json.loads((out / "meta.json").read_text())["schema_version"], 4)


def test_failed_export_preserves_previous_snapshot(built, monkeypatch):
    from optlab.export import queries

    db, tmp, toy_cfg = built
    out = tmp / "preserved"
    out.mkdir()
    (out / "meta.json").write_text('{"previous":true}')
    (out / "results.json").write_text("[1,2,3]")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(queries, "toy", fail)
    with connect(db, readonly=True) as conn:
        with pytest.raises(RuntimeError, match="simulated"):
            export_snapshot(conn, toy_cfg, out)
    assert (out / "meta.json").read_text() == '{"previous":true}'
    assert (out / "results.json").read_text() == "[1,2,3]"


def test_export_refuses_unrelated_directory(built):
    db, tmp, toy_cfg = built
    out = tmp / "unrelated"
    out.mkdir()
    (out / "important.txt").write_text("keep me")
    with connect(db, readonly=True) as conn:
        with pytest.raises(ValueError, match="snapshot destination"):
            export_snapshot(conn, toy_cfg, out)
    assert (out / "important.txt").read_text() == "keep me"


def test_api_refuses_unmigrated_database(tmp_path):
    db = tmp_path / "old.db"
    conn = connect(db)
    conn.executescript((ROOT / "backend/schema.sql").read_text())
    conn.close()
    with pytest.raises(RuntimeError, match="init-db"):
        create_app(db, ROOT / "configs/experiments/toy.yaml")
