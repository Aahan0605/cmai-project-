"""Phase 2 end to end on synthetic cached features (no datasets, no real results)."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from optlab import config
from optlab.config import EvalConfig, ExperimentConfig, SplitConfig
from optlab.data.cache import FeatureCacheWriter
from optlab.data.manifest import VideoRow, assign_splits, write_manifest
from optlab.db.connection import connect, init_schema
from optlab.db.repository import TestSplitAlreadyWritten, seed_catalogues
from optlab.eval.report import write_stats
from optlab.pipeline import Experiment
from optlab.train.data import load_split
from optlab.train.trainer import train

FRAMES = 4


def synthetic(root: Path, dataset: str, rows: list[VideoRow], shift: float, seed: int) -> None:
    """Fake videos: visual features shifted by `shift`, frequency by shift / 3 (a weaker stream)."""
    rng = np.random.default_rng(seed)
    write_manifest(rows, root / "manifests" / f"{dataset}.csv")
    for backbone, dim, scale in [("clip_vit_b16", 8, 1.0), ("freq_spectrum", 4, 1 / 3)]:
        w = FeatureCacheWriter(root / "features" / dataset / backbone, shard_rows=100)
        for r in rows:
            mu = np.full(dim, shift * scale * (1 if r.label else -1))
            w.add(r.video_id, r.label, list(range(FRAMES)), mu + rng.normal(size=(FRAMES, dim)))
        w.close({"backbone": backbone})


@pytest.fixture
def setup(root: Path, tmp_path: Path):
    ff = [VideoRow(f"v{i}", "", int(i % 3 != 0), "m", f"id{i}") for i in range(90)]
    synthetic(tmp_path, "faceforensics_pp", assign_splits(ff, SplitConfig(seed=0)), 0.6, 0)
    celeb = [VideoRow(f"c{i}", "", i % 2, "m", f"cid{i}", "test") for i in range(30)]
    synthetic(tmp_path, "celeb_df_v2", celeb, 0.3, 1)  # weaker signal: a "generalization gap"

    conn = connect(tmp_path / "r.db")
    init_schema(conn, root / "backend" / "schema.sql")
    cfgs = root / "configs"
    optimizers, models = config.load_optimizers(cfgs), config.load_models(cfgs)
    seed_catalogues(
        conn, optimizers.values(), models.values(), config.load_datasets(cfgs).values(), []
    )
    cfg = ExperimentConfig(
        name="synthetic_test",
        phase="baseline",
        hypothesis="h",
        train_dataset="faceforensics_pp",
        eval={"in_distribution": "faceforensics_pp", "cross_dataset": ["celeb_df_v2"]},
        model="clip_vit_b16__freq__weighted_sum",
        optimizers=["sgd_momentum", "adam", "sam_sgd_momentum"],
        tuning={"method": "random", "trials": 3, "epochs": 2, "seed": 0},
        training={"max_epochs": 4, "batch_size": 64, "patience": 2},
        seeds=[0, 1],
        manifests_dir=str(tmp_path / "manifests"),
        features_dir=str(tmp_path / "features"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    exp = Experiment(conn, cfg, tmp_path / "cfg.yaml", optimizers, models, log=lambda _: None)
    return exp, conn, tmp_path


def count(conn: sqlite3.Connection, sql: str) -> int:
    return conn.execute(sql).fetchone()[0]


def test_sweep_end_to_end(setup) -> None:
    exp, conn, tmp_path = setup
    exp.sweep()
    assert count(conn, "SELECT COUNT(*) FROM tuning_trials") == 9  # 3 trials x 3 optimizers
    assert count(conn, "SELECT COUNT(*) FROM tuning_trials WHERE is_selected = 1") == 3
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE status = 'completed'") == 6
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE library_versions_json IS NULL") == 0
    assert json.loads(conn.execute("SELECT library_versions_json FROM runs").fetchone()[0])["torch"]
    # Per run: val (in-dist) + FF++ test + Celeb-DF test, each at frame and video level.
    assert count(conn, "SELECT COUNT(*) FROM evaluations") == 6 * 6
    assert count(conn, "SELECT COUNT(*) FROM v_generalisation_gap") == 6
    for _run_id, total, n in conn.execute(
        "SELECT run_id, SUM(weight), COUNT(*) FROM fusion_weights GROUP BY run_id"
    ):
        assert total == pytest.approx(1.0) and n == 2
    assert count(conn, "SELECT COUNT(*) FROM fusion_solves WHERE kkt_residual >= 1e-6") == 0
    # Selected hyperparameters are what the runs used.
    assert (
        count(
            conn,
            "SELECT COUNT(*) FROM runs r JOIN tuning_trials t ON t.id = r.tuning_trial_id"
            " WHERE t.is_selected = 0 OR t.hyperparams_json != r.hyperparams_json",
        )
        == 0
    )

    # Resumable and idempotent: a second sweep changes nothing.
    snapshot = [tuple(r) for r in conn.execute("SELECT id, status, finished_at FROM runs")]
    exp.sweep()
    assert [tuple(r) for r in conn.execute("SELECT id, status, finished_at FROM runs")] == snapshot
    assert count(conn, "SELECT COUNT(*) FROM tuning_trials") == 9

    # Test-split discipline: a second final evaluation needs force and is flagged.
    run_id = snapshot[0][0]
    with pytest.raises(TestSplitAlreadyWritten):
        exp.evaluate(run_id, final=True)
    exp.evaluate(run_id, final=True, force=True)
    assert (
        count(
            conn,
            f"SELECT COUNT(*) FROM evaluations WHERE run_id = {run_id}"
            " AND split = 'test' AND force_rewritten = 1",
        )
        == 4
    )
    exp.evaluate(run_id)  # validation-only re-evaluation is always allowed

    summary = write_stats(conn, "synthetic_test", tmp_path / "stats")
    text = summary.read_text()
    assert "## Test AUC: celeb_df_v2" in text and "Generalization gap" in text
    assert (tmp_path / "stats" / "convergence_val_auc.png").exists()


def test_divergence_is_recorded(setup) -> None:
    exp, conn, _ = setup
    res = train(
        exp.train_split,
        exp.val_tune,
        exp.optimizers["sgd"],
        {"lr": 1e12},
        0,
        3,
        64,
        2,
        None,
        exp.model_extra,
    )
    assert res.status == "diverged" and res.state_dict is None


def test_val_roles_are_disjoint(setup) -> None:
    exp, _, tmp_path = setup
    tune, fusion = exp.val_tune, exp.val_fusion
    assert not set(tune.video_ids) & set(fusion.video_ids)
    test = load_split(
        tmp_path / "manifests" / "faceforensics_pp.csv", "test", exp._caches("faceforensics_pp")
    )
    assert not set(test.video_ids) & (set(tune.video_ids) | set(exp.train_split.video_ids))


def test_cross_validation_end_to_end(setup, root: Path) -> None:
    import csv

    from optlab.cv import write_folds
    from optlab.eval.cv_report import write_cv_report

    exp, conn, tmp_path = setup
    base = exp.cfg.model_copy(
        update={
            "name": "cv_test",
            "optimizers": ["adam", "sgd_momentum"],
            "seeds": [0, 1],
            "fusion": "simplex_temperature",
            "cv_folds": 3,
            "eval": EvalConfig(in_distribution="faceforensics_pp"),
        }
    )
    paths = write_folds(base, 3, 0, tmp_path / "cv_manifests", tmp_path / "cv_configs")
    tested: list[str] = []
    for k, path in enumerate(paths):
        rows = list(
            csv.DictReader(open(tmp_path / "cv_manifests" / f"f{k}" / "faceforensics_pp.csv"))
        )
        ids = {
            s: {r["identity_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")
        }
        assert not (
            ids["train"] & ids["test"] or ids["train"] & ids["val"] or ids["val"] & ids["test"]
        )
        assert {r["val_role"] for r in rows if r["split"] == "val"} == {"tune", "fusion"}
        tested += sorted(ids["test"])
        fold = Experiment(
            conn,
            config.load_experiment(path),
            path,
            exp.optimizers,
            config.load_models(root / "configs"),
            log=lambda _: None,
        )
        fold.sweep()
    assert len(tested) == len(set(tested)) == 90  # every identity tested exactly once
    assert (
        count(
            conn,
            "SELECT COUNT(*) FROM fusion_solves WHERE formulation = 'simplex_temperature'"
            " AND temperature > 0 AND kkt_residual < 1e-6",
        )
        == 3 * 2 * 2
    )
    assert count(conn, "SELECT COUNT(*) FROM fusion_weights WHERE contribution IS NULL") == 0
    text = write_cv_report(conn, "cv_test", tmp_path / "cv_stats").read_text()
    assert "Paired comparisons:" in text and "mean contribution" in text


def test_calibrated_fusion_persists_and_cannot_refit_test(setup):
    exp, conn, _ = setup
    exp.cfg.stream_calibration = True
    exp.cfg.optimizers = ["adam"]
    exp.cfg.seeds = [0]
    exp.sweep()
    row = conn.execute("SELECT run_id, calibration_json FROM fusion_solves").fetchone()
    cal = json.loads(row["calibration_json"])
    assert len(cal["scales"]) == len(exp.mods)
    assert np.all(np.array(cal["nll_after"]) <= np.array(cal["nll_before"]) + 1e-9)
    exp.evaluate(row["run_id"])
    with pytest.raises(TestSplitAlreadyWritten):
        exp.fuse(row["run_id"])


def test_configured_in_distribution_target_is_used(setup):
    exp, conn, _ = setup
    exp.cfg.eval = EvalConfig(in_distribution="celeb_df_v2")
    exp.cfg.optimizers, exp.cfg.seeds = ["adam"], [0]
    exp.sweep()
    names = conn.execute(
        "SELECT DISTINCT d.name FROM evaluations e JOIN datasets d ON d.id=e.eval_dataset_id "
        "WHERE split='test' AND protocol='in_distribution'"
    ).fetchall()
    assert [r[0] for r in names] == ["celeb_df_v2"]


def test_train_rejects_single_class_validation(setup):
    from optlab.train.data import Split

    exp, _, _ = setup
    val = exp.val_tune
    invalid = Split(val.x, val.y * 0, val.video_ids)
    with pytest.raises(ValueError, match="val_tune.*both"):
        train(
            exp.train_split,
            invalid,
            exp.optimizers["adam"],
            {"lr": 0.01},
            0,
            1,
            64,
            2,
            None,
            exp.model_extra,
        )


def test_experiment_rejects_other_experiments_run(setup):
    exp, conn, _ = setup
    exp.cfg.optimizers, exp.cfg.seeds = ["adam"], [0]
    exp.sweep()
    run = conn.execute("SELECT id FROM runs").fetchone()[0]
    exp.exp_id += 100
    with pytest.raises(ValueError, match="does not belong"):
        exp.evaluate(run)


@pytest.mark.parametrize("kind", ["concat_mlp", "svm_head"])
def test_joint_head_experiment_pipeline(setup, root, kind):
    exp, conn, tmp = setup
    exp.cfg.name = "joint_" + kind
    exp.cfg.model = "clip_vit_b16__freq__" + kind
    exp.cfg.optimizers = ["adam"]
    exp.cfg.seeds = [0]
    other = Experiment(
        conn,
        exp.cfg,
        tmp / "joint.yaml",
        exp.optimizers,
        config.load_models(root / "configs"),
        log=lambda _: None,
    )
    other.sweep()
    row = conn.execute("SELECT modality,weight FROM fusion_weights").fetchone()
    assert tuple(row) == ("joint", 1.0)
    assert count(conn, "SELECT COUNT(*) FROM evaluations") == 6


@pytest.mark.parametrize("kind", ["weighted_sum", "concat_mlp", "svm_head"])
def test_portable_bundle_exports_completed_run(setup, root, kind):
    from optlab.inference import export_model, load_bundle, predict_features

    exp, conn, tmp = setup
    exp.cfg.optimizers, exp.cfg.seeds = ["adam"], [0]
    exp.cfg.model = "clip_vit_b16__freq__" + kind
    exp = Experiment(
        conn,
        exp.cfg,
        exp.config_path,
        exp.optimizers,
        config.load_models(root / "configs"),
        log=lambda _: None,
    )
    exp.config_path.write_text(exp.cfg.model_dump_json())
    exp.sweep()
    run = conn.execute("SELECT id FROM runs").fetchone()[0]
    out = tmp / "demo.pt"
    export_model(tmp / "r.db", run, out, root / "configs")
    result = predict_features(
        load_bundle(out), {m: x[:3].numpy() for m, x in exp.val_tune.x.items()}
    )
    assert result["source_run"]["id"] == run
    assert result["frames_analyzed"] == 3


def test_hybrid_report_keeps_evaluation_datasets_separate(setup):
    from optlab.export.queries import hybrid

    exp, conn, _ = setup
    exp.cfg.optimizers, exp.cfg.seeds = ["adam"], [0]
    exp.sweep()
    conn.execute("UPDATE experiments SET phase='ablation' WHERE id=?", (exp.exp_id,))
    conn.commit()
    rows = hybrid(conn)["studies"][0]["rows"]
    assert len({r["dataset"] for r in rows}) == 2
    assert all(r["n_seeds"] == 1 for r in rows)
