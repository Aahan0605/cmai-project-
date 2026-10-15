import sqlite3
from pathlib import Path

import numpy as np
import pytest

from optlab import config
from optlab.db.connection import connect, init_schema
from optlab.db.merge import MergeConflict, merge_databases
from optlab.db.repository import (
    TestSplitAlreadyWritten,
    columns,
    id_of,
    insert,
    seed_catalogues,
    upsert_experiment,
    write_evaluation,
)


def make_db(root: Path, path: Path) -> sqlite3.Connection:
    conn = connect(path)
    init_schema(conn, root / "backend" / "schema.sql")
    configs = root / "configs"
    seed_catalogues(
        conn,
        config.load_optimizers(configs).values(),
        config.load_models(configs).values(),
        config.load_datasets(configs).values(),
        config.load_toy_experiment(configs / "experiments" / "toy.yaml").benchmarks,
    )
    upsert_experiment(conn, "baseline_test", "baseline", None, "h", None)
    conn.commit()
    return conn


def add_run(conn: sqlite3.Connection, seed: int, optimizer: str = "adamw", **kw) -> int:
    row = {
        "experiment_id": id_of(conn, "experiments", "baseline_test"),
        "optimizer_id": id_of(conn, "optimizers", optimizer),
        "model_config_id": id_of(conn, "model_configs", "clip_vit_b16__freq__weighted_sum"),
        "train_dataset_id": id_of(conn, "datasets", "faceforensics_pp"),
        "seed": seed,
        "status": "completed",
        "hyperparams_json": '{"lr": 0.001}',
        "max_epochs": 30,
        "best_epoch": 7,
        "finished_at": "2026-10-12 10:00:00",
        "library_versions_json": '{"torch": "2.14.1"}',
    }
    row.update(kw)
    return insert(conn, "runs", row)


def eval_row(conn: sqlite3.Connection, run_id: int, auc: float, split: str = "test") -> dict:
    return {
        "run_id": run_id,
        "eval_dataset_id": id_of(conn, "datasets", "celeb_df_v2"),
        "split": split,
        "protocol": "cross_dataset",
        "granularity": "video",
        "auc": auc,
        "eer": 1 - auc,
    }


@pytest.fixture
def conn(root: Path, tmp_path: Path) -> sqlite3.Connection:
    return make_db(root, tmp_path / "r.db")


def test_init_is_idempotent(root: Path, tmp_path: Path) -> None:
    make_db(root, tmp_path / "r.db").close()
    c = make_db(root, tmp_path / "r.db")
    assert c.execute("PRAGMA user_version").fetchone()[0] == 4
    assert c.execute("SELECT COUNT(*) FROM optimizers").fetchone()[0] == 12


def test_migration_upgrades_v1_database(root: Path, tmp_path: Path) -> None:
    conn = connect(tmp_path / "v1.db")
    conn.executescript((root / "backend" / "schema.sql").read_text())  # a pre-migration DB
    assert "library_versions_json" not in columns(conn, "runs")
    assert init_schema(conn, root / "backend" / "schema.sql") == 4
    assert "library_versions_json" in columns(conn, "runs")
    assert {"formulation", "temperature", "calibration_json"} <= set(columns(conn, "fusion_solves"))
    assert init_schema(conn, root / "backend" / "schema.sql") == 4  # re-run is a no-op


@pytest.mark.parametrize(
    "table,row",
    [
        ("dataset_splits", {"dataset_id": 1, "split": "holdout", "manifest_path": "m.csv"}),
        ("optimizers", {"name": "x", "family": "magic"}),
        # ablation parent on a non-hybrid optimizer
        ("optimizers", {"name": "y", "family": "adaptive", "parent_optimizer_id": 1}),
        ("artifacts", {"kind": "log", "path": "a.log"}),  # neither run nor toy run
        ("fusion_weights", {"run_id": 999, "modality": "visual", "weight": 0.5}),  # FK
        (
            "toy_runs",
            {
                "benchmark_id": 1,
                "optimizer_id": 1,
                "seed": 0,
                "hyperparams_json": "{}",
                "max_steps": 0,
                "tolerance": 1e-6,
            },
        ),
    ],
)
def test_constraints_reject_bad_rows(conn: sqlite3.Connection, table: str, row: dict) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn, table, row)


def test_constraints_reject_bad_metrics_and_duplicates(conn: sqlite3.Connection) -> None:
    run = add_run(conn, 0)
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn, "evaluations", eval_row(conn, run, 1.2))
    with pytest.raises(sqlite3.IntegrityError):
        add_run(conn, 0)  # same (experiment, optimizer, model, seed)
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn, "fusion_weights", {"run_id": run, "modality": "visual", "weight": -0.1})


def test_leaderboard_variance_matches_numpy(conn: sqlite3.Connection) -> None:
    aucs = np.random.default_rng(0).uniform(0.6, 0.95, size=5)
    for seed, auc in enumerate(aucs):
        write_evaluation(conn, eval_row(conn, add_run(conn, seed), float(auc)))
    # A failed run must not enter the leaderboard.
    write_evaluation(conn, eval_row(conn, add_run(conn, 9, status="failed"), 0.1))
    row = conn.execute("SELECT * FROM v_leaderboard WHERE split = 'test'").fetchone()
    assert row["n_seeds"] == 5
    assert row["auc_mean"] == pytest.approx(aucs.mean(), abs=1e-12)
    assert row["auc_var"] == pytest.approx(np.var(aucs, ddof=1), abs=1e-12)
    assert row["eer_var"] == pytest.approx(np.var(1 - aucs, ddof=1), abs=1e-12)


def test_leaderboard_variance_null_for_single_seed(conn: sqlite3.Connection) -> None:
    write_evaluation(conn, eval_row(conn, add_run(conn, 0), 0.8))
    assert conn.execute("SELECT auc_var FROM v_leaderboard").fetchone()[0] is None


def test_test_split_written_once(conn: sqlite3.Connection) -> None:
    run = add_run(conn, 0)
    write_evaluation(conn, eval_row(conn, run, 0.8))
    with pytest.raises(TestSplitAlreadyWritten):
        write_evaluation(conn, eval_row(conn, run, 0.9))
    write_evaluation(conn, eval_row(conn, run, 0.9), force=True)
    row = conn.execute("SELECT auc, force_rewritten FROM evaluations").fetchone()
    assert tuple(row) == (0.9, 1)
    # Validation rows may be rewritten freely and are never flagged.
    write_evaluation(conn, eval_row(conn, run, 0.7, split="val"))
    write_evaluation(conn, eval_row(conn, run, 0.75, split="val"))
    row = conn.execute("SELECT auc, force_rewritten FROM evaluations WHERE split='val'").fetchone()
    assert tuple(row) == (0.75, 0)


def test_merge_remaps_ids_and_dedupes(root: Path, tmp_path: Path) -> None:
    a = make_db(root, tmp_path / "a.db")
    add_run(a, 0)
    a.commit()
    a.close()
    b = make_db(root, tmp_path / "b.db")
    # Different insertion order so ids differ between the two files.
    upsert_experiment(b, "extra_first", "baseline", None, None, None)
    add_run(b, 0)  # identical to a's finished run
    run = add_run(b, 1, optimizer="sgd")
    insert(b, "run_epochs", {"run_id": run, "epoch": 0, "val_auc": 0.7})
    write_evaluation(b, eval_row(b, run, 0.8))
    b.commit()
    b.close()

    added = merge_databases([tmp_path / "a.db", tmp_path / "b.db"], tmp_path / "m.db")
    assert added["runs"] == 1
    m = connect(tmp_path / "m.db")
    rows = m.execute(
        "SELECT o.name, r.seed, e.name AS exp FROM runs r"
        " JOIN optimizers o ON o.id = r.optimizer_id"
        " JOIN experiments e ON e.id = r.experiment_id ORDER BY r.seed"
    ).fetchall()
    assert [tuple(r) for r in rows] == [("adamw", 0, "baseline_test"), ("sgd", 1, "baseline_test")]
    assert m.execute("SELECT COUNT(*) FROM run_epochs").fetchone()[0] == 1
    assert m.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 1


def test_merge_running_replaced_by_finished(root: Path, tmp_path: Path) -> None:
    a = make_db(root, tmp_path / "a.db")
    add_run(a, 0, status="running", finished_at=None)
    a.commit()
    a.close()
    b = make_db(root, tmp_path / "b.db")
    add_run(b, 0)
    b.commit()
    b.close()
    merge_databases([tmp_path / "a.db", tmp_path / "b.db"], tmp_path / "m.db")
    status = connect(tmp_path / "m.db").execute("SELECT status FROM runs").fetchall()
    assert [s[0] for s in status] == ["completed"]


def test_merge_conflict_raises_and_leaves_no_output(root: Path, tmp_path: Path) -> None:
    for name, epoch in [("a.db", 7), ("b.db", 9)]:
        c = make_db(root, tmp_path / name)
        add_run(c, 0, best_epoch=epoch)
        c.commit()
        c.close()
    with pytest.raises(MergeConflict):
        merge_databases([tmp_path / "a.db", tmp_path / "b.db"], tmp_path / "m.db")
    assert not (tmp_path / "m.db").exists()


def test_readonly_paths_escape_uri_reserved_characters(tmp_path):
    path = tmp_path / "results?#.db"
    conn = connect(path)
    conn.execute("CREATE TABLE example (id INTEGER)")
    conn.commit()
    conn.close()
    readonly = connect(path, readonly=True)
    assert readonly.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 0
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        readonly.execute("INSERT INTO example VALUES (1)")
    readonly.close()


def test_invalid_forced_evaluation_preserves_previous_row(conn):
    run = add_run(conn, 0)
    write_evaluation(conn, eval_row(conn, run, 0.8))
    with pytest.raises(sqlite3.IntegrityError):
        write_evaluation(conn, eval_row(conn, run, 1.5), force=True)
    assert conn.execute("SELECT auc FROM evaluations").fetchone()[0] == 0.8


def test_merge_detects_changed_children_of_otherwise_identical_runs(root, tmp_path):
    for name, auc in [("a.db", 0.8), ("b.db", 0.9)]:
        conn = make_db(root, tmp_path / name)
        write_evaluation(conn, eval_row(conn, add_run(conn, 0), auc))
        conn.commit()
        conn.close()
    with pytest.raises(MergeConflict):
        merge_databases([tmp_path / "a.db", tmp_path / "b.db"], tmp_path / "merged.db")
    assert not (tmp_path / "merged.db").exists()


def test_failed_migration_rolls_back_ddl(conn, tmp_path, root):
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "005_bad.sql").write_text(
        "ALTER TABLE runs ADD COLUMN unexpected TEXT; SELECT * FROM missing_table;"
    )
    with pytest.raises(sqlite3.OperationalError):
        init_schema(conn, root / "backend/schema.sql", migrations)
    assert "unexpected" not in columns(conn, "runs")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 4


def test_generalization_uses_paired_seeds_and_keeps_datasets_separate(conn):
    from optlab.export.queries import generalisation

    ids = {}
    for seed, auc in [(0, 0.9), (1, 0.5)]:
        run = ids[seed] = add_run(conn, seed)
        row = eval_row(conn, run, auc)
        row.update(
            eval_dataset_id=id_of(conn, "datasets", "faceforensics_pp"), protocol="in_distribution"
        )
        write_evaluation(conn, row)
        other = eval_row(conn, run, auc - 0.1)
        other["eval_dataset_id"] = id_of(conn, "datasets", "optlab_timit_lq")
        write_evaluation(conn, other)
    write_evaluation(conn, eval_row(conn, ids[0], 0.3))
    rows = generalisation(conn, id_of(conn, "experiments", "baseline_test"))["rows"]
    by_dataset = {r["dataset"]: r for r in rows}
    celeb = by_dataset["celeb_df_v2"]
    assert celeb["n"] == 1 and celeb["in_mean"] == 0.9
    assert celeb["gap_mean"] == pytest.approx(0.6)
    lq = by_dataset["optlab_timit_lq"]
    assert lq["n"] == 2 and lq["gap_mean"] == pytest.approx(0.1)


def test_merge_rolls_back_catalogue_inserts_before_a_conflict(root, tmp_path):
    from optlab.db.merge import merge_into

    a, b = make_db(root, tmp_path / "a.db"), make_db(root, tmp_path / "b.db")
    add_run(a, 0)
    add_run(b, 0, best_epoch=99)
    upsert_experiment(b, "extra", "baseline", None, None, None)
    a.commit()
    b.commit()
    with pytest.raises(MergeConflict):
        merge_into(a, b)
    assert a.execute("SELECT 1 FROM experiments WHERE name='extra'").fetchone() is None
    a.close()
    b.close()
