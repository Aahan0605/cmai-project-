"""Merge partial results databases (e.g. from separate Colab/Kaggle sessions).

Rows are matched on their natural UNIQUE keys, never on ids, and foreign keys are remapped.
A run or toy run present in both inputs is kept once if identical; two *finished* versions
that differ raise `MergeConflict` instead of silently overwriting (TRD 11).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from optlab.db.connection import atomic, connect, schema_version
from optlab.db.repository import insert

FINISHED = {"completed", "diverged", "failed"}
NAMED = ["datasets", "model_configs", "optimizers", "experiments", "toy_benchmarks"]


class MergeConflict(RuntimeError):
    pass


def _copy(
    dst: sqlite3.Connection, table: str, row: sqlite3.Row, overrides: Mapping[str, Any]
) -> int:
    data = {k: row[k] for k in row.keys() if k != "id"}
    data.update(overrides)
    return insert(dst, table, data)


def _rows(src: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return src.execute(sql, params).fetchall()


def _same(a: sqlite3.Row, b: sqlite3.Row, fields: list[str]) -> bool:
    return all(a[f] == b[f] for f in fields)


def merge_into(dst: sqlite3.Connection, src: sqlite3.Connection) -> dict[str, int]:
    """Merge `src` into `dst` in one transaction. Returns counts of inserted rows."""
    with atomic(dst):
        return _merge_into(dst, src)


def _merge_into(dst: sqlite3.Connection, src: sqlite3.Connection) -> dict[str, int]:
    if schema_version(dst) != schema_version(src):
        raise MergeConflict("schema versions differ; migrate both databases first")
    ids: dict[str, dict[int, int]] = {t: {} for t in [*NAMED, "tuning_trials"]}
    added = {"runs": 0, "toy_runs": 0, "tuning_trials": 0}

    # 1. Named catalogues. Optimizer parents first so ablation children can be remapped.
    for table in NAMED:
        order = " ORDER BY parent_optimizer_id IS NOT NULL" if table == "optimizers" else ""
        for row in _rows(src, f"SELECT * FROM {table}{order}"):
            hit = dst.execute(f"SELECT * FROM {table} WHERE name = ?", (row["name"],)).fetchone()
            if hit:
                fields = [
                    k
                    for k in row.keys()
                    if k
                    not in (
                        "id",
                        "created_at",
                        "git_commit",
                        "description",
                        "access_notes",
                        "parent_optimizer_id",
                        "num_real_videos",
                        "num_fake_videos",
                    )
                ]
                if not _same(hit, row, fields):
                    raise MergeConflict(
                        f"catalogue {table}/{row['name']} has different definitions"
                    )
                ids[table][row["id"]] = hit[0]
                continue
            over = {}
            if table == "optimizers" and row["parent_optimizer_id"] is not None:
                over["parent_optimizer_id"] = ids["optimizers"][row["parent_optimizer_id"]]
            ids[table][row["id"]] = _copy(dst, table, row, over)

    ds, opt, mc, exp = (
        ids["datasets"],
        ids["optimizers"],
        ids["model_configs"],
        ids["experiments"],
    )

    for row in _rows(src, "SELECT * FROM dataset_splits"):
        dst.execute(
            "INSERT OR IGNORE INTO dataset_splits (dataset_id, split, manifest_path, num_videos,"
            " split_seed, identity_disjoint) VALUES (?, ?, ?, ?, ?, ?)",
            (
                ds[row["dataset_id"]],
                row["split"],
                row["manifest_path"],
                row["num_videos"],
                row["split_seed"],
                row["identity_disjoint"],
            ),
        )

    # 2. Tuning trials.
    for row in _rows(src, "SELECT * FROM tuning_trials"):
        over = {
            "experiment_id": exp[row["experiment_id"]],
            "optimizer_id": opt[row["optimizer_id"]],
            "model_config_id": mc[row["model_config_id"]],
        }
        hit = dst.execute(
            "SELECT * FROM tuning_trials WHERE experiment_id = ? AND optimizer_id = ?"
            " AND model_config_id = ? AND search_method = ? AND trial_index = ?",
            (*over.values(), row["search_method"], row["trial_index"]),
        ).fetchone()
        if hit:
            if not _same(hit, row, ["seed", "hyperparams_json", "val_auc", "diverged"]):
                raise MergeConflict(f"tuning trial differs: {dict(row)}")
            ids["tuning_trials"][row["id"]] = hit["id"]
        else:
            ids["tuning_trials"][row["id"]] = _copy(dst, "tuning_trials", row, over)
            added["tuning_trials"] += 1

    # 3. Runs and their children.
    for row in _rows(src, "SELECT * FROM runs"):
        over = {
            "experiment_id": exp[row["experiment_id"]],
            "optimizer_id": opt[row["optimizer_id"]],
            "model_config_id": mc[row["model_config_id"]],
            "train_dataset_id": ds[row["train_dataset_id"]],
            "tuning_trial_id": ids["tuning_trials"].get(row["tuning_trial_id"]),
        }
        hit = dst.execute(
            "SELECT * FROM runs WHERE experiment_id = ? AND optimizer_id = ?"
            " AND model_config_id = ? AND seed = ?",
            (*list(over.values())[:3], row["seed"]),
        ).fetchone()
        if hit:
            if row["status"] not in FINISHED:
                continue
            if hit["status"] in FINISHED:
                fields = ["status", "hyperparams_json", "best_epoch", "finished_at"]
                if _same(hit, row, fields) and _same_run_children(dst, src, hit["id"], row["id"]):
                    continue
                raise MergeConflict(
                    f"run (experiment={row['experiment_id']}, optimizer={row['optimizer_id']},"
                    f" seed={row['seed']}) finished differently in both databases"
                )
            dst.execute("DELETE FROM runs WHERE id = ?", (hit["id"],))  # cascades children
        new_id = _copy(dst, "runs", row, over)
        added["runs"] += 1
        for child in ["run_epochs", "fusion_weights", "fusion_solves"]:
            for c in _rows(src, f"SELECT * FROM {child} WHERE run_id = ?", (row["id"],)):
                _copy(dst, child, c, {"run_id": new_id})
        for c in _rows(src, "SELECT * FROM evaluations WHERE run_id = ?", (row["id"],)):
            _copy(
                dst,
                "evaluations",
                c,
                {"run_id": new_id, "eval_dataset_id": ds[c["eval_dataset_id"]]},
            )
        for c in _rows(
            src, "SELECT * FROM artifacts WHERE run_id = ? AND toy_run_id IS NULL", (row["id"],)
        ):
            _copy(dst, "artifacts", c, {"run_id": new_id})

    # 4. Toy runs (no status: any existing row is finished).
    for row in _rows(src, "SELECT * FROM toy_runs"):
        over = {
            "benchmark_id": ids["toy_benchmarks"][row["benchmark_id"]],
            "optimizer_id": opt[row["optimizer_id"]],
        }
        hit = dst.execute(
            "SELECT * FROM toy_runs WHERE benchmark_id = ? AND optimizer_id = ? AND seed = ?",
            (*over.values(), row["seed"]),
        ).fetchone()
        if hit:
            if _same(hit, row, ["hyperparams_json", "steps_to_tol", "final_f"]):
                continue
            raise MergeConflict(f"toy run differs: benchmark/optimizer/seed {tuple(over.values())}")
        new_id = _copy(dst, "toy_runs", row, over)
        added["toy_runs"] += 1
        for c in _rows(src, "SELECT * FROM toy_run_curves WHERE toy_run_id = ?", (row["id"],)):
            _copy(dst, "toy_run_curves", c, {"toy_run_id": new_id})
        for c in _rows(
            src, "SELECT * FROM artifacts WHERE toy_run_id = ? AND run_id IS NULL", (row["id"],)
        ):
            _copy(dst, "artifacts", c, {"toy_run_id": new_id})
    return added


def _same_run_children(dst, src, dst_id: int, src_id: int) -> bool:
    """Compare recorded results on natural keys, including evaluation dataset names.

    Artifact paths can differ between machines; their content hashes, when present,
    identify checkpoints. Evaluation IDs and timestamps are not scientific results.
    """
    for table in ("run_epochs", "fusion_weights", "fusion_solves", "evaluations", "artifacts"):
        records = []
        for conn, run_id in ((dst, dst_id), (src, src_id)):
            rows = []
            for record in conn.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,)):
                data = {
                    k: record[k]
                    for k in record.keys()
                    if k not in ("id", "run_id", "created_at", "evaluated_at", "solved_at")
                }
                if table == "evaluations":
                    data["eval_dataset_id"] = conn.execute(
                        "SELECT name FROM datasets WHERE id = ?", (data["eval_dataset_id"],)
                    ).fetchone()[0]
                if table == "artifacts":
                    data.pop("path", None)
                rows.append(data)
            records.append(sorted(rows, key=lambda r: repr(sorted(r.items()))))
        if records[0] != records[1]:
            return False
    return True


def merge_databases(inputs: list[str | Path], out: str | Path) -> dict[str, int]:
    """Write `out` = inputs[0] merged with each later input. Refuses to overwrite `out`."""
    if Path(out).exists():
        raise FileExistsError(f"{out} exists; choose a new output path")
    if len(inputs) < 2:
        raise ValueError("need at least two databases to merge")
    dst = connect(out)
    totals = {"runs": 0, "toy_runs": 0, "tuning_trials": 0}
    try:
        first = connect(inputs[0], readonly=True)
        try:
            first.backup(dst)
        finally:
            first.close()
        for path in inputs[1:]:
            src = connect(path, readonly=True)
            try:
                with dst:  # one transaction per input: all or nothing
                    for k, v in merge_into(dst, src).items():
                        totals[k] += v
            finally:
                src.close()
    except Exception:
        dst.close()
        Path(out).unlink()
        raise
    dst.close()
    return totals
