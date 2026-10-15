"""Thin repository over the results DB. Parameterized SQL only; column names are checked
against the live schema before they are interpolated."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from optlab.config import (
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    ToyBenchmarkConfig,
)
from optlab.db.connection import atomic


class TestSplitAlreadyWritten(RuntimeError):
    """Raised when a test-split evaluation exists and `force` was not given (TRD 8.2)."""

    __test__ = False  # not a pytest class despite the name


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cols = [r[0] for r in conn.execute("SELECT name FROM pragma_table_info(?)", (table,))]
    if not cols:
        raise ValueError(f"unknown table {table!r}")
    return cols


def _checked(conn: sqlite3.Connection, table: str, keys: Iterable[str]) -> list[str]:
    keys = list(keys)
    unknown = set(keys) - set(columns(conn, table))
    if unknown:
        raise ValueError(f"unknown columns for {table}: {sorted(unknown)}")
    return keys


def insert(conn: sqlite3.Connection, table: str, row: Mapping[str, Any]) -> int:
    keys = _checked(conn, table, row)
    sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})"
    return conn.execute(sql, [row[k] for k in keys]).lastrowid  # type: ignore[return-value]


def insert_many(
    conn: sqlite3.Connection, table: str, keys: Sequence[str], rows: Iterable[Sequence[Any]]
) -> None:
    keys = _checked(conn, table, keys)
    sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})"
    conn.executemany(sql, rows)


def upsert_by_name(conn: sqlite3.Connection, table: str, row: Mapping[str, Any]) -> int:
    """Insert or update a catalogue row keyed by its UNIQUE `name`; returns the id."""
    keys = _checked(conn, table, row)
    updates = ", ".join(f"{k} = excluded.{k}" for k in keys if k != "name")
    conn.execute(
        f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))}) "
        f"ON CONFLICT(name) DO UPDATE SET {updates}",
        [row[k] for k in keys],
    )
    return id_of(conn, table, row["name"])


def id_of(conn: sqlite3.Connection, table: str, name: str) -> int:
    _checked(conn, table, ["name"])
    row = conn.execute(f"SELECT id FROM {table} WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"no {table} row named {name!r}; run `optlab init-db` first")
    return row[0]


def git_commit() -> str | None:
    """Current commit hash, suffixed with -dirty when the work tree has local changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            # Generated outputs under results/ do not change what the code computes.
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


# --- catalogues -----------------------------------------------------------------------------


def seed_catalogues(
    conn: sqlite3.Connection,
    optimizers: Iterable[OptimizerConfig],
    models: Iterable[ModelConfig],
    datasets: Iterable[DatasetConfig],
    toy_benchmarks: Iterable[ToyBenchmarkConfig],
) -> None:
    optimizer_configs = list(optimizers)
    for o in optimizer_configs:
        upsert_by_name(
            conn,
            "optimizers",
            {
                "name": o.name,
                "family": o.family,
                "in_syllabus": int(o.in_syllabus),
                "is_hybrid": int(o.family == "hybrid"),
                "update_rule": o.update_rule,
                "reference": o.reference,
                "default_search_space_json": json.dumps(
                    {k: v.model_dump(exclude_none=True) for k, v in o.search_space.items()}
                ),
            },
        )
    # Parents are seeded before child relationships, independent of YAML filename order.
    for o in optimizer_configs:
        if o.parent_optimizer:
            conn.execute(
                "UPDATE optimizers SET parent_optimizer_id = ?, ablation_note = ? WHERE name = ?",
                (id_of(conn, "optimizers", o.parent_optimizer), o.ablation_note, o.name),
            )
    for m in models:
        extra = m.model_dump(
            exclude={
                "name",
                "visual_backbone",
                "audio_backbone",
                "uses_frequency_branch",
                "fusion_type",
                "backbone_frozen",
                "unfrozen_last_n_layers",
            }
        )
        upsert_by_name(
            conn,
            "model_configs",
            {
                "name": m.name,
                "visual_backbone": m.visual_backbone,
                "audio_backbone": m.audio_backbone,
                "uses_frequency_branch": int(m.uses_frequency_branch),
                "fusion_type": m.fusion_type,
                "backbone_frozen": int(m.backbone_frozen),
                "unfrozen_last_n_layers": m.unfrozen_last_n_layers,
                "config_json": json.dumps(extra),
            },
        )
    for d in datasets:
        upsert_by_name(
            conn,
            "datasets",
            {
                "name": d.name,
                "has_audio": int(d.has_audio),
                "compression": d.compression,
                "access_notes": d.access_notes,
            },
        )
    for b in toy_benchmarks:
        upsert_by_name(
            conn,
            "toy_benchmarks",
            {
                "name": b.name,
                "dimension": b.dim,
                "is_convex": int(b.is_convex),
                "condition_number": b.condition_number,
                "global_min_value": 0.0,
                "description": b.description,
            },
        )
    conn.commit()


def upsert_experiment(
    conn: sqlite3.Connection,
    name: str,
    phase: str,
    description: str | None,
    hypothesis: str | None,
    commit: str | None,
) -> int:
    return upsert_by_name(
        conn,
        "experiments",
        {
            "name": name,
            "phase": phase,
            "description": description,
            "hypothesis": hypothesis,
            "git_commit": commit,
        },
    )


def upsert_dataset_split(
    conn: sqlite3.Connection,
    dataset_id: int,
    split: str,
    manifest_path: str,
    num_videos: int,
    split_seed: int | None,
    identity_disjoint: bool = True,
) -> None:
    conn.execute(
        "INSERT INTO dataset_splits (dataset_id, split, manifest_path, num_videos, split_seed,"
        " identity_disjoint) VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(dataset_id, split) DO UPDATE SET manifest_path = excluded.manifest_path,"
        " num_videos = excluded.num_videos, split_seed = excluded.split_seed,"
        " identity_disjoint = excluded.identity_disjoint",
        (dataset_id, split, manifest_path, num_videos, split_seed, int(identity_disjoint)),
    )


# --- evaluations ----------------------------------------------------------------------------


def write_evaluation(conn: sqlite3.Connection, row: Mapping[str, Any], force: bool = False) -> int:
    """Write one evaluation row.

    `val` rows are freely replaced. A `test` row is written once per
    (run, dataset, split, granularity); rewriting it needs `force=True` and marks the row
    `force_rewritten = 1` so the second look at the test set stays visible.
    """
    key = (row["run_id"], row["eval_dataset_id"], row["split"], row["granularity"])
    existing = conn.execute(
        "SELECT id FROM evaluations WHERE run_id = ? AND eval_dataset_id = ? AND split = ?"
        " AND granularity = ?",
        key,
    ).fetchone()
    data = dict(row)
    with atomic(conn):
        if existing is not None:
            if row["split"] == "test":
                if not force:
                    raise TestSplitAlreadyWritten(
                        f"test evaluation already written for run {key[0]} (dataset {key[1]},"
                        f" {key[3]}); pass --force to overwrite and flag it"
                    )
                data["force_rewritten"] = 1
            conn.execute("DELETE FROM evaluations WHERE id = ?", (existing[0],))
        return insert(conn, "evaluations", data)
