"""`optlab export-snapshot`: the API's responses as static JSON (TRD 13, FR-P3).

`/api/<path>` is written to `<out>/<path>.json`, so the dashboard's snapshot mode reads the
same shapes without a backend and can be hosted as a static site.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from optlab.config import ToyExperimentConfig
from optlab.db.connection import require_current_schema
from optlab.export import queries as q


def _write(out: Path, path: str, data: Any) -> None:
    target = out / f"{path}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(q.clean(data), separators=(",", ":")))


def _build_snapshot(conn: sqlite3.Connection, toy_cfg: ToyExperimentConfig, out: Path) -> int:
    meta = q.meta(conn)
    meta["mode"] = "snapshot"
    _write(out, "meta", meta)
    exps = q.experiments(conn)
    _write(out, "experiments", exps)
    _write(out, "optimizers", q.optimizers(conn))
    _write(out, "toy", q.toy(conn, toy_cfg))
    _write(out, "hybrid", q.hybrid(conn))
    for base in meta["cv_bases"]:
        _write(out, f"cv/{base}", q.cv(conn, base))
    count = 0
    for e in exps:
        for view in ("leaderboard", "convergence", "generalisation", "tuning", "fusion", "runs"):
            _write(out, f"experiments/{e['id']}/{view}", getattr(q, view)(conn, e["id"]))
        for (run_id,) in conn.execute("SELECT id FROM runs WHERE experiment_id = ?", (e["id"],)):
            _write(out, f"runs/{run_id}", q.run_detail(conn, run_id))
            count += 1
    return count


def export_snapshot(conn: sqlite3.Connection, toy_cfg: ToyExperimentConfig, out: Path) -> int:
    """Build from a consistent database copy; preserve the last export on any build failure.

    The destination must be an empty directory or an existing OptLab snapshot. Staging
    lives on the same filesystem so publishing uses directory renames, not partial copies.
    """
    out = Path(out)
    require_current_schema(conn)
    if out.exists() and (
        not out.is_dir() or (any(out.iterdir()) and not (out / "meta.json").is_file())
    ):
        raise ValueError("snapshot destination must be empty or contain an existing meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".optlab-export-", dir=out.parent) as tmp:
        staging, previous = Path(tmp) / "data", Path(tmp) / "previous"
        snapshot = sqlite3.connect(":memory:")
        snapshot.row_factory = sqlite3.Row
        try:
            conn.backup(snapshot)
            count = _build_snapshot(snapshot, toy_cfg, staging)
        finally:
            snapshot.close()
        if out.exists():
            out.rename(previous)
        try:
            staging.rename(out)
        except OSError:
            if previous.exists():
                previous.rename(out)
            raise
    return count
