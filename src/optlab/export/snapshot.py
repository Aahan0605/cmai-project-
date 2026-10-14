"""`optlab export-snapshot`: the API's responses as static JSON (TRD 13, FR-P3).

`/api/<path>` is written to `<out>/<path>.json`, so the dashboard's snapshot mode reads the
same shapes without a backend and can be hosted as a static site.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from optlab.config import ToyExperimentConfig
from optlab.export import queries as q


def _write(out: Path, path: str, data: Any) -> None:
    target = out / f"{path}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(q.clean(data), separators=(",", ":")))


def export_snapshot(conn: sqlite3.Connection, toy_cfg: ToyExperimentConfig, out: Path) -> int:
    if out.exists():
        shutil.rmtree(out)  # a stale file must never outlive its run
    meta = q.meta(conn)
    meta["mode"] = "snapshot"
    _write(out, "meta", meta)
    exps = q.experiments(conn)
    _write(out, "experiments", exps)
    _write(out, "optimizers", q.optimizers(conn))
    _write(out, "toy", q.toy(conn, toy_cfg))
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
