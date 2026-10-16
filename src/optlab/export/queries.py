"""Read-only views over the results DB, shared by the API and the static snapshot.

Every function returns plain JSON-able data. The API serves `/api/<path>` and
`optlab export-snapshot` writes the same object to `<out>/<path>.json`, so the dashboard
reads identical shapes in both data modes (TRD 13). NaN/inf become null.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from optlab.config import ToyExperimentConfig
from optlab.db.connection import schema_version
from optlab.eval.stats import bootstrap_ci, pairwise

FOLD = re.compile(r"^(?P<base>.+)_f(?P<k>\d+)$")


def clean(x: Any) -> Any:
    """Recursively make data JSON-safe: numpy scalars to Python, NaN/inf to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _rows(conn: sqlite3.Connection, sql: str, *args: Any) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def _std(v: list[float]) -> float:
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


# --- catalogue -------------------------------------------------------------------------------


def experiments(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(
        conn,
        "SELECT e.id, e.name, e.phase, e.description, e.hypothesis, e.git_commit, e.created_at,"
        " COUNT(r.id) AS n_runs, COALESCE(SUM(r.status = 'completed'), 0) AS n_completed,"
        " COALESCE(SUM(r.status = 'diverged'), 0) AS n_diverged"
        " FROM experiments e LEFT JOIN runs r ON r.experiment_id = e.id"
        " GROUP BY e.id ORDER BY e.id",
    )
    names = {r["name"] for r in rows}
    superseded: set[str] = set()
    for r in rows:
        text = r["description"] or ""
        if "upersedes" in text:
            after = text[text.index("upersedes") :]
            superseded |= {n for n in names if re.search(rf"\b{re.escape(n)}\b", after)}
    for r in rows:
        m = FOLD.match(r["name"])
        r["cv_base"] = m["base"] if m else None
        r["fold"] = int(m["k"]) if m else None
        r["superseded"] = r["name"] in superseded
    return rows


def meta(conn: sqlite3.Connection) -> dict:
    exps = experiments(conn)
    return {
        "schema_version": schema_version(conn),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "n_runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        "n_toy_runs": conn.execute("SELECT COUNT(*) FROM toy_runs").fetchone()[0],
        "cv_bases": sorted({e["cv_base"] for e in exps if e["cv_base"]}),
        "latest_commit": next((e["git_commit"] for e in reversed(exps) if e["git_commit"]), None),
    }


def optimizers(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(conn, "SELECT * FROM optimizers ORDER BY id")
    for r in rows:
        r["search_space"] = json.loads(r.pop("default_search_space_json") or "{}")
    return rows


# --- one experiment --------------------------------------------------------------------------


def leaderboard(conn: sqlite3.Connection, exp_id: int) -> dict:
    """Test-split rows per (dataset, protocol, granularity) with per-seed values, CI and the
    pairwise comparisons used for the "≈ top" badge."""
    evals = _rows(
        conn,
        "SELECT o.name AS optimizer, o.family, d.name AS dataset, e.protocol, e.granularity,"
        " e.auc, e.eer, e.accuracy, e.f1, e.average_precision, r.seed, r.wall_clock_sec"
        " FROM evaluations e JOIN runs r ON r.id = e.run_id"
        " JOIN optimizers o ON o.id = r.optimizer_id JOIN datasets d ON d.id = e.eval_dataset_id"
        " WHERE r.experiment_id = ? AND r.status = 'completed' AND e.split = 'test'",
        exp_id,
    )
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in evals:
        groups[(e["dataset"], e["protocol"], e["granularity"])][e["optimizer"]].append(e)
    tables = []
    for (dataset, protocol, gran), by_opt in sorted(groups.items()):
        rows = []
        for o, es in by_opt.items():
            auc = [e["auc"] for e in sorted(es, key=lambda e: e["seed"])]
            rows.append(
                {
                    "optimizer": o,
                    "family": es[0]["family"],
                    "n_seeds": len(auc),
                    "values": auc,
                    "auc_mean": float(np.mean(auc)),
                    "auc_std": _std(auc),
                    "auc_ci95": bootstrap_ci(auc),
                    "eer_mean": float(np.mean([e["eer"] for e in es])),
                    "accuracy_mean": float(np.mean([e["accuracy"] for e in es])),
                    "f1_mean": float(np.mean([e["f1"] for e in es])),
                    "ap_mean": float(np.mean([e["average_precision"] for e in es])),
                    "wall_clock_mean_sec": float(np.mean([e["wall_clock_sec"] for e in es])),
                }
            )
        rows.sort(key=lambda r: -r["auc_mean"])
        eligible = {r["optimizer"]: r["values"] for r in rows if r["n_seeds"] >= 2}
        tables.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "granularity": gran,
                "rows": rows,
                "pairwise": pairwise(eligible) if len(eligible) >= 2 else [],
            }
        )
    return {"experiment_id": exp_id, "tables": tables}


def convergence(conn: sqlite3.Connection, exp_id: int) -> dict:
    """Per optimizer and epoch: mean and sd across completed seeds (n per epoch is reported,
    since early stopping ends runs at different epochs). Diverged runs are listed, not averaged."""
    runs = _rows(
        conn,
        "SELECT r.id, r.seed, r.status, o.name AS optimizer, o.family FROM runs r"
        " JOIN optimizers o ON o.id = r.optimizer_id"
        " WHERE r.experiment_id = ? ORDER BY o.id, r.seed",
        exp_id,
    )
    out: dict[str, dict] = {}
    for run in runs:
        o = out.setdefault(
            run["optimizer"],
            {
                "family": run["family"],
                "seeds": [],
                "diverged": [],
                "failed": [],
                "curves": defaultdict(lambda: defaultdict(list)),
            },
        )
        if run["status"] != "completed":
            o["diverged" if run["status"] == "diverged" else "failed"].append(run["seed"])
            continue
        o["seeds"].append(run["seed"])
        for ep in _rows(
            conn, "SELECT * FROM run_epochs WHERE run_id = ? ORDER BY epoch", run["id"]
        ):
            for m in ("val_auc", "val_loss", "train_loss", "grad_norm"):
                if ep[m] is not None:
                    o["curves"][m][ep["epoch"]].append(ep[m])
    result = {}
    for name, o in out.items():
        metrics = {}
        for m, by_epoch in o["curves"].items():
            epochs = sorted(by_epoch)
            metrics[m] = {
                "epoch": epochs,
                "mean": [float(np.mean(by_epoch[e])) for e in epochs],
                "std": [_std(by_epoch[e]) for e in epochs],
                "min": [float(np.min(by_epoch[e])) for e in epochs],
                "max": [float(np.max(by_epoch[e])) for e in epochs],
                "n": [len(by_epoch[e]) for e in epochs],
            }
        result[name] = {
            "family": o["family"],
            "seeds": o["seeds"],
            "diverged": o["diverged"],
            "failed": o["failed"],
            "metrics": metrics,
        }
    return {"experiment_id": exp_id, "optimizers": result}


def generalisation(conn: sqlite3.Connection, exp_id: int) -> dict:
    rows = _rows(
        conn,
        "SELECT o.name AS optimizer, o.family, g.seed, ind.auc AS ind,"
        " xd.auc AS xd, ind.auc - xd.auc AS gap, di.name AS in_dataset, dx.name AS dataset"
        " FROM runs g JOIN optimizers o ON o.id = g.optimizer_id"
        " JOIN evaluations ind ON ind.run_id = g.id AND ind.split = 'test'"
        " AND ind.protocol = 'in_distribution' AND ind.granularity = 'video'"
        " JOIN evaluations xd ON xd.run_id = g.id AND xd.split = 'test'"
        " AND xd.protocol = 'cross_dataset' AND xd.granularity = 'video'"
        " JOIN datasets di ON di.id = ind.eval_dataset_id"
        " JOIN datasets dx ON dx.id = xd.eval_dataset_id"
        " WHERE g.experiment_id = ? AND g.status = 'completed'",
        exp_id,
    )
    by: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["optimizer"], r["in_dataset"], r["dataset"])].append(r)
    out = []
    for (o, in_dataset, dataset), rs in by.items():
        ind, xd, gap = ([r[k] for r in rs] for k in ("ind", "xd", "gap"))
        out.append(
            {
                "optimizer": o,
                "in_dataset": in_dataset,
                "dataset": dataset,
                "family": rs[0]["family"],
                "n": len(rs),
                "in_mean": float(np.mean(ind)),
                "in_std": _std(ind),
                "cross_mean": float(np.mean(xd)),
                "cross_std": _std(xd),
                "gap_mean": float(np.mean(gap)),
                "gap_std": _std(gap),
                "gap_ci95": bootstrap_ci(gap),
                "seeds": [r["seed"] for r in rs],
            }
        )
    return {"experiment_id": exp_id, "rows": sorted(out, key=lambda r: r["gap_mean"])}


def tuning(conn: sqlite3.Connection, exp_id: int) -> dict:
    rows = _rows(
        conn,
        "SELECT o.name AS optimizer, o.family, t.search_method, t.trial_index, t.seed,"
        " t.hyperparams_json, t.epochs_run, t.val_auc, t.val_loss, t.diverged, t.is_selected"
        " FROM tuning_trials t JOIN optimizers o ON o.id = t.optimizer_id"
        " WHERE t.experiment_id = ? ORDER BY o.id, t.trial_index",
        exp_id,
    )
    by: dict[str, dict] = {}
    for r in rows:
        o = by.setdefault(
            r["optimizer"], {"family": r["family"], "method": r["search_method"], "trials": []}
        )
        o["trials"].append(
            {
                "index": r["trial_index"],
                "hparams": json.loads(r["hyperparams_json"]),
                "epochs_run": r["epochs_run"],
                "val_auc": r["val_auc"],
                "val_loss": r["val_loss"],
                "diverged": bool(r["diverged"]),
                "selected": bool(r["is_selected"]),
            }
        )
    return {"experiment_id": exp_id, "optimizers": by}


def fusion(conn: sqlite3.Connection, exp_id: int) -> dict:
    solves = _rows(
        conn,
        "SELECT f.*, r.seed, o.name AS optimizer, o.family FROM fusion_solves f"
        " JOIN runs r ON r.id = f.run_id JOIN optimizers o ON o.id = r.optimizer_id"
        " WHERE r.experiment_id = ? ORDER BY o.id, r.seed",
        exp_id,
    )
    weights = defaultdict(dict)
    contribution = defaultdict(dict)
    for w in _rows(
        conn,
        "SELECT w.* FROM fusion_weights w JOIN runs r ON r.id = w.run_id WHERE r.experiment_id = ?",
        exp_id,
    ):
        weights[w["run_id"]][w["modality"]] = w["weight"]
        contribution[w["run_id"]][w["modality"]] = w["contribution"]
    for s in solves:
        s["active_set"] = json.loads(s.pop("active_set_json") or "[]")
        s["calibration"] = json.loads(s.pop("calibration_json", None) or "null")
        s["weights"] = weights[s["run_id"]]
        s["contribution"] = contribution[s["run_id"]]
    return {"experiment_id": exp_id, "runs": solves}


def runs(conn: sqlite3.Connection, exp_id: int) -> list[dict]:
    return _rows(
        conn,
        "SELECT s.run_id AS id, s.optimizer_name AS optimizer, s.optimizer_family AS family,"
        " s.seed, s.status, s.best_epoch, s.early_stopped, s.wall_clock_sec, s.best_val_auc,"
        " s.epochs_logged FROM v_run_summary s WHERE s.experiment_id = ? ORDER BY s.run_id",
        exp_id,
    )


def run_detail(conn: sqlite3.Connection, run_id: int) -> dict | None:
    run = conn.execute(
        "SELECT r.*, o.name AS optimizer, o.family, x.name AS experiment, m.name AS model"
        " FROM runs r JOIN optimizers o ON o.id = r.optimizer_id"
        " JOIN experiments x ON x.id = r.experiment_id"
        " JOIN model_configs m ON m.id = r.model_config_id"
        " WHERE r.id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        return None
    out = dict(run)
    out["hyperparams"] = json.loads(out.pop("hyperparams_json"))
    out["library_versions"] = json.loads(out.pop("library_versions_json") or "null")
    out["epochs"] = _rows(conn, "SELECT * FROM run_epochs WHERE run_id = ? ORDER BY epoch", run_id)
    out["evaluations"] = _rows(
        conn,
        "SELECT d.name AS dataset, e.split, e.protocol, e.granularity, e.checkpoint_epoch,"
        " e.n_samples, e.auc, e.eer, e.accuracy, e.f1, e.average_precision, e.force_rewritten"
        " FROM evaluations e JOIN datasets d ON d.id = e.eval_dataset_id WHERE e.run_id = ?"
        " ORDER BY e.split DESC, d.name, e.granularity",
        run_id,
    )
    f = fusion_of_run(conn, run_id)
    out["fusion"] = f
    out["artifacts"] = _rows(
        conn, "SELECT kind, path, sha256 FROM artifacts WHERE run_id = ?", run_id
    )
    return out


def fusion_of_run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    s = conn.execute("SELECT * FROM fusion_solves WHERE run_id = ?", (run_id,)).fetchone()
    if s is None:
        return None
    s = dict(s)
    s["active_set"] = json.loads(s.pop("active_set_json") or "[]")
    ws = _rows(
        conn, "SELECT modality, weight, contribution FROM fusion_weights WHERE run_id = ?", run_id
    )
    s["calibration"] = json.loads(s.pop("calibration_json", None) or "null")
    s["weights"] = {w["modality"]: w["weight"] for w in ws}
    s["contribution"] = {w["modality"]: w["contribution"] for w in ws}
    return s


# --- cross-validation ------------------------------------------------------------------------


def cv(conn: sqlite3.Connection, base: str) -> dict | None:
    from optlab.eval.cv_report import fold_means, paired

    means = fold_means(conn, base)
    if not means:
        return None
    tables = []
    family = {r[0]: r[1] for r in conn.execute("SELECT name, family FROM optimizers")}
    for (dataset, protocol), per_opt in sorted(means.items()):
        rows = []
        for o, folds in per_opt.items():
            v = [folds[k] for k in sorted(folds)]
            rows.append(
                {
                    "optimizer": o,
                    "family": family[o],
                    "n_folds": len(v),
                    "fold_values": v,
                    "auc_mean": float(np.mean(v)),
                    "auc_std": _std(v),
                    "auc_ci95": bootstrap_ci(v),
                }
            )
        tables.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "rows": sorted(rows, key=lambda r: -r["auc_mean"]),
                "pairwise": paired(per_opt),
            }
        )
    contrib = _rows(
        conn,
        "SELECT w.modality, AVG(w.weight) AS weight, AVG(w.contribution) AS contribution,"
        " COUNT(*) AS n FROM fusion_weights w JOIN runs r ON r.id = w.run_id"
        " JOIN experiments x ON x.id = r.experiment_id WHERE x.name GLOB ? GROUP BY w.modality",
        f"{base}_f[0-9]*",
    )
    return {"base": base, "tables": tables, "fusion": contrib}


# --- toy suite -------------------------------------------------------------------------------


def toy(conn: sqlite3.Connection, cfg: ToyExperimentConfig, max_points: int = 400) -> dict:
    """Benchmarks with a contour grid, per-run results, and seed-`tuning_seed` curves and
    (downsampled) trajectories for the 2-D viewer."""
    import torch

    from optlab.toy.functions import make_function, minimizer

    by_name = {b.name: b for b in cfg.benchmarks}
    benchmarks = _rows(conn, "SELECT * FROM toy_benchmarks ORDER BY id")
    runs_ = _rows(
        conn,
        "SELECT r.id, b.name AS benchmark, o.name AS optimizer, o.family, r.seed,"
        " r.hyperparams_json,"
        " r.max_steps, r.tolerance, r.steps_to_tol, r.final_f, r.final_grad_norm, r.trajectory_path"
        " FROM toy_runs r JOIN toy_benchmarks b ON b.id = r.benchmark_id"
        " JOIN optimizers o ON o.id = r.optimizer_id ORDER BY b.id, o.id, r.seed",
    )
    curves: dict = defaultdict(dict)
    paths: dict = defaultdict(dict)
    for r in runs_:
        r["hparams"] = json.loads(r.pop("hyperparams_json"))
        traj = r.pop("trajectory_path")
        if r["seed"] != cfg.tuning_seed:
            continue
        curves[r["benchmark"]][r["optimizer"]] = [
            [s, f]
            for s, f in conn.execute(
                "SELECT step, f_value FROM toy_run_curves WHERE toy_run_id = ? ORDER BY step",
                (r["id"],),
            )
        ]
        if traj and Path(traj).exists():
            x = np.load(traj)["x"]
            if len(x) and x.shape[1] == 2:
                idx = np.unique(
                    np.linspace(0, len(x) - 1, min(len(x), max_points)).round().astype(int)
                )
                paths[r["benchmark"]][r["optimizer"]] = x[idx].tolist()
    for b in benchmarks:
        bc = by_name.get(b["name"])
        b["minimizer"] = minimizer(bc).tolist() if bc else None
        b["grid"] = None
        if bc and bc.dim == 2:
            pts = np.concatenate(
                [np.array(p) for p in paths[b["name"]].values()] + [np.array([b["minimizer"]])]
            )
            lo = np.maximum(pts.min(0), bc.start_low - 1)
            hi = np.minimum(pts.max(0), bc.start_high + 1)
            pad = 0.1 * (hi - lo) + 0.05
            xs, ys = (
                np.linspace(lo[0] - pad[0], hi[0] + pad[0], 80),
                np.linspace(lo[1] - pad[1], hi[1] + pad[1], 80),
            )
            fn = make_function(bc)
            with torch.no_grad():
                z = [[float(fn(torch.tensor([x, y], dtype=torch.float64))) for x in xs] for y in ys]
            b["grid"] = {"x": xs.tolist(), "y": ys.tolist(), "z": z, "log": bc.kind != "rastrigin"}
    return {
        "experiment": cfg.name,
        "tuning_seed": cfg.tuning_seed,
        "benchmarks": benchmarks,
        "runs": runs_,
        "curves": curves,
        "trajectories": paths,
    }


def hybrid(conn: sqlite3.Connection) -> dict:
    catalogue = _rows(
        conn,
        "SELECT o.id,o.name,o.parent_optimizer_id,o.ablation_note,p.name AS parent FROM "
        "optimizers o LEFT JOIN optimizers p ON p.id=o.parent_optimizer_id WHERE "
        "o.family='hybrid' ORDER BY o.id",
    )
    summary = _rows(
        conn,
        "SELECT x.name AS experiment,m.name AS model,o.name AS optimizer,r.status,COUNT(*) "
        "AS n,AVG(r.wall_clock_sec) AS seconds FROM runs r JOIN experiments x ON "
        "x.id=r.experiment_id JOIN optimizers o ON o.id=r.optimizer_id JOIN model_configs "
        "m ON m.id=r.model_config_id WHERE x.phase IN ('hybrid','ablation') GROUP BY "
        "x.id,m.id,o.id,r.status ORDER BY x.id,m.id,o.id",
    )
    studies = []
    for exp in conn.execute(
        "SELECT DISTINCT x.id,x.name FROM experiments x JOIN runs r ON r.experiment_id=x.id "
        "WHERE x.phase IN ('hybrid','ablation') AND x.name NOT GLOB '*_f[0-9]*'"
    ):
        # Include model identity: fine-tuning controls and thawed variants share one experiment.
        evaluations = _rows(
            conn,
            "SELECT m.name AS model,o.name AS optimizer,d.name AS "
            "dataset,e.protocol,r.seed,e.auc,e.eer,r.wall_clock_sec FROM evaluations e JOIN "
            "runs r ON r.id=e.run_id JOIN datasets d ON d.id=e.eval_dataset_id JOIN "
            "model_configs m "
            "ON m.id=r.model_config_id JOIN optimizers o ON o.id=r.optimizer_id WHERE "
            "r.experiment_id=? AND r.status='completed' AND e.split='test' AND "
            "e.granularity='video'",
            exp["id"],
        )
        groups = defaultdict(list)
        for r in evaluations:
            groups[(r["model"], r["optimizer"], r["dataset"], r["protocol"])].append(r)
        rows = []
        for (model, optimizer, dataset, protocol), values in groups.items():
            auc = [r["auc"] for r in values]
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "protocol": protocol,
                    "optimizer": optimizer,
                    "n_seeds": len(auc),
                    "auc_mean": float(np.mean(auc)),
                    "auc_ci95": bootstrap_ci(auc),
                    "seconds": float(np.mean([r["wall_clock_sec"] for r in values])),
                }
            )
        studies.append({"experiment": exp["name"], "rows": rows})
    return {
        "design_commit": "5155285",
        "catalogue": catalogue,
        "coverage": summary,
        "cross_validation": cv(conn, "hybrid_timit_cv"),
        "studies": studies,
        "interpretation": "Exploratory TIMIT evaluation. A frozen design does not make "
        "previously observed datasets confirmatory evidence.",
    }
