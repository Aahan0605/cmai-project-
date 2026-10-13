"""`optlab stats`: experiment tables and plots from the results DB (TRD 8.3).

Writes CSVs, a convergence plot and `summary.md` under the output directory. Every number is
read from the DB; with few seeds, "distinguishable" requires both a bootstrap CI excluding 0
and a Holm-adjusted Welch p < 0.05, and everything else is reported as within noise.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from optlab.eval.stats import (  # noqa: E402
    area_under_curve,
    bootstrap_ci,
    epochs_to_target,
    pairwise,
)
from optlab.toy.report import DASH_MARKER, FAMILY_COLOR  # noqa: E402


def _fmt(x: float | None, digits: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _md(rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def write_stats(conn: sqlite3.Connection, experiment: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    exp = conn.execute("SELECT * FROM experiments WHERE name = ?", (experiment,)).fetchone()
    if exp is None:
        raise SystemExit(f"no experiment named {experiment!r} in the database")
    exp_id = exp["id"]
    family = {r["name"]: r["family"] for r in conn.execute("SELECT name, family FROM optimizers")}
    sections: list[str] = []

    # 1. Coverage and budget.
    runs = conn.execute(
        "SELECT o.name, r.status, COUNT(*) AS n FROM runs r"
        " JOIN optimizers o ON o.id = r.optimizer_id"
        " WHERE r.experiment_id = ? GROUP BY o.name, r.status",
        (exp_id,),
    ).fetchall()
    trials = conn.execute(
        "SELECT o.name, t.search_method, COUNT(*) AS n, SUM(t.diverged) AS div"
        " FROM tuning_trials t JOIN optimizers o ON o.id = t.optimizer_id"
        " WHERE t.experiment_id = ? GROUP BY o.name, t.search_method",
        (exp_id,),
    ).fetchall()
    status: dict[str, dict[str, int]] = defaultdict(dict)
    for r in runs:
        status[r["name"]][r["status"]] = r["n"]
    optimizers = sorted(status, key=lambda o: (list(FAMILY_COLOR).index(family[o]), o))
    budget = {r["name"]: f"{r['n']} ({r['search_method']}, {r['div']} diverged)" for r in trials}
    cov = [["optimizer", "completed", "diverged", "failed", "tuning trials"]] + [
        [
            o,
            *(str(status[o].get(s, 0)) for s in ("completed", "diverged", "failed")),
            budget.get(o, "—"),
        ]
        for o in optimizers
    ]
    _write_csv(out_dir / "coverage.csv", cov)
    sections.append("## Runs and tuning budget\n\n" + _md(cov))

    # 2. Leaderboards (test split) with per-seed values for the bootstrap CI.
    evals = conn.execute(
        "SELECT o.name AS optimizer, d.name AS dataset, e.protocol, e.granularity, e.auc, e.eer,"
        " r.seed, r.wall_clock_sec FROM evaluations e JOIN runs r ON r.id = e.run_id"
        " JOIN optimizers o ON o.id = r.optimizer_id JOIN datasets d ON d.id = e.eval_dataset_id"
        " WHERE r.experiment_id = ? AND r.status = 'completed' AND e.split = 'test'",
        (exp_id,),
    ).fetchall()
    by_key: dict[tuple, dict[str, list[sqlite3.Row]]] = defaultdict(lambda: defaultdict(list))
    for e in evals:
        by_key[(e["dataset"], e["protocol"], e["granularity"])][e["optimizer"]].append(e)
    for (dataset, protocol, gran), groups in sorted(by_key.items()):
        rows = [["optimizer", "n seeds", "AUC mean ± sd", "95% CI", "EER mean", "fit time (s)"]]
        ranked = sorted(groups, key=lambda o: -np.mean([e["auc"] for e in groups[o]]))
        for o in ranked:
            auc = [e["auc"] for e in groups[o]]
            sd = float(np.std(auc, ddof=1)) if len(auc) > 1 else float("nan")
            lo, hi = bootstrap_ci(auc)
            low_n = " (low n)" if len(auc) < 3 else ""
            rows.append(
                [
                    o,
                    f"{len(auc)}{low_n}",
                    f"{_fmt(np.mean(auc))} ± {_fmt(sd)}",
                    f"[{_fmt(lo)}, {_fmt(hi)}]",
                    _fmt(float(np.mean([e["eer"] for e in groups[o]]))),
                    _fmt(float(np.mean([e["wall_clock_sec"] for e in groups[o]])), 1),
                ]
            )
        name = f"leaderboard_{dataset}_{gran}"
        _write_csv(out_dir / f"{name}.csv", rows)
        sections.append(f"## Test AUC: {dataset} ({protocol}, {gran} level)\n\n" + _md(rows))
        if gran == "video" and sum(len(v) >= 2 for v in groups.values()) >= 2:
            pw = pairwise({o: [e["auc"] for e in groups[o]] for o in ranked if len(groups[o]) >= 2})
            prow = [["a", "b", "mean diff", "95% CI", "Cohen's d", "Holm p", "verdict"]] + [
                [
                    p["a"],
                    p["b"],
                    _fmt(p["mean_diff"]),
                    f"[{_fmt(p['ci95'][0])}, {_fmt(p['ci95'][1])}]",
                    _fmt(p["cohens_d"], 2),
                    _fmt(p["p_holm"], 3),
                    "distinguishable" if p["distinguishable"] else "within noise",
                ]
                for p in pw
            ]
            _write_csv(out_dir / f"pairwise_{dataset}.csv", prow)
            n_dist = sum(p["distinguishable"] for p in pw)
            sections.append(
                f"### Pairwise, {dataset} video AUC\n\n{n_dist} of {len(pw)} pairs distinguishable;"
                " the rest are within seed noise.\n\n" + _md(prow)
            )

    # 3. Generalization gap (in-distribution minus cross-dataset, video level).
    gaps: dict[str, list[float]] = defaultdict(list)
    for r in conn.execute(
        "SELECT o.name, g.auc_gap FROM v_generalisation_gap g"
        " JOIN optimizers o ON o.id = g.optimizer_id"
        " WHERE g.experiment_id = ?",
        (exp_id,),
    ):
        gaps[r[0]].append(r[1])
    if gaps:
        grow = [["optimizer", "n", "gap mean ± sd", "95% CI"]]
        for o in sorted(gaps, key=lambda o: np.mean(gaps[o])):
            g = gaps[o]
            sd = float(np.std(g, ddof=1)) if len(g) > 1 else float("nan")
            lo, hi = bootstrap_ci(g)
            grow.append(
                [o, str(len(g)), f"{_fmt(np.mean(g))} ± {_fmt(sd)}", f"[{_fmt(lo)}, {_fmt(hi)}]"]
            )
        _write_csv(out_dir / "generalization_gap.csv", grow)
        sections.append(
            "## Generalization gap (in-distribution − cross-dataset test AUC, video)\n\n"
            + _md(grow)
        )

    # 4. Convergence on val_tune.
    curves: dict[str, list[list[sqlite3.Row]]] = defaultdict(list)
    wall: dict[str, list[float]] = defaultdict(list)
    for run in conn.execute(
        "SELECT r.id, r.wall_clock_sec, o.name FROM runs r"
        " JOIN optimizers o ON o.id = r.optimizer_id"
        " WHERE r.experiment_id = ? AND r.status = 'completed'",
        (exp_id,),
    ).fetchall():
        curves[run["name"]].append(
            conn.execute(
                "SELECT * FROM run_epochs WHERE run_id = ? ORDER BY epoch", (run["id"],)
            ).fetchall()
        )
        wall[run["name"]].append(run["wall_clock_sec"])
    if curves:
        best = max(np.mean([max(e["val_auc"] for e in c) for c in cs]) for cs in curves.values())
        target = 0.95 * best
        crow = [
            [
                "optimizer",
                f"epochs to val AUC ≥ {_fmt(target)}",
                "seeds reaching it",
                "area under val loss",
                "≈ wall clock to target (s)",
                "divergence rate",
            ]
        ]
        for o in optimizers:
            cs = curves.get(o, [])
            hits = [epochs_to_target([e["val_auc"] for e in c], target) for c in cs]
            reached = [h for h in hits if h is not None]
            # Approximation: run wall clock pro-rated by epochs (per-epoch time isn't logged).
            secs = [
                w * (h + 1) / len(c)
                for c, h, w in zip(cs, hits, wall[o], strict=True)
                if h is not None
            ]
            n_total = sum(status[o].values())
            crow.append(
                [
                    o,
                    _fmt(float(np.median(reached)), 1) if reached else "—",
                    f"{len(reached)}/{len(cs)}",
                    _fmt(
                        float(np.mean([area_under_curve([e["val_loss"] for e in c]) for c in cs])),
                        3,
                    )
                    if cs
                    else "—",
                    "—" if not secs else _fmt(float(np.mean(secs)), 1),
                    f"{status[o].get('diverged', 0)}/{n_total}",
                ]
            )
        _write_csv(out_dir / "convergence.csv", crow)
        sections.append(
            "## Convergence (val_tune; target = 0.95 × best mean peak AUC)\n\n" + _md(crow)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        for o in optimizers:
            cs = curves.get(o)
            if not cs:
                continue
            n = min(len(c) for c in cs)  # early stopping: align on the shortest run
            m = np.array([[e["val_auc"] for e in c[:n]] for c in cs])
            dash, marker = DASH_MARKER.get(o, ("-", "o"))
            ax.plot(
                range(n),
                m.mean(0),
                label=o,
                color=FAMILY_COLOR[family[o]],
                linestyle=dash,
                marker=marker,
                markevery=max(n // 6, 1),
                markersize=5,
            )
            if len(cs) > 1:
                ax.fill_between(
                    range(n),
                    m.mean(0) - m.std(0, ddof=1),
                    m.mean(0) + m.std(0, ddof=1),
                    color=FAMILY_COLOR[family[o]],
                    alpha=0.12,
                )
        ax.set(
            xlabel="epoch",
            ylabel="val_tune video AUC",
            title=f"{experiment}: mean ± 1 sd over seeds",
        )
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_val_auc.png", dpi=130)
        plt.close(fig)
        sections.append("![validation AUC](convergence_val_auc.png)")

    summary = out_dir / "summary.md"
    summary.write_text(
        f"# {experiment}: statistics\n\n_Generated by `optlab stats` from the results database on "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Code version `{exp['git_commit']}`._\n\n"
        f"**Hypothesis (recorded before the runs):** {(exp['hypothesis'] or '').strip()}\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    return summary
