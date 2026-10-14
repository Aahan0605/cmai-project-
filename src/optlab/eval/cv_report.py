"""Cross-validated comparison across the folds of `optlab cv` (Gate 1 recommendation 2).

Unit of analysis: one fold. Per fold and optimizer, test AUC is averaged over seeds; pairs of
optimizers are then compared *paired by fold*, so differences between the people in each
fold (the dominant noise in a single split) cancel out. Pairwise rows report the mean
difference, a percentile bootstrap CI over folds, Cohen's d of the paired differences and a
paired t-test with Holm correction; "distinguishable" needs both the CI to exclude 0 and
Holm p < 0.05.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from optlab.eval.stats import bootstrap_ci, holm  # noqa: E402
from optlab.toy.report import FAMILY_COLOR  # noqa: E402


def _fmt(x: float, d: int = 4) -> str:
    return "—" if x is None or math.isnan(x) else f"{x:.{d}f}"


def _md(rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def fold_means(
    conn: sqlite3.Connection, base: str
) -> dict[tuple[str, str], dict[str, dict[int, float]]]:
    """(dataset, protocol) -> optimizer -> fold -> mean video test AUC over seeds."""
    rows = conn.execute(
        "SELECT x.name AS exp, o.name AS opt, d.name AS ds, e.protocol, e.auc"
        " FROM evaluations e JOIN runs r ON r.id = e.run_id"
        " JOIN experiments x ON x.id = r.experiment_id"
        " JOIN optimizers o ON o.id = r.optimizer_id"
        " JOIN datasets d ON d.id = e.eval_dataset_id"
        " WHERE x.name GLOB ? AND r.status = 'completed' AND e.split = 'test'"
        " AND e.granularity = 'video'",
        (f"{base}_f[0-9]*",),
    ).fetchall()
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        fold = int(r["exp"].rsplit("_f", 1)[1])
        acc[(r["ds"], r["protocol"])][r["opt"]][fold].append(r["auc"])
    return {
        key: {o: {f: float(np.mean(v)) for f, v in folds.items()} for o, folds in by_opt.items()}
        for key, by_opt in acc.items()
    }


def paired(per_opt: dict[str, dict[int, float]], alpha: float = 0.05) -> list[dict]:
    out = []
    for a, b in combinations(per_opt, 2):
        folds = sorted(set(per_opt[a]) & set(per_opt[b]))
        d = np.array([per_opt[a][f] - per_opt[b][f] for f in folds])
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        p = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 and sd > 0 else float("nan")
        out.append(
            {
                "a": a,
                "b": b,
                "n": len(d),
                "mean_diff": float(d.mean()),
                "ci95": bootstrap_ci(d),
                "d": float(d.mean() / sd) if sd and sd > 0 else float("nan"),
                "p": p,
            }
        )
    for row, p_adj in zip(out, holm([r["p"] for r in out]), strict=True):
        lo, hi = row["ci95"]
        row["p_holm"] = p_adj
        row["distinguishable"] = bool(p_adj < alpha and (lo > 0 or hi < 0))
    return out


def write_cv_report(conn: sqlite3.Connection, base: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    family = {r[0]: r[1] for r in conn.execute("SELECT name, family FROM optimizers")}
    means = fold_means(conn, base)
    if not means:
        raise SystemExit(f"no test evaluations for folds of {base!r}; run `optlab cv` first")
    sections = []
    fig, axes = plt.subplots(1, len(means), figsize=(6 * len(means), 4.5), squeeze=False)
    for ax, ((ds, protocol), per_opt) in zip(axes[0], sorted(means.items()), strict=True):
        ranked = sorted(per_opt, key=lambda o: -np.mean(list(per_opt[o].values())))
        rows = [["optimizer", "folds", "AUC mean ± sd over folds", "95% CI"]]
        for o in ranked:
            v = list(per_opt[o].values())
            lo, hi = bootstrap_ci(v)
            rows.append(
                [
                    o,
                    str(len(v)),
                    f"{_fmt(np.mean(v))} ± {_fmt(np.std(v, ddof=1))}",
                    f"[{_fmt(lo)}, {_fmt(hi)}]",
                ]
            )
        with open(out_dir / f"leaderboard_{ds}.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        pw = paired(per_opt)
        prow = [["a", "b", "folds", "mean diff", "95% CI", "paired d", "Holm p", "verdict"]] + [
            [
                r["a"],
                r["b"],
                str(r["n"]),
                _fmt(r["mean_diff"]),
                f"[{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}]",
                _fmt(r["d"], 2),
                _fmt(r["p_holm"], 3),
                "distinguishable" if r["distinguishable"] else "within noise",
            ]
            for r in sorted(pw, key=lambda r: r["p_holm"])
        ]
        with open(out_dir / f"pairwise_{ds}.csv", "w", newline="") as f:
            csv.writer(f).writerows(prow)
        n_dist = sum(r["distinguishable"] for r in pw)
        sections.append(
            f"## {ds} ({protocol}): video test AUC across folds\n\n"
            + _md(rows)
            + f"\n\n**Paired comparisons:** {n_dist} of {len(pw)} pairs distinguishable."
            " Strongest ten by Holm p:\n\n" + _md(prow[:11])
        )
        m = [np.mean(list(per_opt[o].values())) for o in ranked]
        ci = np.array([bootstrap_ci(list(per_opt[o].values())) for o in ranked])
        ax.errorbar(
            range(len(ranked)),
            m,
            yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.scatter(
            range(len(ranked)), m, c=[FAMILY_COLOR[family[o]] for o in ranked], zorder=3, s=40
        )
        ax.set_xticks(range(len(ranked)), ranked, rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="video test AUC (mean over folds, 95% CI)", title=f"{ds} ({protocol})")
    fig.tight_layout()
    fig.savefig(out_dir / "cv_auc.png", dpi=130)
    plt.close(fig)

    contrib = conn.execute(
        "SELECT w.modality, AVG(w.weight), AVG(w.contribution), COUNT(*) FROM fusion_weights w"
        " JOIN runs r ON r.id = w.run_id JOIN experiments x ON x.id = r.experiment_id"
        " WHERE x.name GLOB ? GROUP BY w.modality ORDER BY AVG(w.contribution) DESC",
        (f"{base}_f[0-9]*",),
    ).fetchall()
    sections.append(
        "## Fusion: mixing weight vs contribution (all runs, all folds)\n\n"
        "Contribution = the share of"
        " the fused logit's spread on val_fusion that a stream supplies (w_m · sd(z_m) / Σ).\n\n"
        + _md(
            [["stream", "mean weight", "mean contribution", "runs"]]
            + [[r[0], _fmt(r[1], 3), _fmt(r[2], 3), str(r[3])] for r in contrib]
        )
    )
    n_runs = conn.execute(
        "SELECT r.status, COUNT(*) FROM runs r JOIN experiments x ON x.id = r.experiment_id"
        " WHERE x.name GLOB ? GROUP BY r.status",
        (f"{base}_f[0-9]*",),
    ).fetchall()
    summary = out_dir / "summary.md"
    summary.write_text(
        f"# {base}: cross-validated statistics\n\n_Generated by `optlab cv-stats` from the results"
        f" database on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}._\n\n"
        f"Runs by status: {', '.join(f'{s} {n}' for s, n in n_runs)}.\n\n"
        "One fold = one held-out group of people; per fold, AUC is averaged over seeds, and"
        " optimizers are compared paired by fold.\n\n![cross-validated AUC](cv_auc.png)\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    return summary
