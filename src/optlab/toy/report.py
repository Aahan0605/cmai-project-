"""Toy-suite tables, plots and summary, generated from the results DB only.

The benchmark config is used solely to redraw the function surface under the contour plots;
every number in the tables and the summary comes from `toy_runs` / `toy_run_curves`.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from optlab.config import ToyExperimentConfig  # noqa: E402
from optlab.toy.functions import make_function, minimizer  # noqa: E402

# docs/UIUX.md 5.1 (light theme): colour = family, dash + marker = optimizer.
FAMILY_COLOR = {
    "plain": "#6B7280",
    "momentum": "#0284C7",
    "adaptive": "#B86E00",
    "sign": "#B5588C",
    "sharpness_aware": "#009E73",
    "second_order": "#0072B2",
    "hybrid": "#D55E00",
}
DASH_MARKER = {
    "sgd": ("-", "o"),
    "sgd_momentum": ("-", "s"),
    "nag": ("--", "^"),
    "rmsprop": (":", "D"),
    "adam": ("-", "o"),
    "adamw": ("--", "s"),
    "radam": ("-.", "v"),
    "adabelief": ((0, (10, 3)), "X"),
    "lion": ("-", "*"),
    "sam_sgd_momentum": ("-", "h"),
    "newton": ("-", "p"),
    "lbfgs": ("--", "p"),
}

RUNS_SQL = """
SELECT b.name AS benchmark, b.dimension, b.global_min_value, o.name AS optimizer,
       o.family, r.id, r.seed, r.hyperparams_json, r.max_steps, r.tolerance,
       r.steps_to_tol, r.final_f, r.final_grad_norm, r.trajectory_path
FROM toy_runs r
JOIN toy_benchmarks b ON b.id = r.benchmark_id
JOIN optimizers o ON o.id = r.optimizer_id
ORDER BY b.id, o.id, r.seed
"""


def _style(optimizer: str, family: str) -> dict:
    dash, marker = DASH_MARKER.get(optimizer, ("-", "o"))
    return {"color": FAMILY_COLOR[family], "linestyle": dash, "marker": marker}


def _group(rows: list[sqlite3.Row]) -> dict[tuple[str, str], list[sqlite3.Row]]:
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault((r["benchmark"], r["optimizer"]), []).append(r)
    return groups


def _steps_cell(runs: list[sqlite3.Row]) -> str:
    reached = [r["steps_to_tol"] for r in runs if r["steps_to_tol"] is not None]
    count = f"{len(reached)}/{len(runs)}"
    if not reached:
        return f"— ({count})"
    if len(reached) == 1:
        return f"{reached[0]} ({count})"
    return f"{median(reached):g} [{min(reached)}–{max(reached)}] ({count})"


def _final_cell(runs: list[sqlite3.Row]) -> str:
    finals = [r["final_f"] for r in runs if r["final_f"] is not None]
    diverged = len(runs) - len(finals)
    cell = f"{median(finals):.3g}" if finals else "—"
    return cell + (f" ({diverged} diverged)" if diverged else "")


def _table(
    groups: dict[tuple[str, str], list[sqlite3.Row]],
    benchmarks: list[str],
    optimizers: list[str],
    cell,
) -> list[list[str]]:
    rows = [["optimizer", *benchmarks]]
    for o in optimizers:
        rows.append([o, *(cell(groups[(b, o)]) if (b, o) in groups else "" for b in benchmarks)])
    return rows


def _markdown(rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _lr_cell(runs: list[sqlite3.Row]) -> str:
    lrs = {json.loads(r["hyperparams_json"]).get("lr") for r in runs}
    return ", ".join("fixed" if lr is None else f"{lr:g}" for lr in sorted(lrs, key=str))


def plot_benchmark(
    benchmark: str,
    runs_by_opt: dict[str, list[dict]],
    exp: ToyExperimentConfig,
    out_dir: Path,
) -> list[Path]:
    cfg = next(b for b in exp.benchmarks if b.name == benchmark)
    seed = exp.tuning_seed
    first = {o: next(r for r in rs if r["seed"] == seed) for o, rs in runs_by_opt.items()}
    paths = []

    # f(x_t) - f* vs step, one seed, every optimizer (curves come from toy_run_curves).
    fig, ax = plt.subplots(figsize=(8, 5))
    for o, r in first.items():
        curve = np.array(r["curve"])
        if len(curve) == 0:
            continue
        gap = np.maximum(curve[:, 1] - r["global_min_value"], 1e-16)
        ax.plot(curve[:, 0], gap, label=o, markevery=0.1, markersize=5, **_style(o, r["family"]))
    ax.axhline(cfg.tolerance, color="black", linewidth=0.8, linestyle=":", label="tolerance")
    ax.set_xscale("symlog", linthresh=1)
    ax.set(yscale="log", xlabel="step", ylabel="f(x_t) − f*", xlim=(0, None))
    ax.set_title(f"{benchmark}: convergence (seed {seed}, lr selected on seed {seed})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    paths.append(out_dir / f"{benchmark}_curves.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    if cfg.dim != 2:
        return paths

    # Contours with trajectories, one panel per optimizer.
    trajs = {o: np.load(r["trajectory_path"])["x"] for o, r in first.items()}
    x_star = minimizer(cfg)
    pts = np.concatenate([t for t in trajs.values() if len(t)] + [x_star[None]])
    # Whole paths, but clamped near the start box so one diverging run cannot zoom out the rest.
    lo = np.maximum(pts.min(axis=0), cfg.start_low - 1)
    hi = np.minimum(pts.max(axis=0), cfg.start_high + 1)
    pad = 0.1 * (hi - lo) + 0.05
    lo, hi = lo - pad, hi + pad
    fn = make_function(cfg)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1))
    with torch.no_grad():
        z = np.array([float(fn(p)) for p in grid]).reshape(gx.shape)
    log_levels = cfg.kind != "rastrigin"

    n = len(trajs)
    cols = 4
    fig, axes = plt.subplots(
        (n + cols - 1) // cols, cols, figsize=(4 * cols, 3.6 * ((n + cols - 1) // cols))
    )
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (o, path) in zip(axes.ravel(), trajs.items(), strict=False):
        if log_levels:
            levels = np.logspace(np.log10(max(z.min(), 1e-6)), np.log10(z.max()), 20)
            ax.contour(gx, gy, np.maximum(z, 1e-6), levels=levels, norm=LogNorm(), cmap="Greys")
        else:
            ax.contour(gx, gy, z, levels=20, cmap="Greys")
        style = _style(o, first[o]["family"])
        if len(path):
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.2,
            )
            ax.plot(*path[0], "o", mfc="none", mec="black", ms=7)
            ax.plot(*path[-1], "*", color=style["color"], ms=11, mec="black")
        ax.plot(*x_star, "x", color="black", ms=9)
        r = first[o]
        status = (
            "diverged"
            if r["final_f"] is None
            else (
                f"tol in {r['steps_to_tol']}"
                if r["steps_to_tol"] is not None
                else f"f={r['final_f']:.3g}"
            )
        )
        ax.set_title(f"{o} — {status}", fontsize=9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{benchmark}: trajectories (seed {seed}); ◯ start, ★ end, ✕ global minimum")
    fig.tight_layout()
    paths.append(out_dir / f"{benchmark}_trajectories.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def write_report(
    conn: sqlite3.Connection, exp: ToyExperimentConfig, out_dir: Path, summary_path: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(RUNS_SQL).fetchall()
    if not rows:
        raise SystemExit("no toy runs in the database; run `optlab toy` first")
    groups = _group(rows)
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in rows))
    configured = exp.optimizers + exp.reference_optimizers
    present = {r["optimizer"] for r in rows}
    optimizers = [o for o in configured if o in present] + sorted(present - set(configured))

    steps = _table(groups, benchmarks, optimizers, _steps_cell)
    finals = _table(groups, benchmarks, optimizers, _final_cell)
    lrs = _table(groups, benchmarks, optimizers, _lr_cell)
    for name, table in [("steps_to_tol", steps), ("final_f", finals), ("selected_lr", lrs)]:
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            csv.writer(f).writerows(table)

    plots: list[Path] = []
    for b in benchmarks:
        by_opt = {}
        for o in optimizers:
            if (b, o) not in groups:
                continue
            runs = []
            for r in groups[(b, o)]:
                d = dict(r)
                d["curve"] = conn.execute(
                    "SELECT step, f_value FROM toy_run_curves WHERE toy_run_id = ? ORDER BY step",
                    (r["id"],),
                ).fetchall()
                runs.append(d)
            by_opt[o] = runs
        plots += plot_benchmark(b, by_opt, exp, out_dir)

    experiment = conn.execute(
        "SELECT name, git_commit, hypothesis, description, created_at FROM experiments"
        " WHERE name = ?",
        (exp.name,),
    ).fetchone()
    settings = conn.execute(
        "SELECT b.name, b.dimension, b.condition_number, MAX(r.max_steps), MAX(r.tolerance),"
        " COUNT(DISTINCT r.seed) FROM toy_runs r JOIN toy_benchmarks b ON b.id = r.benchmark_id"
        " GROUP BY b.id ORDER BY b.id"
    ).fetchall()
    settings_md = _markdown(
        [["benchmark", "dim", "κ", "max steps", "tolerance |f − f*|", "seeds"]]
        + [
            [s[0], str(s[1]), f"{s[2]:g}" if s[2] else "", str(s[3]), f"{s[4]:g}", str(s[5])]
            for s in settings
        ]
    )
    rel = [p.relative_to(summary_path.parent).as_posix() for p in plots]
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    text = (
        f"""# Toy benchmark summary

_Generated by `optlab toy-report` from the results database on {generated}. Do not edit by hand;
run `make toy` to regenerate._

- Experiment: `{experiment["name"]}` (recorded {experiment["created_at"]} UTC)
- Code version: `{experiment["git_commit"]}`
- Toy runs in the database: {len(rows)}

## Protocol

{experiment["description"].strip()}

One step is one `optimizer.step()`. For L-BFGS that is one outer iteration (its line-search
function evaluations are not counted); for Newton it is one `x ← x − H⁻¹g`. A run stops at the
first step with |f − f*| ≤ tolerance, on divergence (non-finite or |f| above the threshold), or
at max steps. Diverged runs are kept and shown as diverged, not dropped.

{settings_md}

## Hypothesis (recorded before the runs)

{experiment["hypothesis"].strip()}

## Steps to tolerance

Median [min–max] over the seeds that reached tolerance, and (seeds reached / seeds run).
"—" means no seed reached tolerance within max steps.

{_markdown(steps)}

## Final f

Median final f over non-diverged seeds (f* = 0 for every benchmark).

{_markdown(finals)}

## Selected learning rate

{_markdown(lrs)}

## Plots

"""
        + "\n".join(f"![{Path(p).stem}]({p})" for p in rel)
        + "\n"
    )
    summary_path.write_text(text)
