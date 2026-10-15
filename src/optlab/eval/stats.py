"""Seed-level statistics (TRD 8.3). With 5 seeds, power is low: prefer CIs and effect sizes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling seeds with replacement)."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    means = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of mean(a) - mean(b), resampling each group independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    return (float(np.quantile(diff, alpha / 2)), float(np.quantile(diff, 1 - alpha / 2)))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def welch_p(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2 or (np.ptp(a) == 0 and np.ptp(b) == 0):
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    # An undefined test (NaN, e.g. zero variance in both groups) counts as not significant.
    p = np.nan_to_num(np.asarray(pvalues, float), nan=1.0)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def pairwise(groups: Mapping[str, Sequence[float]], alpha: float = 0.05) -> list[dict]:
    """Every pair: mean difference, bootstrap CI, Cohen's d, Welch p, Holm-adjusted p.

    `distinguishable` needs both the CI to exclude 0 and the Holm-adjusted p < alpha.
    """
    rows = []
    for a, b in combinations(groups, 2):
        xa, xb = groups[a], groups[b]
        rows.append(
            {
                "a": a,
                "b": b,
                "mean_diff": float(np.mean(xa) - np.mean(xb)),
                "ci95": bootstrap_diff_ci(xa, xb),
                "cohens_d": cohens_d(xa, xb),
                "p_welch": welch_p(xa, xb),
            }
        )
    for row, p_adj in zip(rows, holm([r["p_welch"] for r in rows]), strict=True):
        lo, hi = row["ci95"]
        row["p_holm"] = p_adj
        row["distinguishable"] = bool(p_adj < alpha and (lo > 0 or hi < 0))
    return rows


def epochs_to_target(curve: Sequence[float], target: float) -> int | None:
    """First epoch whose value reaches the target (validation AUC), or None."""
    hits = [i for i, v in enumerate(curve) if v >= target]
    return hits[0] if hits else None


def area_under_curve(curve: Sequence[float]) -> float:
    """Trapezoidal area under a per-epoch curve (e.g. validation loss); lower = faster."""
    c = np.asarray(curve, float)
    return float(np.sum((c[1:] + c[:-1]) / 2)) if len(c) > 1 else float(c.sum())
