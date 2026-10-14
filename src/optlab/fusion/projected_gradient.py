"""Solver C: projected gradient on the logistic fusion loss, finished by a Newton polish.

Projected gradient finds the active set (which weights are 0) robustly, but its 1/L step is
very slow when modalities' logits differ in scale by orders of magnitude (the Hessian is
then badly conditioned). The polish runs equality-constrained Newton on the free weights
(KKT system [H 1; 1^T 0][d; nu] = [-g; 0]) with a backtracking step that keeps w >= 0;
a weight that reaches 0 joins the active set. Newton is invariant to that scaling.
"""

from __future__ import annotations

import numpy as np

from optlab.fusion.kkt import (
    FusionSolution,
    finish,
    kkt_diagnostics,
    logistic_grad,
    logistic_loss,
    project_simplex,
)


def logistic_hessian(w: np.ndarray, z: np.ndarray, y_pm: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-y_pm * (z @ w)))
    return (z * (p * (1 - p))[:, None]).T @ z / len(y_pm)


def newton_polish(
    w: np.ndarray, z: np.ndarray, y_pm: np.ndarray, tol: float = 1e-12, max_iter: int = 100
) -> np.ndarray:
    w = w.copy()
    for _ in range(max_iter):
        if kkt_diagnostics(logistic_grad(w, z, y_pm), w)[1] < tol:
            break
        free = np.flatnonzero(w > 0)
        g = logistic_grad(w, z, y_pm)[free]
        h = logistic_hessian(w, z, y_pm)[np.ix_(free, free)]
        k = len(free)
        kkt = np.zeros((k + 1, k + 1))
        kkt[:k, :k] = h + 1e-12 * np.eye(k)  # tiny ridge: a constant-logit stream is singular
        kkt[:k, k] = kkt[k, :k] = 1.0
        d = np.linalg.solve(kkt, np.append(-g, 0.0))[:k]
        # Largest step that keeps the free weights non-negative, then Armijo backtracking.
        shrinking = d < 0
        t = min(1.0, float(np.min(-w[free][shrinking] / d[shrinking]))) if shrinking.any() else 1.0
        f0 = logistic_loss(w, z, y_pm)
        while t > 1e-16:
            cand = w.copy()
            cand[free] = w[free] + t * d
            cand[cand < 1e-15] = 0.0
            if logistic_loss(cand, z, y_pm) <= f0 + 1e-4 * t * float(g @ d):
                break
            t /= 2
        else:
            break  # no descent possible from here: at the optimum up to rounding
        w = cand / cand.sum()
    return w


def solve(
    z: np.ndarray, y_pm: np.ndarray, max_iter: int = 100_000, tol: float = 1e-12
) -> FusionSolution:
    # Logistic loss Hessian <= Z^T Z / (4n): 1/L step guarantees descent.
    lipschitz = np.linalg.eigvalsh(z.T @ z / (4 * len(y_pm)))[-1]
    step = 1.0 / max(lipschitz, 1e-12)
    w = np.full(z.shape[1], 1.0 / z.shape[1])
    for _ in range(max_iter):
        new = project_simplex(w - step * logistic_grad(w, z, y_pm))
        if np.linalg.norm(new - w) < tol:
            w = new
            break
        w = new
    w = newton_polish(w, z, y_pm)
    return finish("projected_gradient", w, logistic_loss(w, z, y_pm), logistic_grad(w, z, y_pm))
