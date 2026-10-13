"""Solver C: projected gradient on the logistic fusion loss (cross-check for the QP)."""

from __future__ import annotations

import numpy as np

from optlab.fusion.kkt import FusionSolution, finish, logistic_grad, logistic_loss, project_simplex


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
    return finish("projected_gradient", w, logistic_loss(w, z, y_pm), logistic_grad(w, z, y_pm))
