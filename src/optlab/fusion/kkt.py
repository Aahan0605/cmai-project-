"""Simplex-constrained fusion (TRD 9): w >= 0, 1^T w = 1. Shared losses and KKT diagnostics.

Z holds per-modality logits (n x M); y_pm the labels in {-1, +1}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionSolution:
    solver: str  # 'kkt_closed_form' | 'qp_cvxpy' | 'projected_gradient'
    weights: np.ndarray
    objective: float
    multiplier: float  # lambda for 1^T w = 1
    active_set: list[int]  # indices m with w_m = 0
    kkt_residual: float


def logistic_loss(w: np.ndarray, z: np.ndarray, y_pm: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, -y_pm * (z @ w))))


def logistic_grad(w: np.ndarray, z: np.ndarray, y_pm: np.ndarray) -> np.ndarray:
    margin = y_pm * (z @ w)
    return -(z * (y_pm / (1 + np.exp(margin)))[:, None]).mean(0)


def squared_loss(w: np.ndarray, z: np.ndarray, y_pm: np.ndarray) -> float:
    return float(np.mean((z @ w - y_pm) ** 2))


def squared_grad(w: np.ndarray, z: np.ndarray, y_pm: np.ndarray) -> np.ndarray:
    return 2 * z.T @ (z @ w - y_pm) / len(y_pm)


def kkt_diagnostics(
    grad: np.ndarray, w: np.ndarray, active_tol: float = 1e-8
) -> tuple[float, float, list[int]]:
    """lambda, KKT residual and active set at a candidate solution.

    Stationarity: grad_m + lambda - mu_m = 0, mu_m >= 0, mu_m w_m = 0, 1^T w = 1, w >= 0.
    lambda comes from the inactive coordinates (where mu_m = 0), mu = grad + lambda. The
    residual is the norm of every violation: stationarity on inactive coordinates, negative
    mu, complementarity mu_m w_m, primal feasibility.
    """
    active = [m for m in range(len(w)) if w[m] <= active_tol]
    inactive = [m for m in range(len(w)) if m not in active] or list(range(len(w)))
    lam = -float(np.mean(grad[inactive]))
    mu = grad + lam
    parts = np.concatenate(
        [
            mu[inactive],  # should be 0 where w_m > 0
            np.minimum(mu[active], 0.0),  # dual feasibility
            mu[active] * w[active],  # complementarity
            [w.sum() - 1.0],
            np.minimum(w, 0.0),
        ]
    )
    return lam, float(np.linalg.norm(parts)), active


def finish(solver: str, w: np.ndarray, objective: float, grad: np.ndarray) -> FusionSolution:
    lam, residual, active = kkt_diagnostics(grad, w)
    return FusionSolution(solver, w, objective, lam, active, residual)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (sort-based, O(M log M))."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    k = np.flatnonzero(u - css / np.arange(1, len(v) + 1) > 0)[-1]
    return np.maximum(v - css[k] / (k + 1), 0.0)
