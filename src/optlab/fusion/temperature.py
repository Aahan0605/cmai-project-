"""Simplex fusion with a learned temperature (Gate 1 fix for uninterpretable weights).

    minimize_{u >= 0}  (1/n) sum_i log(1 + exp(-y_i u^T z_i)) + (lam/2) ||u||^2

With u = s w (w on the simplex, s = 1^T u > 0) this is the simplex model with a free
temperature s, so the mixing weights w no longer have to absorb the logits' scale. The
substitution makes it convex in u. The small ridge `lam` keeps a minimiser when val_fusion
is perfectly separable (the loss alone would push s to infinity).

KKT: g_m + lam u_m - mu_m = 0, mu_m >= 0, mu_m u_m = 0 (no equality constraint, so no lambda).

A stream whose logits are constant on val_fusion carries no information to rank clips; a
weight on it only shifts every score by the same amount (an intercept), and that direction is
nearly flat, which stalls both solvers. Such streams are fixed at u_m = 0 and the problem is
solved over the informative streams; their contribution is 0 by definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optlab.fusion.kkt import logistic_grad, logistic_loss
from optlab.fusion.projected_gradient import logistic_hessian


@dataclass
class TemperatureSolution:
    solver: str  # 'qp_cvxpy' | 'projected_gradient'
    u: np.ndarray
    weights: np.ndarray  # u / sum(u), on the simplex
    temperature: float  # sum(u)
    contribution: np.ndarray  # w_m sd(z_m) / sum_k w_k sd(z_k)
    objective: float
    active_set: list[int]
    kkt_residual: float


def _grad(u, z, y, lam):
    return logistic_grad(u, z, y) + lam * u


def _objective(u, z, y, lam):
    return logistic_loss(u, z, y) + 0.5 * lam * float(u @ u)


def kkt_residual(u: np.ndarray, g: np.ndarray, tol: float = 1e-10) -> tuple[float, list[int]]:
    active = [m for m in range(len(u)) if u[m] <= tol]
    free = [m for m in range(len(u)) if m not in active]
    parts = np.concatenate(
        [g[free], np.minimum(g[active], 0.0), g[active] * u[active], np.minimum(u, 0)]
    )
    return float(np.linalg.norm(parts)), active


def _finish(
    solver: str, u: np.ndarray, z: np.ndarray, y: np.ndarray, lam: float
) -> TemperatureSolution:
    u = np.maximum(u, 0.0)
    s = float(u.sum())
    w = u / s if s > 0 else np.full(len(u), 1.0 / len(u))
    spread = w * z.std(axis=0)
    contrib = spread / spread.sum() if spread.sum() > 0 else np.full(len(u), 1.0 / len(u))
    res, active = kkt_residual(u, _grad(u, z, y, lam))
    return TemperatureSolution(solver, u, w, s, contrib, _objective(u, z, y, lam), active, res)


def solve_cvxpy(z: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> TemperatureSolution:
    import cvxpy as cp

    u = cp.Variable(z.shape[1])
    obj = cp.sum(cp.logistic(-cp.multiply(y, z @ u))) / len(y) + 0.5 * lam * cp.sum_squares(u)
    problem = cp.Problem(cp.Minimize(obj), [u >= 0])
    problem.solve()
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"temperature fusion did not solve: {problem.status}")
    return _finish("qp_cvxpy", np.asarray(u.value), z, y, lam)


def solve_newton(
    z: np.ndarray, y: np.ndarray, lam: float = 1e-3, max_iter: int = 200
) -> TemperatureSolution:
    """Projected Newton on the free coordinates; a coordinate leaves the free set when it
    reaches 0 and rejoins when its gradient turns negative (the KKT sign condition)."""
    m = z.shape[1]
    u = np.full(m, 1.0 / m)
    for _ in range(max_iter):
        g = _grad(u, z, y, lam)
        if kkt_residual(u, g)[0] < 1e-12:
            break
        free = np.flatnonzero((u > 0) | (g < 0))
        h = logistic_hessian(u, z, y)[np.ix_(free, free)] + lam * np.eye(len(free))
        d = np.zeros(m)
        d[free] = -np.linalg.solve(h, g[free])
        neg = d < 0
        t = min(1.0, float(np.min(-u[neg] / d[neg]))) if neg.any() else 1.0
        f0 = _objective(u, z, y, lam)
        while t > 1e-16:
            cand = np.maximum(u + t * d, 0.0)
            if _objective(cand, z, y, lam) <= f0 + 1e-4 * t * float(g @ d):
                break
            t /= 2
        else:
            break
        u = cand
    return _finish("projected_gradient", u, z, y, lam)


def solve(
    z: np.ndarray, y: np.ndarray, lam: float = 1e-3, tol: float = 1e-6
) -> TemperatureSolution:
    """cvxpy first; projected Newton when cvxpy's own KKT residual misses `tol`. Streams with
    constant logits on these samples are held at u = 0 (see module docstring)."""
    m = z.shape[1]
    sd = z.std(axis=0)
    keep = np.flatnonzero(sd > 1e-6 * max(float(sd.max()), 1e-300))
    if len(keep) == 0:  # every head is constant: nothing to fuse, scores carry no ranking
        uniform = np.full(m, 1 / m)
        return TemperatureSolution(
            "projected_gradient",
            np.zeros(m),
            uniform,
            0.0,
            uniform,
            float(np.log(2)),
            list(range(m)),
            0.0,
        )
    zk = z[:, keep]
    sol = None
    try:
        sol = solve_cvxpy(zk, y, lam)
    except Exception:  # noqa: BLE001 - any solver failure falls through to Newton
        pass
    if sol is None or sol.kkt_residual > tol:
        sol = solve_newton(zk, y, lam)
    if sol.kkt_residual > tol:
        raise RuntimeError(f"temperature fusion did not converge: {sol.kkt_residual:.1e}")
    u = np.zeros(m)
    u[keep] = sol.u
    full = _finish(sol.solver, u, z, y, lam)
    full.kkt_residual = sol.kkt_residual  # residual of the problem actually solved
    full.active_set = sorted(set(range(m)) - set(keep[sol.u > 1e-10].tolist()))
    return full
