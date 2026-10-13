"""Solver B: closed form for the squared-loss variant L(w) = ||Zw - y||^2 / n (TRD 9.2).

For a given set F of free (non-zero) weights, the KKT conditions are linear:
    [2 Z_F^T Z_F / n   1] [w_F   ]   [2 Z_F^T y / n]
    [1^T               0] [lambda] = [1            ]
with w_m = 0 off F. Every non-empty F is solved and the lowest-loss non-negative solution
kept; since the problem is convex, that solution satisfies all KKT conditions.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from optlab.fusion.kkt import FusionSolution, finish, squared_grad, squared_loss


def solve(z: np.ndarray, y_pm: np.ndarray) -> FusionSolution:
    # ponytail: enumerates all 2^M - 1 active sets; exact and instant for M <= 3 modalities,
    # switch to a primal-dual active-set method if M ever grows past ~15.
    n, m = z.shape
    best: np.ndarray | None = None
    for k in range(1, m + 1):
        for free in combinations(range(m), k):
            zf = z[:, free]
            kkt = np.zeros((k + 1, k + 1))
            kkt[:k, :k] = 2 * zf.T @ zf / n
            kkt[:k, k] = kkt[k, :k] = 1.0
            rhs = np.append(2 * zf.T @ y_pm / n, 1.0)
            try:
                w_free = np.linalg.solve(kkt, rhs)[:k]
            except np.linalg.LinAlgError:
                continue
            if (w_free < -1e-12).any():
                continue
            w = np.zeros(m)
            w[list(free)] = np.maximum(w_free, 0.0)
            if best is None or squared_loss(w, z, y_pm) < squared_loss(best, z, y_pm):
                best = w
    if best is None:
        raise RuntimeError("no feasible closed-form fusion solution")
    return finish("kkt_closed_form", best, squared_loss(best, z, y_pm), squared_grad(best, z, y_pm))
