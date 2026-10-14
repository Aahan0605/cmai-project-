"""Solver A: the convex fusion problem solved with cvxpy (TRD 9.1).

minimize (1/n) sum log(1 + exp(-y_i w^T z_i))  subject to  w >= 0, 1^T w = 1.
The loss is convex in w and the uniform vector is strictly feasible, so Slater holds and the
KKT conditions characterise the optimum.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from optlab.fusion.kkt import FusionSolution, finish, logistic_grad, logistic_loss


def solve(z: np.ndarray, y_pm: np.ndarray) -> FusionSolution:
    w = cp.Variable(z.shape[1])
    objective = cp.sum(cp.logistic(-cp.multiply(y_pm, z @ w))) / len(y_pm)
    problem = cp.Problem(cp.Minimize(objective), [w >= 0, cp.sum(w) == 1])
    problem.solve()
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"fusion QP did not solve: {problem.status}")
    # An "inaccurate" optimum is returned as is; the caller judges it by its KKT residual.
    # Solver output can sit 1e-10 outside the simplex; snap it back before the KKT check.
    sol = np.clip(np.asarray(w.value), 0.0, None)
    sol /= sol.sum()
    return finish("qp_cvxpy", sol, logistic_loss(sol, z, y_pm), logistic_grad(sol, z, y_pm))
