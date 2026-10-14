import numpy as np
import pytest

from optlab.fusion import kkt_closed_form, projected_gradient, qp
from optlab.fusion.kkt import kkt_diagnostics, project_simplex, squared_grad


def problem(seed: int, m: int = 3, n: int = 400):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n) * 2 - 1.0
    # Modalities of different quality; the last one is pure noise, so its weight should be 0.
    strength = np.linspace(1.5, 0.0, m)
    z = y[:, None] * strength[None, :] + rng.normal(size=(n, m))
    return z, y


@pytest.mark.parametrize("seed", range(5))
def test_qp_and_projected_gradient_agree_on_simplex(seed: int) -> None:
    z, y = problem(seed)
    a, c = qp.solve(z, y), projected_gradient.solve(z, y)
    for sol in (a, c):
        assert sol.weights.min() >= 0 and sol.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert a.kkt_residual < 1e-6 and c.kkt_residual < 1e-6
    np.testing.assert_allclose(a.weights, c.weights, atol=1e-5)
    assert a.objective == pytest.approx(c.objective, abs=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_matches_cvxpy_on_squared_loss(seed: int) -> None:
    import cvxpy as cp

    z, y = problem(seed)
    b = kkt_closed_form.solve(z, y)
    w = cp.Variable(z.shape[1])
    cp.Problem(cp.Minimize(cp.sum_squares(z @ w - y) / len(y)), [w >= 0, cp.sum(w) == 1]).solve()
    np.testing.assert_allclose(b.weights, w.value, atol=1e-6)
    assert b.kkt_residual < 1e-6


def test_qp_returns_inaccurate_optimum_for_caller_to_judge() -> None:
    # Large logits (like a very confident head) stress the conic solver; the result must still
    # be on the simplex, and projected gradient must reach a tight KKT residual.
    z, y = problem(0)
    z = z * 20
    a = qp.solve(z, y)
    assert a.weights.min() >= 0 and a.weights.sum() == pytest.approx(1.0)
    assert projected_gradient.solve(z, y).kkt_residual < 1e-6


def test_noise_modality_is_switched_off() -> None:
    z, y = problem(0)
    sol = qp.solve(z, y)
    assert 2 in sol.active_set and sol.weights[2] < 1e-8


def test_kkt_residual_flags_a_non_optimal_point() -> None:
    z, y = problem(0)
    uniform = np.full(3, 1 / 3)
    assert kkt_diagnostics(squared_grad(uniform, z, y), uniform)[1] > 1e-3


def test_project_simplex() -> None:
    np.testing.assert_allclose(project_simplex(np.array([0.2, 0.3, 0.5])), [0.2, 0.3, 0.5])
    np.testing.assert_allclose(project_simplex(np.array([2.0, 0.0, -1.0])), [1.0, 0.0, 0.0])
    p = project_simplex(np.random.default_rng(1).normal(size=6))
    assert p.min() >= 0 and p.sum() == pytest.approx(1.0)
