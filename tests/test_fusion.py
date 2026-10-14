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


@pytest.mark.parametrize("seed", range(5))
def test_projected_gradient_handles_badly_scaled_logits(seed: int) -> None:
    # Like a trained detector: one confident stream (large logits), two near-constant ones.
    z, y = problem(seed)
    z = z * np.array([7.0, 0.2, 0.01])
    sol = projected_gradient.solve(z, y)
    assert sol.kkt_residual < 1e-9
    assert sol.objective <= qp.solve(z, y).objective + 1e-9


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


@pytest.mark.parametrize("seed", range(5))
def test_temperature_fusion_cvxpy_and_newton_agree(seed: int) -> None:
    from optlab.fusion import temperature

    z, y = problem(seed)
    z = z * np.array([7.0, 0.2, 0.01])  # badly scaled, like trained heads
    a, b = temperature.solve_cvxpy(z, y), temperature.solve_newton(z, y)
    assert b.kkt_residual < 1e-9
    assert a.objective == pytest.approx(b.objective, abs=1e-7)
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-4)
    assert b.weights.sum() == pytest.approx(1.0) and b.temperature > 0


def test_temperature_fusion_gives_noise_no_contribution() -> None:
    # Plain simplex fusion parks weight on a near-constant stream to temper an overconfident
    # one; with a free temperature that is unnecessary, so the noise stream's share vanishes.
    from optlab.fusion import temperature

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400) * 2 - 1.0
    z = np.column_stack([8 * (y + rng.normal(size=400)), 0.01 * rng.normal(size=400)])
    plain = qp.solve(z, y)
    calibrated = temperature.solve(z, y)
    assert plain.weights[1] > 0.5  # the artefact seen in baseline_timit_v3
    assert calibrated.contribution[1] < 0.01


def test_temperature_fusion_is_bounded_on_separable_data() -> None:
    from optlab.fusion import temperature

    y = np.array([-1.0, -1.0, 1.0, 1.0])
    z = np.column_stack([y * 3, np.zeros(4)])  # perfectly separable: loss alone has no minimiser
    sol = temperature.solve(z, y)
    assert np.isfinite(sol.temperature) and sol.kkt_residual < 1e-6
