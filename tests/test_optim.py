from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from optlab.config import load_optimizers
from optlab.optim import BUILDERS, build_optimizer
from optlab.optim.adam import Adam
from optlab.optim.momentum import Momentum
from optlab.optim.nag import NAG
from optlab.optim.registry import build_from_config
from optlab.optim.rmsprop import RMSProp
from optlab.optim.sam import SAM
from optlab.optim.sgd import SGD

D = torch.float64  # all optimizer tests run in double precision (TRD 15)


def random_quadratic(seed: int = 0, dim: int = 8):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(dim, dim, generator=g, dtype=D)
    a = m @ m.T / dim + 0.1 * torch.eye(dim, dtype=D)
    b = torch.randn(dim, generator=g, dtype=D)
    x0 = torch.randn(dim, generator=g, dtype=D)
    w0 = torch.randn(3, dim, generator=g, dtype=D)

    def loss(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Two parameter tensors so per-tensor state handling is exercised.
        return 0.5 * x @ a @ x + b @ x + 0.5 * (w @ a * w).sum()

    return loss, x0, w0


def run(make_opt: Callable, steps: int = 100) -> list[torch.Tensor]:
    loss, x0, w0 = random_quadratic()
    x, w = torch.nn.Parameter(x0.clone()), torch.nn.Parameter(w0.clone())
    opt = make_opt([x, w])

    def closure() -> torch.Tensor:
        opt.zero_grad()
        value = loss(x, w)
        value.backward()
        return value

    for _ in range(steps):
        opt.step(closure)
    return [x.detach(), w.detach()]


CASES = [
    (lambda p: SGD(p, 0.05), lambda p: torch.optim.SGD(p, 0.05)),
    (lambda p: SGD(p, 0.05, 0.01), lambda p: torch.optim.SGD(p, 0.05, weight_decay=0.01)),
    (lambda p: Momentum(p, 0.02, 0.9), lambda p: torch.optim.SGD(p, 0.02, momentum=0.9)),
    (
        lambda p: Momentum(p, 0.02, 0.9, 0.01),
        lambda p: torch.optim.SGD(p, 0.02, momentum=0.9, weight_decay=0.01),
    ),
    (
        lambda p: NAG(p, 0.02, 0.9),
        lambda p: torch.optim.SGD(p, 0.02, momentum=0.9, nesterov=True),
    ),
    (
        lambda p: NAG(p, 0.02, 0.9, 0.01),
        lambda p: torch.optim.SGD(p, 0.02, momentum=0.9, nesterov=True, weight_decay=0.01),
    ),
    (lambda p: RMSProp(p, 0.01), lambda p: torch.optim.RMSprop(p, 0.01, alpha=0.99, eps=1e-8)),
    (
        lambda p: RMSProp(p, 0.01, weight_decay=0.01),
        lambda p: torch.optim.RMSprop(p, 0.01, alpha=0.99, eps=1e-8, weight_decay=0.01),
    ),
    (lambda p: Adam(p, 0.05), lambda p: torch.optim.Adam(p, 0.05)),
    (
        lambda p: Adam(p, 0.05, (0.8, 0.99), 1e-6, 0.01),
        lambda p: torch.optim.Adam(p, 0.05, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.01),
    ),
]


@pytest.mark.parametrize("ours,reference", CASES)
def test_from_scratch_matches_torch(ours: Callable, reference: Callable) -> None:
    for a, b in zip(run(ours), run(reference), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_from_scratch_actually_optimizes() -> None:
    loss, x0, w0 = random_quadratic()
    for opt in [lambda p: SGD(p, 0.05), lambda p: Adam(p, 0.05)]:
        x, w = run(opt, steps=300)
        assert loss(x, w) < loss(x0, w0)


class Recorder(torch.optim.Optimizer):
    """Base optimizer that only records what SAM hands it."""

    def __init__(self, params) -> None:
        super().__init__(params, {})
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def step(self, closure=None) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                self.seen.append((p.detach().clone(), p.grad.clone()))


def test_sam_restores_parameters_and_uses_perturbed_gradient() -> None:
    a = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=D)
    x = torch.nn.Parameter(torch.tensor([0.7, -1.3], dtype=D))
    original = x.detach().clone()
    rho = 0.1
    opt = SAM([x], Recorder, rho=rho)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        value = 0.5 * x @ a @ x
        value.backward()
        return value

    opt.step(closure)
    theta_at_base_step, grad_seen = opt.base.seen[0]
    assert torch.equal(theta_at_base_step, original)  # bit-exact restore
    assert torch.equal(x.detach(), original)
    g = a @ original
    expected = a @ (original + rho * g / g.norm())  # gradient at the ascent point
    torch.testing.assert_close(grad_seen, expected, atol=1e-12, rtol=0)


def test_sam_step_requires_closure() -> None:
    x = torch.nn.Parameter(torch.zeros(2, dtype=D))
    with pytest.raises(ValueError):
        SAM([x], lambda g: SGD(g, 0.1)).step()


def test_adamw_decay_is_decoupled() -> None:
    lr, wd = 0.1, 0.5
    p0 = torch.tensor([2.0, -3.0, 0.5], dtype=D)

    def one_step(name: str) -> tuple[torch.Tensor, torch.optim.Optimizer]:
        p = torch.nn.Parameter(p0.clone())
        opt = build_optimizer(name, [p], {"lr": lr, "weight_decay": wd})
        p.grad = torch.zeros_like(p)  # loss gradient is zero: only decay acts
        opt.step()
        return p.detach(), opt

    p_adamw, opt = one_step("adamw")
    # Decoupled: theta <- theta (1 - lr wd); the decay never enters the moment estimates.
    torch.testing.assert_close(p_adamw, p0 * (1 - lr * wd), atol=1e-15, rtol=0)
    assert torch.count_nonzero(next(iter(opt.state.values()))["exp_avg_sq"]) == 0
    # Coupled L2 in Adam goes through normalisation: a near-sign step of size ~lr.
    p_adam, _ = one_step("adam")
    torch.testing.assert_close(p_adam, p0 - lr * torch.sign(p0), atol=1e-6, rtol=0)


def test_registry_builds_every_configured_optimizer(root: Path) -> None:
    configs = load_optimizers(root / "configs")
    assert set(BUILDERS) == set(configs) - {"newton"}
    for name in BUILDERS:
        tuned = {"lr": 1e-3} | ({"rho": 0.05} if name.startswith("sam") else {})
        x, w = run(lambda p, n=name, t=tuned: build_from_config(configs[n], p, t), steps=5)
        assert torch.isfinite(x).all() and torch.isfinite(w).all()


def test_registry_rejects_unknown_hparam() -> None:
    with pytest.raises(TypeError):
        build_optimizer("adam", [torch.nn.Parameter(torch.zeros(1))], {"lr": 1e-3, "mom": 0.9})
