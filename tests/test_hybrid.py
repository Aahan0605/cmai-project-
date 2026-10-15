import copy

import pytest
import torch

from optlab.optim.hybrid import Hybrid


def test_adamw_limit_matches_torch():
    torch.manual_seed(4)
    p = torch.nn.Parameter(torch.randn(7, dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    ours = Hybrid([p], lr=0.01, weight_decay=0.03, blend=1, lookahead_k=0)
    reference = torch.optim.AdamW([q], lr=0.01, weight_decay=0.03)
    for _ in range(100):
        gradient = torch.randn_like(p)
        p.grad, q.grad = gradient.clone(), gradient.clone()
        ours.step()
        reference.step()
    torch.testing.assert_close(p, q, atol=1e-12, rtol=0)


def test_first_step_matches_written_equation():
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float64))
    gradient = torch.tensor([1.0, 4.0], dtype=torch.float64)
    p.grad = gradient
    original = p.detach().clone()
    adam = gradient / (gradient.abs() + 1e-8)
    r = adam.norm() / (gradient.norm() + 1e-8)
    expected = 0.998 * original - 0.1 * (0.5 * adam + 0.5 * r * gradient)
    Hybrid([p], lr=0.1, weight_decay=0.02).step()
    torch.testing.assert_close(p, expected, atol=1e-12, rtol=0)


def test_lookahead_interpolates_from_initial_parameters():
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    opt = Hybrid([p], lr=0.1, blend=1, lookahead_k=2, lookahead_alpha=0.5)
    for _ in range(2):
        p.grad = torch.ones_like(p)
        opt.step()
    assert p.item() == pytest.approx(0.9, abs=1e-8)
    assert torch.equal(p, opt.state[p]["slow"])


def test_checkpoint_continuation_and_groups():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    opt = Hybrid([p], lr=0.01)

    def step(param, optimizer):
        param.grad = param.detach().square()
        optimizer.step()

    for _ in range(3):
        step(p, opt)
    q = torch.nn.Parameter(p.detach().clone())
    resumed = Hybrid([q], lr=0.01)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for _ in range(8):
        step(p, opt)
        step(q, resumed)
    torch.testing.assert_close(p, q, atol=0, rtol=0)
    a = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    c = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    grouped = Hybrid([{"params": [a]}, {"params": [b]}], lr=0.01)
    isolated = Hybrid([c], lr=0.01)
    for _ in range(10):
        a.grad = c.grad = torch.tensor([1.0, 4.0], dtype=torch.float64)
        b.grad = torch.tensor([100.0, 0.001], dtype=torch.float64)
        grouped.step()
        isolated.step()
    torch.testing.assert_close(a, c, atol=0, rtol=0)


def test_zero_gradient_is_finite():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = Hybrid([p], lr=0.01)
    for _ in range(10):
        p.grad = torch.zeros_like(p)
        opt.step()
    assert torch.equal(p, torch.ones_like(p))


@pytest.mark.parametrize(
    "kwargs", [{"blend": 1.1}, {"lookahead_k": -1}, {"eps": 0}, {"lr": float("nan")}]
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        Hybrid([torch.nn.Parameter(torch.ones(1))], **({"lr": 0.01} | kwargs))
