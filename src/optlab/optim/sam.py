from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021) as a wrapper over any base optimizer.

    e = rho g / ||g||;  g' = grad L(theta + e);  restore theta exactly;  base step with g'.
    ||g|| is taken per parameter group, so separate heads (groups) stay independent (TRD 4).
    Costs two forward/backward passes per step; `step` requires a closure that zeroes grads,
    computes the loss, calls backward and returns the loss.
    """

    def __init__(
        self, params: Any, base_optimizer: Callable[[Any], Optimizer], rho: float = 0.05
    ) -> None:
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        super().__init__(params, dict(rho=rho))
        self.base = base_optimizer(self.param_groups)
        self.param_groups = self.base.param_groups

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor:  # type: ignore[override]
        if closure is None:
            raise ValueError("SAM needs a closure")
        with torch.enable_grad():
            loss = closure()
        saved: list[tuple[Tensor, Tensor]] = []
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]))
            scale = group["rho"] / (norm + 1e-12)
            for p in params:
                saved.append((p, p.detach().clone()))
                p.add_(p.grad, alpha=float(scale))
        with torch.enable_grad():
            closure()
        for p, theta in saved:
            p.copy_(theta)
        self.base.step()
        return loss
