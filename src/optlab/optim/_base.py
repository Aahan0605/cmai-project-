"""Shared step loop for the from-scratch optimizers (FR-O2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class FromScratch(Optimizer):
    """Subclasses implement `update(p, g, group, state)` for one parameter tensor.

    Weight decay is coupled L2 (g <- g + wd * theta), matching torch.optim for SGD-family,
    RMSProp and Adam (TRD 6.3).
    """

    def __init__(self, params: Any, lr: float, weight_decay: float, **defaults: Any) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, **defaults))

    def update(self, p: Tensor, g: Tensor, group: dict[str, Any], state: dict[str, Any]) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["weight_decay"]:
                    g = g.add(p, alpha=group["weight_decay"])
                self.update(p, g, group, self.state[p])
        return loss
