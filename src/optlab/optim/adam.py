from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from optlab.optim._base import FromScratch


class Adam(FromScratch):
    """m <- b1 m + (1-b1) g;  v <- b2 v + (1-b2) g^2;
    theta <- theta - lr mhat / (sqrt(vhat) + eps), mhat = m/(1-b1^t), vhat = v/(1-b2^t)."""

    def __init__(
        self,
        params: Any,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay, betas=tuple(betas), eps=eps)

    def update(self, p: Tensor, g: Tensor, group: dict[str, Any], state: dict[str, Any]) -> None:
        b1, b2 = group["betas"]
        if not state:
            state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
        state["t"] += 1
        t, m, v = state["t"], state["m"], state["v"]
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])
