from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from optlab.optim._base import FromScratch


class RMSProp(FromScratch):
    """s <- alpha s + (1 - alpha) g^2;  theta <- theta - lr g / (sqrt(s) + eps)."""

    def __init__(
        self,
        params: Any,
        lr: float,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay, alpha=alpha, eps=eps)

    def update(self, p: Tensor, g: Tensor, group: dict[str, Any], state: dict[str, Any]) -> None:
        s = state.setdefault("s", torch.zeros_like(p))
        s.mul_(group["alpha"]).addcmul_(g, g, value=1 - group["alpha"])
        p.addcdiv_(g, s.sqrt().add_(group["eps"]), value=-group["lr"])
