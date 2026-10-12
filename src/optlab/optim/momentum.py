from __future__ import annotations

from typing import Any

from torch import Tensor

from optlab.optim._base import FromScratch


class Momentum(FromScratch):
    """Heavy ball, PyTorch form: v <- mu v + g;  theta <- theta - lr * direction(g, v)."""

    def __init__(
        self, params: Any, lr: float, momentum: float = 0.9, weight_decay: float = 0.0
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        super().__init__(params, lr, weight_decay, momentum=momentum)

    def direction(self, g: Tensor, v: Tensor, mu: float) -> Tensor:
        return v

    def update(self, p: Tensor, g: Tensor, group: dict[str, Any], state: dict[str, Any]) -> None:
        mu = group["momentum"]
        if "v" not in state:
            state["v"] = g.clone()  # v_0 = 0, so v_1 = g
        else:
            state["v"].mul_(mu).add_(g)
        p.add_(self.direction(g, state["v"], mu), alpha=-group["lr"])
