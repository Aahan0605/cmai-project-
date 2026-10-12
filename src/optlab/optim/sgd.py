from __future__ import annotations

from typing import Any

from torch import Tensor

from optlab.optim._base import FromScratch


class SGD(FromScratch):
    """theta <- theta - lr * g"""

    def __init__(self, params: Any, lr: float, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr, weight_decay)

    def update(self, p: Tensor, g: Tensor, group: dict[str, Any], state: dict[str, Any]) -> None:
        p.add_(g, alpha=-group["lr"])
