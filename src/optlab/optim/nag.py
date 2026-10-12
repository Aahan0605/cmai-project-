from __future__ import annotations

from torch import Tensor

from optlab.optim.momentum import Momentum


class NAG(Momentum):
    """Nesterov, PyTorch form: v <- mu v + g;  theta <- theta - lr (g + mu v)."""

    def direction(self, g: Tensor, v: Tensor, mu: float) -> Tensor:
        return g.add(v, alpha=mu)
