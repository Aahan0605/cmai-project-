"""OptLab blended Adam/momentum with Lookahead. See docs/HYBRID_DESIGN.md.

An experimental combination of established techniques, not a novelty or superiority claim.
The normalized momentum branch uses one scalar RMS match per parameter group, preserving
momentum's direction while making the two branches comparable in magnitude.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class Hybrid(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        blend: float = 0.5,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
    ):
        beta1, beta2 = betas
        if not all(
            math.isfinite(float(v))
            for v in (lr, beta1, beta2, eps, weight_decay, blend, lookahead_alpha)
        ):
            raise ValueError("hybrid hyperparameters must be finite")
        if lr <= 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("lr/eps must be positive; weight_decay must be nonnegative")
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and 0 <= blend <= 1):
            raise ValueError("invalid beta or blend")
        if not isinstance(lookahead_k, int) or lookahead_k < 0 or not 0 < lookahead_alpha <= 1:
            raise ValueError("invalid Lookahead interval or interpolation")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=tuple(betas),
                eps=eps,
                weight_decay=weight_decay,
                blend=blend,
                lookahead_k=lookahead_k,
                lookahead_alpha=lookahead_alpha,
                hybrid_step=0,
            ),
        )
        # Slow weights must start at construction, even for intermittently missing gradients.
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["slow"] = p.detach().clone()

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            active = [p for p in group["params"] if p.grad is not None]
            if not active:
                continue
            b1, b2 = group["betas"]
            directions = []
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("Hybrid does not support sparse gradients")
                state = self.state[p]
                if "m" not in state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["step"] += 1
                state["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                state["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m = state["m"] / (1 - b1 ** state["step"])
                v = state["v"] / (1 - b2 ** state["step"])
                adam = m / (v.sqrt() + group["eps"])
                directions.append((p, m, adam))
            anorm = torch.sqrt(sum(a.square().sum() for _, _, a in directions))
            mnorm = torch.sqrt(sum(m.square().sum() for _, m, _ in directions))
            match = anorm / (mnorm + group["eps"])
            for p, m, adam in directions:
                p.mul_(1 - group["lr"] * group["weight_decay"])
                direction = group["blend"] * adam + (1 - group["blend"]) * match * m
                p.add_(direction, alpha=-group["lr"])
            group["hybrid_step"] += 1
            k = group["lookahead_k"]
            if k and group["hybrid_step"] % k == 0:
                for p in group["params"]:
                    slow = self.state[p]["slow"]
                    slow.lerp_(p, group["lookahead_alpha"])
                    p.copy_(slow)
        return loss
