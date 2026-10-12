"""One registry: optimizer name + hyperparameter dict -> torch.optim.Optimizer (FR-O1).

Builders take hyperparameters as keyword arguments, so an unknown or misspelled key raises
TypeError instead of being silently ignored. Fixed values (`OptimizerConfig.fixed`) and tuned
values are merged by `build_from_config`. Newton is toy-only and lives in `optlab.toy.runner`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
from torch.optim import Optimizer

from optlab.config import OptimizerConfig
from optlab.optim.adam import Adam
from optlab.optim.momentum import Momentum
from optlab.optim.nag import NAG
from optlab.optim.rmsprop import RMSProp
from optlab.optim.sam import SAM
from optlab.optim.sgd import SGD

Builder = Callable[..., Optimizer]


def _sam_sgd_momentum(
    params: Any, lr: float, rho: float = 0.05, momentum: float = 0.9, weight_decay: float = 0.0
) -> Optimizer:
    return SAM(params, lambda groups: Momentum(groups, lr, momentum, weight_decay), rho=rho)


def _adabelief(
    params: Any,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-16,
    weight_decay: float = 0.0,
    weight_decouple: bool = True,
    rectify: bool = False,
) -> Optimizer:
    return AdaBelief(
        params,
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        weight_decay=weight_decay,
        weight_decouple=weight_decouple,
        rectify=rectify,
        print_change_log=False,
    )


def _with_tuple_betas(cls: type[Optimizer]) -> Builder:
    # weight_decay defaults to 0 here: torch.optim.AdamW would otherwise apply 0.01 silently.
    def build(
        params: Any, lr: float, betas: Any = (0.9, 0.999), weight_decay: float = 0.0, **kw: Any
    ) -> Optimizer:
        return cls(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay, **kw)

    return build


BUILDERS: dict[str, Builder] = {
    "sgd": SGD,
    "sgd_momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSProp,
    "adam": Adam,
    "adamw": _with_tuple_betas(torch.optim.AdamW),
    "radam": _with_tuple_betas(torch.optim.RAdam),
    "adabelief": _adabelief,
    "lion": _with_tuple_betas(Lion),
    "sam_sgd_momentum": _sam_sgd_momentum,
    "lbfgs": torch.optim.LBFGS,
}


def build_optimizer(name: str, params: Any, hparams: Mapping[str, Any]) -> Optimizer:
    if name not in BUILDERS:
        raise KeyError(f"unknown optimizer {name!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[name](params, **hparams)


def build_from_config(cfg: OptimizerConfig, params: Any, tuned: Mapping[str, Any]) -> Optimizer:
    return build_optimizer(cfg.name, params, {**cfg.fixed, **tuned})
