"""Toy benchmark functions with known minima (TRD 10). All work on a 1-D float64 tensor."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from optlab.config import ToyBenchmarkConfig

Fn = Callable[[Tensor], Tensor]


def quadratic_matrix(dim: int, condition_number: float, rotation_seed: int) -> Tensor:
    """A = Q diag(lambda) Q^T, eigenvalues log-spaced in [1, kappa], Q a fixed random rotation
    (rotated so that diagonal preconditioners such as Adam get no free axis alignment)."""
    eig = torch.logspace(0, math.log10(condition_number), dim, dtype=torch.float64)
    g = torch.Generator().manual_seed(rotation_seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))
    return q @ torch.diag(eig) @ q.T


def make_function(cfg: ToyBenchmarkConfig) -> Fn:
    if cfg.kind == "quadratic":
        a = quadratic_matrix(cfg.dim, cfg.condition_number or 1.0, cfg.rotation_seed)
        return lambda x: 0.5 * x @ a @ x
    if cfg.kind == "rosenbrock":
        return lambda x: (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
    if cfg.kind == "rastrigin":
        return lambda x: 10 * x.numel() + (x**2 - 10 * torch.cos(2 * math.pi * x)).sum()
    raise ValueError(cfg.kind)


def minimizer(cfg: ToyBenchmarkConfig) -> np.ndarray:
    return np.ones(cfg.dim) if cfg.kind == "rosenbrock" else np.zeros(cfg.dim)


def start_point(cfg: ToyBenchmarkConfig, seed: int) -> np.ndarray:
    """Fixed start per (benchmark, seed): uniform in [start_low, start_high]^dim."""
    return np.random.default_rng(seed).uniform(cfg.start_low, cfg.start_high, cfg.dim)
