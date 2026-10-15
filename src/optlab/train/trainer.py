"""Head training loop (TRD 5): one code path for every optimizer via `optimizer.step(closure)`."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce

from optlab.config import OptimizerConfig
from optlab.eval.metrics import video_scores
from optlab.models.heads import LateFusion
from optlab.optim.registry import build_from_config
from optlab.train.data import Split


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    lr: float
    grad_norm: float  # mean over the epoch's steps of the global pre-clip gradient norm
    step_time_sec: float  # mean seconds per optimizer step


@dataclass
class TrainResult:
    status: str  # 'completed' | 'diverged'
    epochs: list[EpochLog] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_auc: float | None = None
    best_val_loss: float | None = None
    early_stopped: bool = False
    state_dict: dict[str, Tensor] | None = None  # weights at the best epoch
    wall_clock_sec: float = 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(dims: dict[str, int], model_extra: dict[str, Any]) -> LateFusion:
    return LateFusion(
        dims, hidden=model_extra.get("head_hidden", 256), dropout=model_extra.get("dropout", 0.2)
    )


def loss_fn(logits: Tensor, y: Tensor) -> Tensor:
    """sum_m BCE(z_m, y): heads have disjoint parameters, so each learns its own stream."""
    return bce(logits, y[:, None].expand_as(logits), reduction="none").mean(0).sum()


@torch.no_grad()
def evaluate_split(model: LateFusion, split: Split, batch_size: int = 8192) -> tuple[float, float]:
    """(validation loss, video-level AUC of the uniformly fused logit)."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    device = next(model.parameters()).device
    logits = torch.cat(
        [
            model({m: x[i : i + batch_size].to(device) for m, x in split.x.items()}).cpu()
            for i in range(0, len(split), batch_size)
        ]
    )
    loss = float(loss_fn(logits, split.y))
    vid, labels, _ = video_scores(
        logits.mean(1).numpy().astype(np.float64), split.y.numpy(), split.video_ids
    )
    auc = float(roc_auc_score(labels, vid)) if len(set(labels)) == 2 else float("nan")
    return loss, auc


def _clip_per_head(model: LateFusion, clip_norm: float | None) -> float:
    """Global gradient norm before clipping; clipping (if enabled) is applied per head."""
    norms = []
    for head in model.heads.values():
        params = [p for p in head.parameters() if p.grad is not None]
        if clip_norm is not None:
            norms.append(torch.nn.utils.clip_grad_norm_(params, clip_norm))
        else:
            norms.append(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params])))
    return float(torch.linalg.vector_norm(torch.stack(norms)))


def train(
    train_split: Split,
    val_split: Split,
    opt_cfg: OptimizerConfig,
    hparams: dict[str, Any],
    seed: int,
    max_epochs: int,
    batch_size: int,
    patience: int,
    clip_norm: float | None,
    model_extra: dict[str, Any],
    on_epoch: Callable[[EpochLog], None] | None = None,
    device: str = "cpu",
) -> TrainResult:
    """Train the heads; stop early on val_tune video AUC; keep the best weights.

    Seed controls parameter init and data order (TRD 5). Sampling is class-balanced with
    replacement, one epoch = len(train) samples. NaN/Inf loss -> status 'diverged'.
    """
    for name, split in (("train", train_split), ("val_tune", val_split)):
        if set(split.y.tolist()) != {0, 1}:
            raise ValueError(f"{name} must contain both real and fake examples")
    if max_epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch size and patience must be positive")
    if {m: x.shape[1] for m, x in train_split.x.items()} != {
        m: x.shape[1] for m, x in val_split.x.items()
    }:
        raise ValueError("training and validation feature dimensions must match")
    seed_everything(seed)
    dims = {m: x.shape[1] for m, x in train_split.x.items()}
    model = build_model(dims, model_extra).to(device)
    opt = build_from_config(opt_cfg, model.param_groups(), hparams)
    gen = torch.Generator().manual_seed(seed)
    labels = train_split.y
    class_weight = torch.where(labels == 1, 1 / labels.sum(), 1 / (len(labels) - labels.sum()))

    result = TrainResult(status="completed")
    best_state, since_best = None, 0
    start = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        order = torch.multinomial(class_weight, len(labels), replacement=True, generator=gen)
        losses, gnorms, step_times = [], [], []
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            xb = {m: x[idx].to(device) for m, x in train_split.x.items()}
            yb = labels[idx].to(device)
            norm_box: list[float] = []

            def closure(
                xb: dict[str, Tensor] = xb, yb: Tensor = yb, box: list = norm_box
            ) -> Tensor:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                box.append(_clip_per_head(model, clip_norm))  # first call = norm at theta
                return loss

            t0 = time.perf_counter()
            loss = opt.step(closure)
            step_times.append(time.perf_counter() - t0)
            if not math.isfinite(float(loss.detach())):
                result.status = "diverged"
                result.wall_clock_sec = time.perf_counter() - start
                return result
            losses.append(float(loss.detach()))
            gnorms.append(norm_box[0])
        val_loss, val_auc = evaluate_split(model, val_split)
        if not (
            math.isfinite(val_loss)
            and math.isfinite(val_auc)
            and all(math.isfinite(float(p.detach().sum())) for p in model.parameters())
        ):
            result.status = "diverged"
            result.wall_clock_sec = time.perf_counter() - start
            return result
        log = EpochLog(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            val_loss=val_loss,
            val_auc=val_auc,
            lr=float(opt.param_groups[0]["lr"]),
            grad_norm=float(np.mean(gnorms)),
            step_time_sec=float(np.mean(step_times)),
        )
        result.epochs.append(log)
        if on_epoch:
            on_epoch(log)
        if result.best_val_auc is None or val_auc > result.best_val_auc:
            result.best_epoch, result.best_val_auc, result.best_val_loss = epoch, val_auc, val_loss
            best_state, since_best = copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= patience:
                result.early_stopped = True
                break
    result.state_dict = best_state
    result.wall_clock_sec = time.perf_counter() - start
    return result
