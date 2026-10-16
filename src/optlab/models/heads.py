"""Late-fusion detector (TRD 4): one small MLP head per modality, fused on the simplex."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from optlab.config import ModelConfig


def modalities(cfg: ModelConfig) -> list[str]:
    """Active streams, in a fixed order shared by heads, caches and fusion weights."""
    return (
        ["visual"]
        + (["audio"] if cfg.audio_backbone else [])
        + (["frequency"] if cfg.uses_frequency_branch else [])
    )


class LateFusion(nn.Module):
    """z_m(x) per modality; fused logit z(x) = sum_m w_m z_m(x) with w on the simplex.

    The heads are trained jointly on sum_m BCE(z_m, y); `w` is fitted afterwards on the
    val_fusion subset (optlab.fusion) and is not a trainable parameter.
    """

    def __init__(self, dims: dict[str, int], hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        # No batch norm: keeps SAM and clipping well-defined per sample batch (TRD 4).
        self.heads = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
                )
                for m, d in dims.items()
            }
        )
        self.register_buffer("weights", torch.full((len(dims),), 1.0 / len(dims)))

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        """Per-modality logits, shape (batch, n_modalities), in `self.heads` order."""
        return torch.cat([head(x[m]) for m, head in self.heads.items()], dim=1)

    def fused(self, logits: Tensor) -> Tensor:
        return logits @ self.weights

    def param_groups(self) -> list[dict]:
        """One parameter group per head, so SAM's norm and clipping stay per head (TRD 4)."""
        return [
            {"params": list(head.parameters()), "modality": m} for m, head in self.heads.items()
        ]


class JointHead(LateFusion):
    """Train-only standardized concatenation MLP or linear soft-margin SVM.

    SVM objective: 0.5 ||w||² + C mean(max(0, 1-y*z)); intercept is unpenalized.
    It is optimized through the same gradient loop, rather than solving its dual explicitly.
    """

    def __init__(self, dims, hidden=256, dropout=0.2, kind="concat_mlp", svm_c=1.0):
        super().__init__({"joint": sum(dims.values())}, hidden, dropout)
        self.input_modalities = list(dims)
        self.kind, self.svm_c = kind, svm_c
        if kind == "svm_head":
            self.heads["joint"] = nn.Linear(sum(dims.values()), 1)
        self.register_buffer("center", torch.zeros(sum(dims.values())))
        self.register_buffer("scale", torch.ones(sum(dims.values())))

    @torch.no_grad()
    def fit_standardizer(self, x):
        features = torch.cat([x[m] for m in self.input_modalities], dim=1)
        self.center.copy_(features.mean(0))
        self.scale.copy_(features.std(0, unbiased=False).clamp_min(1e-6))

    def forward(self, x):
        features = torch.cat([x[m] for m in self.input_modalities], dim=1)
        return super().forward({"joint": (features - self.center) / self.scale})

    def objective(self, logits, y):
        if self.kind != "svm_head":
            return torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], y)
        hinge = torch.relu(1 - (2 * y - 1) * logits[:, 0]).mean()
        penalty = 0.5 * self.heads["joint"].weight.square().sum().to(logits.device)
        return penalty + self.svm_c * hinge
