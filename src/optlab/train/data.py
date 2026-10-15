"""Whole splits as in-memory tensors. Heads are tiny, so batches are sliced from RAM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from optlab.data.cache import FeatureCache
from optlab.data.dataset import FeatureDataset


@dataclass
class Split:
    x: dict[str, Tensor]  # modality -> (n_frames, dim) float32
    y: Tensor  # (n_frames,) float32, 1 = fake
    video_ids: list[str]

    def __post_init__(self) -> None:
        if self.y.ndim != 1 or len(self.video_ids) != len(self.y):
            raise ValueError("split labels and video IDs must be aligned vectors")
        if not torch.isin(self.y, torch.tensor([0, 1], device=self.y.device)).all():
            raise ValueError("split labels must be binary (0 = real, 1 = fake)")
        if not self.x or any(
            x.ndim != 2 or len(x) != len(self.y) or not torch.isfinite(x).all()
            for x in self.x.values()
        ):
            raise ValueError("split features must be finite matrices aligned with labels")

    def __len__(self) -> int:
        return len(self.y)


def cache_dirs(features_dir: Path, dataset: str, backbones: dict[str, str]) -> dict[str, Path]:
    return {m: features_dir / dataset / b for m, b in backbones.items()}


def load_split(
    manifest: Path, split: str, caches: dict[str, Path], val_role: str | None = None
) -> Split:
    """`caches` maps modality ('visual', 'audio', 'frequency') to a feature-cache directory."""
    opened = {m: FeatureCache(p) for m, p in caches.items()}
    ds = FeatureDataset(
        manifest,
        split,
        opened["visual"],
        frequency=opened.get("frequency"),
        audio=opened.get("audio"),
        val_role=val_role,
    )
    if len(ds) == 0:
        raise ValueError(f"no {split} frames (val_role={val_role}) in {manifest}")
    items = [ds[i] for i in range(len(ds))]
    x = {"visual": torch.stack([it[0] for it in items])}
    if "audio" in opened:
        x["audio"] = torch.stack([it[1] for it in items])
    if "frequency" in opened:
        x["frequency"] = torch.stack([it[2] for it in items])
    y = torch.tensor([it[3] for it in items], dtype=torch.float32)
    return Split(x, y, [it[4] for it in items])
