"""`FeatureDataset` over cached features (TRD 3.3)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler

from optlab.data.cache import FeatureCache
from optlab.data.manifest import read_manifest

Item = tuple[Tensor, Tensor | None, Tensor | None, int, str]


class FeatureDataset(Dataset[Item]):
    """One item per visual frame: (x_visual, x_audio | None, x_freq | None, label, video_id).

    Only videos of the requested split (and val role) that have visual features are used.
    The per-video audio embedding is broadcast to every frame of that video. Frequency
    features must cover exactly the same (video, frame) pairs as the visual ones.
    """

    def __init__(
        self,
        manifest: Path,
        split: str,
        visual: FeatureCache,
        frequency: FeatureCache | None = None,
        audio: FeatureCache | None = None,
        val_role: str | None = None,
    ) -> None:
        rows = [
            r
            for r in read_manifest(manifest)
            if r.split == split and (val_role is None or r.val_role == val_role)
        ]
        labels = {r.video_id: r.label for r in rows}
        self.visual, self.frequency, self.audio = visual, frequency, audio
        self.items = [(v, f, row) for v, _, f, row in visual.index if v in labels]
        self.labels = [labels[v] for v, _, _ in self.items]
        for cache in (visual, frequency, audio):
            if cache is not None:
                for vid, label, _, _ in cache.index:
                    if vid in labels and label != labels[vid]:
                        raise ValueError(f"label mismatch for {vid} between manifest and cache")
        self.freq_row = (
            {(v, f): row for v, _, f, row in frequency.index} if frequency is not None else None
        )
        self.audio_row = {v: row for v, _, _, row in audio.index} if audio is not None else None
        if self.audio_row is not None:
            if len(self.audio_row) != len(audio.index):
                raise ValueError("audio cache must contain exactly one embedding per video")
            missing_audio = {v for v, _, _ in self.items} - self.audio_row.keys()
            if missing_audio:
                raise ValueError(f"missing audio features, e.g. {sorted(missing_audio)[0]}")
        if self.freq_row is not None:
            missing = [(v, f) for v, f, _ in self.items if (v, f) not in self.freq_row]
            if missing:
                raise ValueError(
                    f"{len(missing)} frames lack frequency features, e.g. {missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Item:
        vid, frame, row = self.items[i]
        x_visual = torch.from_numpy(self.visual[row])
        x_freq = (
            torch.from_numpy(self.frequency[self.freq_row[(vid, frame)]])
            if self.frequency is not None and self.freq_row is not None
            else None
        )
        x_audio = None
        if self.audio is not None and self.audio_row is not None:
            if vid not in self.audio_row:
                raise KeyError(f"no audio features for {vid}")
            x_audio = torch.from_numpy(self.audio[self.audio_row[vid]])
        return x_visual, x_audio, x_freq, self.labels[i], vid


def collate(batch: list[Item]) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor, list[str]]:
    xv, xa, xf, y, vid = zip(*batch, strict=True)
    stack = lambda xs: None if xs[0] is None else torch.stack(xs)  # noqa: E731
    return torch.stack(xv), stack(xa), stack(xf), torch.tensor(y, dtype=torch.float32), list(vid)


def balanced_sampler(dataset: FeatureDataset, generator: torch.Generator) -> WeightedRandomSampler:
    """Each class drawn with equal probability (FF++ has several fakes per real)."""
    counts = Counter(dataset.labels)
    weights = torch.tensor([1.0 / counts[y] for y in dataset.labels], dtype=torch.float64)
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )
