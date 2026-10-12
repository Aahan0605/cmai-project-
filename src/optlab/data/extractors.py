"""Frozen feature extractors (TRD 3.2 step 4). Each maps a batch to an (n, dim) float array.

The frequency stream is hand-built (no backbone). CLIP and wav2vec 2.0 need the `[extract]`
extra and download pretrained weights on first use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


class Extractor(Protocol):
    name: str

    def __call__(self, batch: list[np.ndarray]) -> np.ndarray: ...


class FrequencyExtractor:
    """Azimuthally averaged log power spectrum of the grayscale face crop.

    The crop is resized to `size` x `size`; |FFT|^2 is averaged over rings of equal radius
    from the zero frequency, binned into `bins` radial bands up to the Nyquist radius, then
    log(1 + .)-transformed. Output: one `bins`-vector per crop.
    """

    name = "freq_spectrum"

    def __init__(self, size: int = 128, bins: int = 64) -> None:
        self.size, self.bins = size, bins
        yy, xx = np.indices((size, size)) - size // 2
        radius = np.hypot(xx, yy)
        edges = np.linspace(0, size // 2, bins + 1)
        self.band = np.clip(np.digitize(radius, edges) - 1, 0, bins - 1)  # r = Nyquist -> last
        self.band[radius > size // 2] = -1  # corners beyond Nyquist are dropped
        self.counts = np.bincount(self.band[self.band >= 0].ravel(), minlength=bins)

    def __call__(self, batch: list[np.ndarray]) -> np.ndarray:
        out = np.empty((len(batch), self.bins), dtype=np.float32)
        for i, rgb in enumerate(batch):
            gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gray = F.interpolate(
                torch.from_numpy(gray)[None, None], size=(self.size, self.size),
                mode="bilinear", antialias=True,
            )[0, 0].numpy() / 255.0
            power = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
            mask = self.band >= 0
            sums = np.bincount(self.band[mask], weights=power[mask], minlength=self.bins)
            out[i] = np.log1p(sums / np.maximum(self.counts, 1))
        return out


class ClipVisualExtractor:
    """CLIP image encoder via open_clip, one embedding per face crop."""

    def __init__(self, name: str, arch: str, pretrained: str, device: str = "cpu") -> None:
        import open_clip

        self.name, self.device = name, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        self.model.eval()

    @torch.no_grad()
    def __call__(self, batch: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        x = torch.stack([self.preprocess(Image.fromarray(img)) for img in batch]).to(self.device)
        return self.model.encode_image(x).float().cpu().numpy()


class Wav2Vec2AudioExtractor:
    """wav2vec 2.0 via transformers; mean-pooled last hidden state, one vector per clip."""

    def __init__(self, name: str, hf_id: str, sample_rate: int, device: str = "cpu") -> None:
        from transformers import AutoFeatureExtractor, Wav2Vec2Model

        self.name, self.device, self.sample_rate = name, device, sample_rate
        self.processor = AutoFeatureExtractor.from_pretrained(hf_id)
        self.model = Wav2Vec2Model.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def __call__(self, batch: list[np.ndarray]) -> np.ndarray:
        out = []
        for wav in batch:  # one clip at a time: lengths differ and clips are long
            inputs = self.processor(wav, sampling_rate=self.sample_rate, return_tensors="pt")
            hidden = self.model(inputs.input_values.to(self.device)).last_hidden_state
            out.append(hidden.mean(dim=1)[0].float().cpu().numpy())
        return np.stack(out)
