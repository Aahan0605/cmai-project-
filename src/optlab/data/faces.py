"""Face detection and cropping (TRD 3.2 step 2). Detectors are pluggable via `FaceDetector`."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class FaceDetector(Protocol):
    def detect(self, frames: list[np.ndarray]) -> list[Box | None]:
        """One box (the largest face) or None per RGB uint8 frame."""
        ...


class MTCNNDetector:
    """MTCNN from facenet-pytorch (`[extract]` extra), largest face per frame."""

    def __init__(self, device: str = "cpu") -> None:
        from facenet_pytorch import MTCNN

        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)

    def detect(self, frames: list[np.ndarray]) -> list[Box | None]:
        boxes, _ = self.mtcnn.detect(frames)
        return [None if b is None else tuple(float(v) for v in b[0]) for b in boxes]


def crop_face(frame: np.ndarray, box: Box, margin: float, size: int) -> np.ndarray:
    """Square crop around the box, grown by `margin` of its longer side on every side, clipped
    to the frame, resized to size x size (bilinear, antialiased). Returns RGB uint8."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = frame.shape[:2]
    left, top = int(max(0, round(cx - side / 2))), int(max(0, round(cy - side / 2)))
    right, bottom = int(min(w, round(cx + side / 2))), int(min(h, round(cy + side / 2)))
    if right <= left or bottom <= top:
        raise ValueError(f"box {box} lies outside the {w}x{h} frame")
    crop = torch.from_numpy(np.ascontiguousarray(frame[top:bottom, left:right]))
    crop = crop.permute(2, 0, 1)[None].float()
    resized = F.interpolate(crop, size=(size, size), mode="bilinear", antialias=True)
    return resized[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
