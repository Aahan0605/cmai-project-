"""Extraction pipeline: manifest -> sampled frames -> face crops -> cached features.

Readers, detector and extractors are injected so the pipeline is testable without videos,
OpenCV or pretrained weights. Feature caches are written to `{out}/{dataset}/{backbone}/`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from optlab.config import ExtractionConfig
from optlab.data.cache import FeatureCacheWriter
from optlab.data.extractors import Extractor
from optlab.data.faces import FaceDetector, crop_face
from optlab.data.manifest import read_manifest

FrameReader = Callable[[Path, int], tuple[list[int], list[np.ndarray]]]
AudioLoader = Callable[[Path, int], np.ndarray]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(manifest: Path, backbone: str, cfg: ExtractionConfig, **extra: Any) -> dict[str, Any]:
    # Recorded next to the cache: `artifacts` rows need a run or toy run (schema CHECK).
    return {
        "backbone": backbone,
        "manifest": manifest.as_posix(),
        "manifest_sha256": sha256(manifest),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def extract_visual(
    manifest: Path,
    video_root: Path,
    out_dir: Path,
    cfg: ExtractionConfig,
    detector: FaceDetector,
    extractor: Extractor,
    frequency: Extractor | None,
    read_frames: FrameReader,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Visual embeddings (and frequency vectors from the same crops) per sampled frame.

    Frames without a detected face are dropped; videos with no face at all are skipped and
    counted in meta.json.
    """
    preprocessing = {
        "frames_per_video": cfg.frames_per_video,
        "face_detector": cfg.face_detector,
        "face_margin": cfg.face_margin,
        "face_size": cfg.face_size,
    }
    writers = {extractor.name: (extractor, FeatureCacheWriter(out_dir / extractor.name, cfg.shard_rows))}
    if frequency is not None:
        writers[frequency.name] = (frequency, FeatureCacheWriter(out_dir / frequency.name, cfg.shard_rows))
    stats = {"videos": 0, "skipped_no_face": 0, "frames": 0, "frames_no_face": 0}
    rows = read_manifest(manifest)
    for n, row in enumerate(rows):
        indices, frames = read_frames(video_root / row.path, cfg.frames_per_video)
        boxes = detector.detect(frames) if frames else []
        kept = [(i, crop_face(f, b, cfg.face_margin, cfg.face_size))
                for i, f, b in zip(indices, frames, boxes, strict=True) if b is not None]
        stats["frames_no_face"] += len(frames) - len(kept)
        if not kept:
            stats["skipped_no_face"] += 1
            continue
        idx, crops = [k[0] for k in kept], [k[1] for k in kept]
        for ext, writer in writers.values():
            writer.add(row.video_id, row.label, idx, ext(crops))
        stats["videos"] += 1
        stats["frames"] += len(kept)
        if (n + 1) % 100 == 0:
            log(f"{n + 1}/{len(rows)} videos")
    for name, (_, writer) in writers.items():
        extra = {"preprocessing": preprocessing, "stats": stats}
        if name == "freq_spectrum":
            extra["freq_size"], extra["freq_bins"] = cfg.freq_size, cfg.freq_bins
        writer.close(_meta(manifest, name, cfg, **extra))
    return stats


def extract_audio(
    manifest: Path,
    video_root: Path,
    out_dir: Path,
    cfg: ExtractionConfig,
    extractor: Extractor,
    load_audio: AudioLoader,
) -> dict[str, int]:
    """One audio embedding per video (frame_idx 0 in the index)."""
    writer = FeatureCacheWriter(out_dir / extractor.name, cfg.shard_rows)
    rows = read_manifest(manifest)
    for row in rows:
        wav = load_audio(video_root / row.path, cfg.audio_sample_rate)
        writer.add(row.video_id, row.label, [0], extractor([wav]))
    stats = {"videos": len(rows)}
    writer.close(_meta(manifest, extractor.name, cfg, sample_rate=cfg.audio_sample_rate, stats=stats))
    return stats
