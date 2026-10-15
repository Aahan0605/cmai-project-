"""Feature cache (TRD 3.2 step 5): float16 .npy shards + index.csv + meta.json per
`{dataset}/{backbone}` directory. Read back with memmap and cast to float32 per access."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

INDEX_FIELDS = ["video_id", "label", "frame_idx", "row"]


class FeatureCacheWriter:
    """Row r lives in shard r // shard_rows at offset r % shard_rows."""

    def __init__(self, out_dir: Path, shard_rows: int = 65536) -> None:
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
            raise FileExistsError(f"{out_dir} is not empty; choose a new cache directory")
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(prefix=".optlab-cache-", dir=out_dir.parent)
        self._staging = Path(self._temp.name) / "data"
        self._staging.mkdir()
        self.out_dir, self.shard_rows = out_dir, shard_rows
        self.index: list[tuple[str, int, int, int]] = []
        self.pending: list[np.ndarray] = []
        self.shards = 0
        self._keys: set[tuple[str, int]] = set()
        self._dim: int | None = None
        self._closed = False

    def add(self, video_id: str, label: int, frame_idx: list[int], feats: np.ndarray) -> None:
        if self._closed:
            raise ValueError("feature cache writer is closed")
        feats = np.asarray(feats)
        if feats.ndim != 2 or feats.shape[1] < 1:
            raise ValueError("features must be a matrix with a non-empty feature dimension")
        if len(frame_idx) != len(feats):
            raise ValueError("one frame index per feature row")
        if label not in (0, 1) or not video_id or any(f < 0 for f in frame_idx):
            raise ValueError("feature rows need a video ID, binary label and non-negative frame")
        if not np.isfinite(feats).all() or np.any(np.abs(feats) > np.finfo(np.float16).max):
            raise ValueError("feature values must be finite and representable as float16")
        if self._dim is not None and feats.shape[1] != self._dim:
            raise ValueError("feature dimension changed within a cache")
        keys = [(video_id, f) for f in frame_idx]
        if len(set(keys)) != len(keys) or self._keys.intersection(keys):
            raise ValueError("duplicate video/frame feature rows")
        self._keys.update(keys)
        self._dim = feats.shape[1]
        start = len(self.index)
        self.index += [(video_id, label, f, start + i) for i, f in enumerate(frame_idx)]
        self.pending += list(feats.astype(np.float16))
        while len(self.pending) >= self.shard_rows:
            self._flush(self.pending[: self.shard_rows])
            self.pending = self.pending[self.shard_rows :]

    def _flush(self, rows: list[np.ndarray]) -> None:
        np.save(self._staging / f"shard_{self.shards:05d}.npy", np.stack(rows))
        self.shards += 1

    def close(self, meta: dict[str, Any]) -> None:
        if self._closed:
            raise ValueError("feature cache writer is closed")
        if self.pending:
            self._flush(self.pending)
        with open(self._staging / "index.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(INDEX_FIELDS)
            writer.writerows(self.index)
        meta = {**meta, "rows": len(self.index), "shard_rows": self.shard_rows}
        (self._staging / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        if self.out_dir.exists():
            self.out_dir.rmdir()  # only an empty destination is allowed
        self._staging.rename(self.out_dir)
        self._temp.cleanup()
        self._closed = True


class FeatureCache:
    def __init__(self, directory: Path) -> None:
        self.meta = json.loads((directory / "meta.json").read_text())
        self.shard_rows = self.meta["shard_rows"]
        with open(directory / "index.csv", newline="") as f:
            self.index = [
                (r["video_id"], int(r["label"]), int(r["frame_idx"]), int(r["row"]))
                for r in csv.DictReader(f)
            ]
        paths = sorted(directory.glob("shard_*.npy"))
        if [p.name for p in paths] != [f"shard_{i:05d}.npy" for i in range(len(paths))]:
            raise ValueError(f"{directory}: feature shard numbering must be contiguous")
        self.shards = [np.load(p, mmap_mode="r") for p in paths]
        if self.shard_rows < 1 or self.meta["rows"] != len(self.index):
            raise ValueError(f"{directory}: feature metadata does not match the index")
        if [r for _, _, _, r in self.index] != list(range(len(self.index))):
            raise ValueError(f"{directory}: feature index rows must be sequential")
        if len({(v, f) for v, _, f, _ in self.index}) != len(self.index):
            raise ValueError(f"{directory}: duplicate video/frame index rows")
        expected = (len(self.index) + self.shard_rows - 1) // self.shard_rows
        if len(self.shards) != expected:
            raise ValueError(f"{directory}: missing or extra feature shards")
        for i, shard in enumerate(self.shards):
            n = min(self.shard_rows, len(self.index) - i * self.shard_rows)
            if shard.ndim != 2 or len(shard) != n:
                raise ValueError(f"{directory}: invalid feature shard {i}")
        if len({s.shape[1] for s in self.shards}) > 1:
            raise ValueError(f"{directory}: inconsistent feature dimensions")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, row: int) -> np.ndarray:
        if not 0 <= row < len(self):
            raise IndexError(row)
        shard, offset = divmod(row, self.shard_rows)
        values = np.asarray(self.shards[shard][offset], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite cached features at row {row}")
        return values
