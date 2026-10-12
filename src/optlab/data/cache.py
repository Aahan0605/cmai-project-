"""Feature cache (TRD 3.2 step 5): float16 .npy shards + index.csv + meta.json per
`{dataset}/{backbone}` directory. Read back with memmap and cast to float32 per access."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

INDEX_FIELDS = ["video_id", "label", "frame_idx", "row"]


class FeatureCacheWriter:
    """Row r lives in shard r // shard_rows at offset r % shard_rows."""

    def __init__(self, out_dir: Path, shard_rows: int = 65536) -> None:
        if (out_dir / "index.csv").exists():
            raise FileExistsError(f"{out_dir} already holds a feature cache")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir, self.shard_rows = out_dir, shard_rows
        self.index: list[tuple[str, int, int, int]] = []
        self.pending: list[np.ndarray] = []
        self.shards = 0

    def add(self, video_id: str, label: int, frame_idx: list[int], feats: np.ndarray) -> None:
        if len(frame_idx) != len(feats):
            raise ValueError("one frame index per feature row")
        start = len(self.index)
        self.index += [(video_id, label, f, start + i) for i, f in enumerate(frame_idx)]
        self.pending += list(feats.astype(np.float16))
        while len(self.pending) >= self.shard_rows:
            self._flush(self.pending[: self.shard_rows])
            self.pending = self.pending[self.shard_rows :]

    def _flush(self, rows: list[np.ndarray]) -> None:
        np.save(self.out_dir / f"shard_{self.shards:05d}.npy", np.stack(rows))
        self.shards += 1

    def close(self, meta: dict[str, Any]) -> None:
        if self.pending:
            self._flush(self.pending)
        with open(self.out_dir / "index.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(INDEX_FIELDS)
            writer.writerows(self.index)
        meta = {**meta, "rows": len(self.index), "shard_rows": self.shard_rows}
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


class FeatureCache:
    def __init__(self, directory: Path) -> None:
        self.meta = json.loads((directory / "meta.json").read_text())
        self.shard_rows = self.meta["shard_rows"]
        with open(directory / "index.csv", newline="") as f:
            self.index = [
                (r["video_id"], int(r["label"]), int(r["frame_idx"]), int(r["row"]))
                for r in csv.DictReader(f)
            ]
        self.shards = [np.load(p, mmap_mode="r") for p in sorted(directory.glob("shard_*.npy"))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, row: int) -> np.ndarray:
        shard, offset = divmod(row, self.shard_rows)
        return np.asarray(self.shards[shard][offset], dtype=np.float32)
