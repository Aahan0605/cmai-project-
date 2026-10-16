import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from optlab.config import ExtractionConfig, SplitConfig
from optlab.data.cache import FeatureCache, FeatureCacheWriter
from optlab.data.dataset import FeatureDataset, balanced_sampler, collate
from optlab.data.extract import extract_audio, extract_visual
from optlab.data.extractors import FrequencyExtractor
from optlab.data.faces import crop_face
from optlab.data.manifest import (
    VideoRow,
    assign_splits,
    parse_celeb_df,
    parse_faceforensics,
    read_manifest,
    write_manifest,
)
from optlab.data.preprocess import sample_frame_indices

METHODS = ["Deepfakes", "FaceSwap"]


def fake_ffpp(root: Path, n_pairs: int = 40) -> None:
    """FF++-shaped tree of empty files: identities paired (a, b) both ways, like the real set."""
    (root / "original_sequences/youtube/c23/videos").mkdir(parents=True)
    for m in METHODS:
        (root / f"manipulated_sequences/{m}/c23/videos").mkdir(parents=True)
    for p in range(n_pairs):
        a, b = f"{2 * p:03d}", f"{2 * p + 1:03d}"
        for x in (a, b):
            (root / f"original_sequences/youtube/c23/videos/{x}.mp4").touch()
        for m in METHODS:
            for t, s in ((a, b), (b, a)):
                (root / f"manipulated_sequences/{m}/c23/videos/{t}_{s}.mp4").touch()


def test_ffpp_splits_are_identity_disjoint_and_deterministic(tmp_path: Path) -> None:
    fake_ffpp(tmp_path)
    rows = parse_faceforensics(tmp_path, "c23", METHODS)
    assert len(rows) == 80 + 160 and sum(r.label for r in rows) == 160
    cfg = SplitConfig(seed=3)
    split = assign_splits(rows, cfg)
    ids_by_split: dict[str, set[str]] = {}
    for r in split:
        ids_by_split.setdefault(r.split, set()).update(r.identity_id.split("+"))
    names = list(ids_by_split)
    assert set(names) == {"train", "val", "test"}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not ids_by_split[a] & ids_by_split[b]
    share = {s: sum(r.split == s for r in split) / len(split) for s in names}
    assert share["train"] == pytest.approx(0.7, abs=0.05)
    assert {r.val_role for r in split if r.split == "val"} == {"tune", "fusion"}
    assert all(r.val_role == "" for r in split if r.split != "val")
    again = assign_splits(parse_faceforensics(tmp_path, "c23", METHODS), cfg)
    assert [r.split for r in again] == [r.split for r in split]

    write_manifest(split, tmp_path / "m.csv")
    assert read_manifest(tmp_path / "m.csv") == split


def test_split_refuses_one_giant_identity_component() -> None:
    chain = [VideoRow(str(i), "", 1, "x", f"id{i}+id{i + 1}") for i in range(10)]
    with pytest.raises(ValueError, match="identity-disjoint"):
        assign_splits(chain, SplitConfig())


def test_celeb_df_uses_folder_labels(tmp_path: Path) -> None:
    (tmp_path / "List_of_testing_videos.txt").write_text(
        "1 YouTube-real/00170.mp4\n1 Celeb-real/id1_0007.mp4\n0 Celeb-synthesis/id1_id6_0007.mp4\n"
    )
    rows = parse_celeb_df(tmp_path)
    assert [(r.label, r.identity_id, r.split) for r in rows] == [
        (0, "yt00170", "test"),
        (0, "id1", "test"),
        (1, "id1+id6", "test"),
    ]


def test_sample_frame_indices() -> None:
    assert sample_frame_indices(160, 16) == list(range(5, 160, 10))
    assert sample_frame_indices(3, 16) == [0, 1, 2]
    assert sample_frame_indices(0, 16) == []


def test_crop_face_clips_to_frame_and_resizes() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100] = 255
    crop = crop_face(frame, (80.0, 10.0, 120.0, 60.0), margin=0.3, size=32)
    assert crop.shape == (32, 32, 3) and crop.dtype == np.uint8
    # Box centred on the white/black edge: left half white, right half black.
    assert crop[:, :12].min() == 255 and crop[:, -12:].max() == 0
    near_edge = crop_face(frame, (0.0, 0.0, 30.0, 30.0), margin=0.5, size=16)
    assert near_edge.shape == (16, 16, 3)


def test_frequency_extractor() -> None:
    ext = FrequencyExtractor(size=32, bins=8)
    flat = np.full((40, 40, 3), 128, dtype=np.uint8)
    yy, xx = np.indices((40, 40))
    checker = (((xx + yy) % 2) * 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    out = ext([flat, checker])
    assert out.shape == (2, 8) and out.dtype == np.float32
    assert out[0, 0] > 0 and np.allclose(out[0, 1:], 0, atol=1e-5)  # constant: DC only
    assert out[1, -1] > out[0, -1]  # checkerboard: high-frequency energy


def test_cache_round_trip_across_shards(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    w = FeatureCacheWriter(tmp_path / "c", shard_rows=3)
    a, b = rng.normal(size=(4, 5)), rng.normal(size=(3, 5))
    w.add("v1", 1, [0, 2, 4, 6], a)
    w.add("v2", 0, [1, 3, 5], b)
    w.close({"backbone": "test"})
    cache = FeatureCache(tmp_path / "c")
    assert len(cache) == 7 and len(cache.shards) == 3
    expected = np.concatenate([a, b]).astype(np.float16).astype(np.float32)
    assert np.array_equal(np.stack([cache[i] for i in range(7)]), expected)
    with pytest.raises(FileExistsError):
        FeatureCacheWriter(tmp_path / "c")


class FakeDetector:
    """Face in the centre of every frame except frame index 0 of video 'real/000'."""

    def detect(self, frames):
        return [(10.0, 10.0, 30.0, 30.0) if f[0, 0, 0] != 7 else None for f in frames]


class FakeVisual:
    name = "fake_visual"

    def __call__(self, batch):
        return np.stack([[img.mean(), img[..., 0].std(), 1.0] for img in batch])


class FakeAudio:
    name = "fake_audio"

    def __call__(self, batch):
        return np.stack([[w.mean(), w.std()] for w in batch])


def test_extract_then_feature_dataset(tmp_path: Path) -> None:
    fake_ffpp(tmp_path / "ff", n_pairs=10)
    rows = assign_splits(parse_faceforensics(tmp_path / "ff", "c23", METHODS), SplitConfig())
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest)

    def read_frames(path: Path, k: int):
        rng = np.random.default_rng(zlib.crc32(path.name.encode()))
        frames = [rng.integers(8, 255, (40, 40, 3), dtype=np.uint8) for _ in range(k)]
        if path.name == "000.mp4":
            frames[0][0, 0, 0] = 7  # FakeDetector finds no face in this frame
        return list(range(0, 2 * k, 2)), frames

    cfg = ExtractionConfig(frames_per_video=4, face_size=24, freq_size=16, freq_bins=6)
    out = tmp_path / "features"
    stats = extract_visual(
        manifest,
        tmp_path / "ff",
        out,
        cfg,
        FakeDetector(),
        FakeVisual(),
        FrequencyExtractor(cfg.freq_size, cfg.freq_bins),
        read_frames,
        log=lambda _: None,
    )
    assert stats["videos"] == len(rows) and stats["frames_no_face"] == 1
    extract_audio(
        manifest,
        tmp_path / "ff",
        out,
        cfg,
        FakeAudio(),
        lambda p, sr: np.full(sr, len(p.name), dtype=np.float32),
    )

    visual = FeatureCache(out / "fake_visual")
    assert visual.meta["manifest_sha256"] and visual.meta["preprocessing"]["face_size"] == 24
    ds = FeatureDataset(
        manifest,
        "train",
        visual,
        frequency=FeatureCache(out / "freq_spectrum"),
        audio=FeatureCache(out / "fake_audio"),
    )
    train_videos = {r.video_id for r in rows if r.split == "train"}
    assert {vid for _, _, _, _, vid in ds} == train_videos
    xv, xa, xf, y, vid = ds[0]
    assert xv.shape == (3,) and xa.shape == (2,) and xf.shape == (6,) and xv.dtype == torch.float32
    tune = FeatureDataset(manifest, "val", visual, val_role="tune")
    assert {r.val_role for r in rows if r.video_id in {i[0] for i in tune.items}} == {"tune"}

    loader = DataLoader(
        ds,
        batch_size=8,
        collate_fn=collate,
        sampler=balanced_sampler(ds, torch.Generator().manual_seed(0)),
    )
    xv, xa, xf, y, vids = next(iter(loader))
    assert xv.shape == (8, 3) and xa.shape == (8, 2) and xf.shape == (8, 6) and len(vids) == 8
    weights = balanced_sampler(ds, torch.Generator()).weights
    labels = torch.tensor(ds.labels)
    assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())


@pytest.mark.parametrize("values", [np.array([[np.inf]]), np.array([[1e10]]), np.array([1.0])])
def test_cache_rejects_invalid_features_without_publishing(tmp_path, values):
    out = tmp_path / "invalid"
    writer = FeatureCacheWriter(out)
    with pytest.raises(ValueError):
        writer.add("v", 1, [0], values)
    assert not out.exists()


def test_cache_rejects_duplicates_and_missing_shards(tmp_path):
    out = tmp_path / "cache"
    writer = FeatureCacheWriter(out, shard_rows=1)
    writer.add("v", 1, [0], np.ones((1, 2)))
    with pytest.raises(ValueError, match="duplicate"):
        writer.add("v", 1, [0], np.ones((1, 2)))
    writer.close({})
    (out / "shard_00000.npy").unlink()
    with pytest.raises(ValueError, match="shards"):
        FeatureCache(out)


@pytest.mark.parametrize(
    "rows",
    [
        [VideoRow("v", "", 0, "real", "a", "train"), VideoRow("v", "", 1, "fake", "b", "test")],
        [VideoRow("v", "", 0, "real", "a", "train"), VideoRow("w", "", 1, "fake", "a+b", "test")],
        [
            VideoRow("v", "", 0, "real", "a", "val", "tune"),
            VideoRow("w", "", 1, "fake", "a+b", "val", "fusion"),
        ],
    ],
)
def test_manifest_rejects_duplicate_videos_and_identity_leakage(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    with pytest.raises(ValueError):
        read_manifest(path)


def test_cache_rejects_reordered_shard_numbering(tmp_path):
    out = tmp_path / "cache"
    writer = FeatureCacheWriter(out, shard_rows=1)
    writer.add("v", 1, [0, 1], np.array([[1.0, 2.0], [3.0, 4.0]]))
    writer.close({})
    (out / "shard_00000.npy").rename(out / "shard_00002.npy")
    with pytest.raises(ValueError, match="numbering"):
        FeatureCache(out)


def test_explicit_metadata_adapter_confines_files_and_requires_both_classes(tmp_path):
    from optlab.data.manifest import parse_metadata

    root = tmp_path / "media"
    root.mkdir()
    (root / "real.png").write_bytes(b"fixture")
    (root / "fake.png").write_bytes(b"fixture")
    rows = [VideoRow("r", "real.png", 0, "real", "a"), VideoRow("f", "fake.png", 1, "swap", "b+c")]
    metadata = tmp_path / "metadata.csv"
    write_manifest(rows, metadata)
    assert parse_metadata(root, metadata) == rows
    rows[1].path = "../metadata.csv"
    write_manifest(rows, metadata)
    with pytest.raises(ValueError, match="under the dataset root"):
        parse_metadata(root, metadata)
    write_manifest(rows[:1], metadata)
    with pytest.raises(ValueError, match="both real"):
        parse_metadata(root, metadata)


def test_still_image_reader_applies_rgb_and_single_frame(tmp_path):
    from PIL import Image

    from optlab.data.preprocess import read_image_frame

    path = tmp_path / "image.png"
    Image.new("L", (10, 12), 100).save(path)
    indices, frames = read_image_frame(path)
    assert indices == [0] and frames[0].shape == (12, 10, 3)
