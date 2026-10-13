"""OptLab-TIMIT: an openly licensed real/fake face dataset built from VidTIMIT + DeepfakeTIMIT.

Sources (both on Zenodo, open access, CC BY-NC-ND 4.0, research use):
  VidTIMIT       record 158963: 43 people x 10 spoken sentences, JPEG frames 512x384 @ 25 fps
                 plus a 32 kHz WAV per sentence (the real videos).
  DeepfakeTIMIT  record 4068245: face swaps between 16 pairs of those people (32 subjects x 10
                 sentences), in a lower-quality (64 px model) and a higher-quality (128 px)
                 version, H.264 512x384 @ 25 fps; the audio is the untouched original.

Standardisation, so a detector cannot separate the classes by file format: every clip
(real and fake) ends as H.264 MP4 at 512x384, 25 fps, written by the same encoder. Real
frames get one extra H.264 pass first, matching the H.264 encode the fakes already went
through, so both classes share the history JPEG -> H.264 -> H.264. Residual difference: the
fakes' first H.264 encode used the DeepfakeTIMIT authors' (unknown) settings.

Layout written under `root`:
  real/<subject>/<sentence>.mp4 + .wav
  fake_hq/<subject>/<sentence>-<face donor>.mp4 + .wav   (likewise fake_lq/)
Only manifests are committed; the videos stay local (the licence forbids redistribution
of derivatives).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from optlab.data.manifest import VideoRow

ZENODO_API = "https://zenodo.org/api/records/{}"
VIDTIMIT, DEEPFAKETIMIT = 158963, 4068245
FPS = 25
QUALITIES = {"hq": "higher_quality", "lq": "lower_quality"}

Log = Callable[[str], None]


def zenodo_files(record: int) -> dict[str, dict]:
    """File name -> {'url', 'md5', 'size'} from the Zenodo API."""
    with urllib.request.urlopen(ZENODO_API.format(record), timeout=60) as r:
        files = json.load(r)["files"]
    return {
        f["key"]: {
            "url": f["links"]["self"],
            "md5": f["checksum"].removeprefix("md5:"),
            "size": f["size"],
        }
        for f in files
    }


def download(url: str, dest: Path, md5: str, size: int | None = None, retries: int = 20) -> None:
    """Resumable download (HTTP Range) with MD5 verification; a bad file is deleted.

    A dropped connection can end a transfer early without raising, so the loop keeps
    resuming until the file reaches the expected `size`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == md5:
        return  # already complete (a Range request past the end would fail)
    stalled = 0  # consecutive attempts that added no bytes; progress resets it
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if size is not None and have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with (
                urllib.request.urlopen(req, timeout=120) as r,
                open(dest, "ab" if have else "wb") as f,
            ):
                if have and r.status != 206:  # server ignored Range: start over
                    f.truncate(0)
                shutil.copyfileobj(r, f, length=1 << 20)
            if size is None:
                break
        except OSError:
            pass
        if (dest.stat().st_size if dest.exists() else 0) > have:
            stalled = 0
        else:
            stalled += 1
            if stalled >= retries:
                raise OSError(f"could not download {url}: no progress in {retries} attempts")
            time.sleep(min(5 * stalled, 60))
    digest = hashlib.md5(dest.read_bytes()).hexdigest()
    if digest != md5:
        dest.unlink()
        raise OSError(f"MD5 mismatch for {dest.name}: got {digest}, expected {md5}")


def _safe_member(name: str) -> bool:
    parts = Path(name).parts
    return not name.startswith("/") and ".." not in parts


def write_h264(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    """BGR uint8 frames -> H.264 MP4 (OpenCV's bundled FFmpeg encoder)."""
    import cv2

    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write H.264 here")
    for f in frames:
        writer.write(f)
    writer.release()
    tmp.replace(path)  # atomic: a half-written clip never looks finished


def read_all(path: Path) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise OSError(f"no frames decoded from {path}")
    return frames


def standardize_real(zip_path: Path, root: Path) -> int:
    """One VidTIMIT subject zip -> real/<subject>/<sentence>.mp4 + .wav. Head-rotation
    sequences (head, head2, head3) are skipped: they have no fake counterpart."""
    import cv2

    subject = zip_path.stem
    count = 0
    with tempfile.TemporaryDirectory(dir=root) as tmp, zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if _safe_member(n)]
        sentences = sorted(
            {
                Path(n).parts[2]
                for n in names
                if len(Path(n).parts) >= 4
                and Path(n).parts[1] == "video"
                and not Path(n).parts[2].startswith("head")
            }
        )
        for sent in sentences:
            frame_names = sorted(
                (
                    n
                    for n in names
                    if n.startswith(f"{subject}/video/{sent}/") and not n.endswith("/")
                ),
                key=lambda n: int(Path(n).name),
            )
            frames = [
                cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
                for n in frame_names
            ]
            first_pass = Path(tmp) / f"{sent}.mp4"
            write_h264(frames, first_pass)  # matches the fakes' existing H.264 generation
            write_h264(read_all(first_pass), root / "real" / subject / f"{sent}.mp4")
            with (
                z.open(f"{subject}/audio/{sent}.wav") as src,
                open(root / "real" / subject / f"{sent}.wav", "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def standardize_fakes(extracted: Path, root: Path) -> int:
    """DeepfakeTIMIT/<quality>/<subject>/<sentence>-video-<donor>.avi -> fake_<q>/... .mp4"""
    count = 0
    for q, folder in QUALITIES.items():
        for avi in sorted((extracted / "DeepfakeTIMIT" / folder).glob("*/*-video-*.avi")):
            sent, donor = avi.stem.split("-video-")
            out = root / f"fake_{q}" / avi.parent.name / f"{sent}-{donor}.mp4"
            if not out.exists():
                write_h264(read_all(avi), out)
                shutil.copyfile(avi.with_name(f"{sent}.wav"), out.with_suffix(".wav"))
            count += 1
    return count


def fetch(raw: Path, root: Path, log: Log = print) -> None:
    """Download and standardise everything; resumable. Raw archives are deleted once
    their clips are written, so peak disk use stays near one archive."""
    root.mkdir(parents=True, exist_ok=True)
    fakes = zenodo_files(DEEPFAKETIMIT)["DeepfakeTIMIT.tar.gz"]
    if len(list(root.glob("fake_*/*/*.mp4"))) < 640 and (raw / "dftimit").is_dir():
        log(
            f"standardised {standardize_fakes(raw / 'dftimit', root)} fake clips"
        )  # already unpacked
    if len(list(root.glob("fake_*/*/*.mp4"))) < 640:
        tar_path = raw / "DeepfakeTIMIT.tar.gz"
        if not tar_path.exists():
            log("downloading DeepfakeTIMIT.tar.gz")
            download(fakes["url"], tar_path, fakes["md5"], fakes["size"])
        with tempfile.TemporaryDirectory(dir=raw) as tmp, tarfile.open(tar_path) as tar:
            tar.extractall(tmp, filter="data")  # rejects absolute paths, '..', links
            log(f"standardised {standardize_fakes(Path(tmp), root)} fake clips")
        tar_path.unlink()
    for name, info in sorted(zenodo_files(VIDTIMIT).items()):
        if not name.endswith(".zip"):
            continue
        subject = name.removesuffix(".zip")
        if len(list((root / "real" / subject).glob("*.mp4"))) == 10:
            continue
        zip_path = raw / name
        log(f"downloading {name} ({info['size'] / 1e6:.0f} MB)")
        download(info["url"], zip_path, info["md5"], info["size"])
        log(f"{subject}: {standardize_real(zip_path, root)} real clips")
        zip_path.unlink()


def parse(root: Path, quality: str) -> list[VideoRow]:
    """Every real clip plus the fakes of one quality. A fake shows two people (the person
    in the video and the face donor), so both identities link it for the split."""
    rows = [
        VideoRow(
            f"real/{p.parent.name}/{p.stem}",
            p.relative_to(root).as_posix(),
            0,
            "real",
            p.parent.name,
        )
        for p in sorted((root / "real").glob("*/*.mp4"))
    ]
    for p in sorted((root / f"fake_{quality}").glob("*/*.mp4")):
        sent, donor = p.stem.rsplit("-", 1)
        rows.append(
            VideoRow(
                f"fake_{quality}/{p.parent.name}/{sent}",
                p.relative_to(root).as_posix(),
                1,
                f"deepfaketimit_{quality}",
                f"{p.parent.name}+{donor}",
            )
        )
    if not any(r.label == 0 for r in rows) or not any(r.label == 1 for r in rows):
        raise FileNotFoundError(f"OptLab-TIMIT not found under {root}; run `optlab fetch-timit`")
    return rows
