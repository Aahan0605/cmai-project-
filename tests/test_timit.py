"""OptLab-TIMIT build steps on tiny synthetic inputs (no network)."""

import hashlib
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # part of the [extract] extra

from optlab.config import SplitConfig  # noqa: E402
from optlab.data import timit  # noqa: E402
from optlab.data.manifest import assign_splits  # noqa: E402
from optlab.data.preprocess import load_audio  # noqa: E402

SENTS = ["sa1", "sx2"]


def wav_bytes(seconds: float = 0.5, rate: int = 32000) -> bytes:
    import io

    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def frame(i: int) -> np.ndarray:
    img = np.zeros((96, 128, 3), np.uint8)
    cv2.circle(img, (40 + i, 48), 20, (200, 180, 160), -1)
    return img


def vidtimit_zip(path: Path, subject: str) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for sent in SENTS + ["head"]:  # 'head' must be skipped
            for i in range(1, 6):
                ok, jpg = cv2.imencode(".jpg", frame(i))
                z.writestr(f"{subject}/video/{sent}/{i:03d}", jpg.tobytes())
        for sent in SENTS:
            z.writestr(f"{subject}/audio/{sent}.wav", wav_bytes())


def deepfaketimit_tree(root: Path, pairs: list[tuple[str, str]]) -> None:
    for q in timit.QUALITIES.values():
        for a, b in pairs:
            for target, donor in [(a, b), (b, a)]:
                d = root / "DeepfakeTIMIT" / q / target
                for sent in SENTS:
                    timit.write_h264([frame(i) for i in range(5)], d / f"{sent}-video-{donor}.avi")
                    (d / f"{sent}.wav").write_bytes(wav_bytes())


def test_build_and_parse(tmp_path: Path) -> None:
    root = tmp_path / "timit"
    root.mkdir()
    subjects = ["faaa0", "mbbb0", "mccc0", "fddd0", "meee0", "ffff0", "mggg0"]
    for s in subjects:
        vidtimit_zip(tmp_path / f"{s}.zip", s)
        assert timit.standardize_real(tmp_path / f"{s}.zip", root) == len(SENTS)
    deepfaketimit_tree(tmp_path / "raw", [("faaa0", "mbbb0")])
    assert timit.standardize_fakes(tmp_path / "raw", root) == 2 * 2 * len(SENTS)

    clip = root / "real" / "faaa0" / "sa1.mp4"
    cap = cv2.VideoCapture(str(clip))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5 and cap.get(cv2.CAP_PROP_FPS) == 25
    assert int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little") in (b"avc1", b"h264")
    assert not (root / "real" / "faaa0" / "head.mp4").exists()

    hq = timit.parse(root, "hq")
    assert sum(r.label == 0 for r in hq) == len(subjects) * len(SENTS)
    fakes = [r for r in hq if r.label]
    assert {r.identity_id for r in fakes} == {"faaa0+mbbb0", "mbbb0+faaa0"}
    # Real and fake clips of a pair share identities, so the pair lands in one split.
    split = assign_splits(hq, SplitConfig(seed=0, train=0.5, val=0.25, test=0.25))
    pair = {r.split for r in split if r.identity_id.split("+")[0] in ("faaa0", "mbbb0")}
    assert len(pair) == 1

    audio = load_audio(clip, 16000)
    assert audio.dtype == np.float32 and len(audio) == 8000  # 0.5 s resampled 32k -> 16k
    assert np.abs(audio).max() == pytest.approx(0.3, abs=0.01)


def test_download_skips_a_complete_file(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    timit.download("http://invalid.invalid/x", f, hashlib.md5(b"hello").hexdigest())
    assert f.read_bytes() == b"hello"


def test_parse_requires_both_classes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        timit.parse(tmp_path, "hq")
