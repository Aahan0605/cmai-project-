"""Identity-disjoint split manifests (TRD 3.2 step 1, FR-D4).

Manifest CSV columns: video_id, path, label, method, identity_id, split, val_role.
`label` is 1 for fake and 0 for real. `identity_id` lists every identity appearing in the
video joined by '+' (a face-swap has a target and a source). Videos sharing any identity form
one connected component, and whole components are assigned to a split, so no identity
appears in two splits.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from optlab.config import SplitConfig

FIELDS = ["video_id", "path", "label", "method", "identity_id", "split", "val_role"]


@dataclass
class VideoRow:
    video_id: str
    path: str  # relative to the dataset root
    label: int  # 1 = fake, 0 = real
    method: str  # manipulation method, or 'real'
    identity_id: str  # identities joined by '+'
    split: str = ""
    val_role: str = ""  # 'tune' | 'fusion' for val rows, '' otherwise


# --- dataset layouts --------------------------------------------------------------------------
# Folder layouts follow the public dataset READMEs; they are untested against the real
# downloads until access is granted, so the parsers fail loudly when nothing matches.


def parse_faceforensics(root: Path, compression: str, methods: list[str]) -> list[VideoRow]:
    """FF++: original_sequences/youtube/<c>/videos/NNN.mp4 and
    manipulated_sequences/<Method>/<c>/videos/TTT_SSS.mp4 (target_source)."""
    rows = [
        VideoRow(f"real/{p.stem}", p.relative_to(root).as_posix(), 0, "real", p.stem)
        for p in sorted((root / "original_sequences/youtube" / compression / "videos").glob("*.mp4"))
    ]
    for method in methods:
        for p in sorted((root / "manipulated_sequences" / method / compression / "videos").glob("*.mp4")):
            target, source = p.stem.split("_")
            rows.append(
                VideoRow(f"{method}/{p.stem}", p.relative_to(root).as_posix(), 1, method,
                         f"{target}+{source}")
            )
    if not rows:
        raise FileNotFoundError(f"no FaceForensics++ videos found under {root}")
    return rows


def parse_celeb_df(root: Path) -> list[VideoRow]:
    """Celeb-DF v2 official test list (List_of_testing_videos.txt, '<flag> <folder>/<file>').

    Labels come from the folder (Celeb-real, YouTube-real = real; Celeb-synthesis = fake), not
    from the list's flag column. All rows are split 'test': this dataset is cross-dataset only.
    """
    rows = []
    for line in (root / "List_of_testing_videos.txt").read_text().split("\n"):
        if not line.strip():
            continue
        rel = line.split()[-1]
        folder, stem = rel.split("/")[0], Path(rel).stem
        if folder == "Celeb-synthesis":
            target, source, _ = stem.split("_")
            rows.append(VideoRow(f"fake/{stem}", rel, 1, "celeb_synthesis", f"{target}+{source}", "test"))
        elif folder == "Celeb-real":
            rows.append(VideoRow(f"real/{stem}", rel, 0, "real", stem.split("_")[0], "test"))
        elif folder == "YouTube-real":
            rows.append(VideoRow(f"youtube/{stem}", rel, 0, "real", f"yt{stem}", "test"))
        else:
            raise ValueError(f"unexpected Celeb-DF folder in test list: {rel}")
    return rows


# --- splitting --------------------------------------------------------------------------------


def identity_components(rows: list[VideoRow]) -> list[list[int]]:
    """Union-find over identities; returns row indices grouped by connected component."""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in rows:
        ids = row.identity_id.split("+")
        for other in ids[1:]:
            parent[find(other)] = find(ids[0])
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(row.identity_id.split("+")[0]), []).append(i)
    return sorted(groups.values())


def _assign(components: list[list[int]], targets: dict[str, float], rng: random.Random) -> dict[int, str]:
    """Shuffle components, then give each to the split furthest below its target share."""
    order = components[:]
    rng.shuffle(order)
    total = sum(len(c) for c in order)
    filled = dict.fromkeys(targets, 0)
    out = {}
    for comp in order:
        name = max(targets, key=lambda s: targets[s] * total - filled[s])
        filled[name] += len(comp)
        out.update(dict.fromkeys(comp, name))
    return out


def assign_splits(rows: list[VideoRow], cfg: SplitConfig) -> list[VideoRow]:
    comps = identity_components(rows)
    largest = max(len(c) for c in comps)
    if largest > 0.5 * len(rows):
        raise ValueError(
            f"one identity component holds {largest}/{len(rows)} videos; "
            "an identity-disjoint split is not possible"
        )
    rng = random.Random(cfg.seed)
    split_of = _assign(comps, {"train": cfg.train, "val": cfg.val, "test": cfg.test}, rng)
    val_comps = [c for c in comps if split_of[c[0]] == "val"]
    role_of = _assign(
        val_comps, {"tune": 1 - cfg.val_fusion_fraction, "fusion": cfg.val_fusion_fraction}, rng
    )
    for i, row in enumerate(rows):
        row.split = split_of[i]
        row.val_role = role_of.get(i, "")
    return rows


def write_manifest(rows: list[VideoRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def read_manifest(path: Path) -> list[VideoRow]:
    with open(path, newline="") as f:
        return [VideoRow(**{**r, "label": int(r["label"])}) for r in csv.DictReader(f)]
