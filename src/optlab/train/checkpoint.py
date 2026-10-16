"""Replace checkpoints only after serialization succeeds."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def save_checkpoint(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        torch.save(value, staging)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
