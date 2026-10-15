from collections import Counter

import pytest

from optlab.config import ExperimentConfig
from optlab.cv import fold_assignment, write_folds
from optlab.data.manifest import VideoRow, read_manifest, write_manifest


def test_all_32_people_held_out_once_with_disjoint_validation_roles(tmp_path):
    rows = []
    for pair in range(16):
        a, b = f"p{pair * 2}", f"p{pair * 2 + 1}"
        rows.extend(
            [
                VideoRow(f"r{a}", "", 0, "real", a),
                VideoRow(f"r{b}", "", 0, "real", b),
                VideoRow(f"f{a}", "", 1, "swap", f"{a}+{b}"),
                VideoRow(f"f{b}", "", 1, "swap", f"{b}+{a}"),
            ]
        )
    write_manifest(rows, tmp_path / "hq.csv")
    cfg = ExperimentConfig(
        name="test",
        phase="baseline",
        hypothesis="test",
        train_dataset="hq",
        eval={"in_distribution": "hq"},
        model="test",
        optimizers=["adam"],
        seeds=[0],
        manifests_dir=str(tmp_path),
    )
    paths = write_folds(cfg, 8, 0, tmp_path / "folds", tmp_path / "configs")
    tested = Counter()
    for k in range(len(paths)):
        groups = {}
        for row in read_manifest(tmp_path / "folds" / f"f{k}" / "hq.csv"):
            role = row.val_role if row.split == "val" else row.split
            groups.setdefault(role, set()).update(row.identity_id.split("+"))
        assert set(groups) == {"train", "tune", "fusion", "test"}
        assert len(groups["test"]) == 4
        for role, ids in groups.items():
            assert all(not ids & other for name, other in groups.items() if name != role)
        tested.update(groups["test"])
    assert len(tested) == 32 and set(tested.values()) == {1}


def test_two_folds_cannot_leave_training_data():
    with pytest.raises(ValueError, match="at least three"):
        fold_assignment([], 2, 0)
