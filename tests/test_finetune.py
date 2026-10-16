import pytest
import torch
from torch import nn

from optlab.train.finetune import TailClassifier


@pytest.mark.parametrize("thaw", [False, True])
def test_tail_updates_only_enabled_backbone_parameters(thaw):
    torch.manual_seed(0)
    model = TailClassifier(
        nn.ModuleList([nn.Linear(4, 4)]), nn.LayerNorm(4), torch.randn(4, 3), thaw, hidden=4
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    loss = nn.functional.binary_cross_entropy_with_logits(
        model(torch.randn(8, 2, 4)), torch.tensor([0.0, 1.0] * 4)
    )
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(v, before[k])
        for k, v in model.state_dict().items()
        if k.startswith("head.")
    )
    changed = any(
        not torch.equal(v, before[k])
        for k, v in model.state_dict().items()
        if k.startswith("blocks.")
    )
    assert changed == thaw


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    from optlab.train.checkpoint import save_checkpoint

    path = tmp_path / "model.pt"
    path.write_bytes(b"previous")

    def fail(value, staging):
        staging.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint({}, path)
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]
