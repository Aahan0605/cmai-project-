import pytest
import torch

from optlab.models.heads import JointHead
from optlab.train.trainer import loss_fn


@pytest.mark.parametrize("kind", ["concat_mlp", "svm_head"])
def test_joint_heads_standardize_from_training_and_backpropagate(kind):
    torch.manual_seed(0)
    x = {"visual": torch.randn(10, 4), "frequency": torch.randn(10, 2)}
    model = JointHead({"visual": 4, "frequency": 2}, hidden=8, kind=kind)
    model.fit_standardizer(x)
    center = model.center.clone()
    y = torch.arange(10) % 2
    z = model(x)
    assert z.shape == (10, 1)
    loss = loss_fn(z, y.float(), model)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    model({k: v + 1000 for k, v in x.items()})
    assert torch.equal(model.center, center)


def test_svm_objective_matches_soft_margin_primal():
    model = JointHead({"visual": 2}, kind="svm_head", svm_c=2)
    model.heads["joint"].weight.data.fill_(1)
    logits = torch.tensor([[2.0], [-2.0], [0.0]])
    labels = torch.tensor([1.0, 0.0, 1.0])
    assert model.objective(logits, labels).item() == pytest.approx(1 + 2 / 3)
