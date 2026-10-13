import numpy as np
import pytest

from optlab.config import SpaceDim
from optlab.tune.search import METHODS, from_unit, search

SPACE = {
    "lr": SpaceDim(type="log_uniform", low=1e-4, high=1.0),
    "weight_decay": SpaceDim(type="log_uniform", low=1e-6, high=1e-2, p_zero=0.5),
}


def objective(_: int, hp: dict) -> float:
    # Peak at lr = 0.01; weight decay mildly penalised.
    return -abs(np.log10(hp["lr"]) + 2) - hp["weight_decay"]


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("budget", [1, 7, 20])
def test_every_method_uses_exactly_the_budget(method: str, budget: int) -> None:
    calls = []
    trials = search(method, SPACE, lambda i, hp: calls.append(i) or objective(i, hp), budget, 0)
    assert len(trials) == budget and calls == list(range(budget))
    for t in trials:
        assert 1e-4 <= t.hparams["lr"] <= 1.0
        assert t.hparams["weight_decay"] == 0.0 or 1e-6 <= t.hparams["weight_decay"] <= 1e-2


@pytest.mark.parametrize("method", list(METHODS))
def test_search_is_deterministic(method: str) -> None:
    a = [t.hparams for t in search(method, SPACE, objective, 10, seed=3)]
    b = [t.hparams for t in search(method, SPACE, objective, 10, seed=3)]
    assert a == b


def test_from_unit_mapping() -> None:
    assert from_unit(SPACE, np.array([0.0, 0.0])) == {
        "lr": pytest.approx(1e-4),
        "weight_decay": 0.0,
    }
    assert from_unit(SPACE, np.array([1.0, 1.0])) == {
        "lr": pytest.approx(1.0),
        "weight_decay": pytest.approx(1e-2),
    }
    assert from_unit(SPACE, np.array([0.5, 0.5]))["lr"] == pytest.approx(1e-2)
    choice = {"b": SpaceDim(type="choice", values=[0.8, 0.9, 0.99])}
    assert [from_unit(choice, np.array([u]))["b"] for u in (0.0, 0.5, 1.0)] == [0.8, 0.9, 0.99]


def test_pso_and_ga_find_the_good_region() -> None:
    for method in ("pso", "ga"):
        best = max(search(method, SPACE, objective, 30, seed=0), key=lambda t: t.score)
        assert 1e-3 < best.hparams["lr"] < 1e-1
