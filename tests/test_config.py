from pathlib import Path

import pytest
from pydantic import ValidationError

from optlab.config import (
    SpaceDim,
    load_datasets,
    load_extraction,
    load_models,
    load_optimizers,
    load_toy_experiment,
)


def test_committed_configs_load(root: Path) -> None:
    configs = root / "configs"
    optimizers = load_optimizers(configs)
    toy = load_toy_experiment(configs / "experiments" / "toy.yaml")
    assert set(toy.optimizers + toy.reference_optimizers) <= set(optimizers)
    assert load_models(configs) and load_datasets(configs)
    assert load_extraction(configs).frames_per_video == 16
    # TRD section 7: lr and weight_decay are tuned for every first-order optimizer.
    for name in toy.optimizers:
        assert {"lr", "weight_decay"} <= set(optimizers[name].search_space)


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "log_uniform", "low": 0.0, "high": 1.0},
        {"type": "uniform", "low": 1.0, "high": 1.0},
        {"type": "choice", "values": []},
    ],
)
def test_space_dim_rejects_bad(bad: dict) -> None:
    with pytest.raises(ValidationError):
        SpaceDim.model_validate(bad)
