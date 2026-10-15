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


@pytest.mark.parametrize("field,value", [("train", -0.1), ("val", float("nan"))])
def test_split_config_rejects_invalid_fractions(field, value):
    from optlab.config import SplitConfig

    with pytest.raises(ValidationError):
        SplitConfig(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", []),
        ("seeds", [0, 0]),
        ("optimizers", []),
        ("cv_folds", 2),
        ("training", {"clip_norm": -1}),
        ("eval", {"in_distribution": "a", "cross_dataset": ["a"]}),
    ],
)
def test_experiment_config_rejects_ambiguous_or_unusable_settings(root, field, value):
    from optlab.config import ExperimentConfig, load_experiment

    cfg = load_experiment(root / "configs/experiments/calibrated_timit_cv.yaml").model_dump()
    cfg[field] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)
