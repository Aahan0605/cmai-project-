"""Pydantic config models and YAML loaders. YAML under `configs/` feeds the DB catalogues."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T", bound=BaseModel)

Family = Literal[
    "plain", "momentum", "adaptive", "sign", "sharpness_aware", "second_order", "hybrid"
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpaceDim(_Strict):
    """One tuned hyperparameter. `p_zero` puts that much probability mass exactly on 0."""

    type: Literal["log_uniform", "uniform", "choice"]
    low: float | None = None
    high: float | None = None
    values: list[Any] | None = None
    p_zero: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> SpaceDim:
        if self.type == "choice":
            if not self.values:
                raise ValueError("choice dimension needs non-empty `values`")
        elif self.low is None or self.high is None or not self.low < self.high:
            raise ValueError("numeric dimension needs low < high")
        elif self.type == "log_uniform" and self.low <= 0:
            raise ValueError("log_uniform needs low > 0")
        return self


class OptimizerConfig(_Strict):
    name: str
    family: Family
    in_syllabus: bool = False
    update_rule: str | None = None
    reference: str | None = None
    fixed: dict[str, Any] = Field(default_factory=dict)  # not tuned (e.g. momentum, betas)
    search_space: dict[str, SpaceDim] = Field(default_factory=dict)


class ModelConfig(_Strict):
    name: str
    visual_backbone: str
    audio_backbone: str | None = None
    uses_frequency_branch: bool = False
    fusion_type: Literal["concat_mlp", "weighted_sum", "svm_head"]
    backbone_frozen: bool = True
    unfrozen_last_n_layers: int = Field(0, ge=0)
    head_hidden: int = 256
    dropout: float = 0.2


class DatasetConfig(_Strict):
    name: Literal[
        "faceforensics_pp", "celeb_df_v2", "fakeavceleb", "optlab_timit_hq", "optlab_timit_lq"
    ]
    has_audio: bool = False
    compression: str | None = None
    access_notes: str | None = None
    methods: list[str] | None = None  # FF++ manipulation folders to include


class SplitConfig(_Strict):
    """Identity-disjoint split fractions (by video count). Splits are never reseeded per run."""

    seed: int = 0
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    val_fusion_fraction: float = Field(0.5, ge=0.0, le=1.0)  # share of val used to fit fusion

    @model_validator(mode="after")
    def _check(self) -> SplitConfig:
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("train + val + test must sum to 1")
        return self


class ExtractionConfig(_Strict):
    frames_per_video: int = Field(16, ge=1)
    face_margin: float = Field(0.3, ge=0.0)  # box grows by this fraction on each side
    face_size: int = 224
    face_detector: Literal["mtcnn"] = "mtcnn"
    visual_backbones: dict[str, dict[str, str]] = Field(default_factory=dict)
    audio_backbones: dict[str, dict[str, str]] = Field(default_factory=dict)
    audio_sample_rate: int = 16000
    freq_size: int = 128  # grayscale crop is resized to this before the FFT
    freq_bins: int = 64
    shard_rows: int = Field(65536, ge=1)


class ToyBenchmarkConfig(_Strict):
    name: str
    kind: Literal["quadratic", "rosenbrock", "rastrigin"]
    dim: int = Field(2, ge=1)
    condition_number: float | None = None
    rotation_seed: int = 0  # fixes the quadratic's eigenbasis; the same for every run seed
    max_steps: int = Field(gt=0)
    tolerance: float = Field(gt=0)
    start_low: float
    start_high: float
    references: bool = False  # also run the second-order references (Newton, L-BFGS)
    description: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ToyBenchmarkConfig:
        if self.kind == "quadratic" and (self.condition_number or 0) < 1:
            raise ValueError("quadratic benchmark needs condition_number >= 1")
        if self.kind == "rosenbrock" and self.dim < 2:
            raise ValueError("rosenbrock needs dim >= 2")
        return self

    @property
    def is_convex(self) -> bool:
        return self.kind == "quadratic"


class ToyExperimentConfig(_Strict):
    name: str
    phase: Literal["toy"] = "toy"
    description: str | None = None
    hypothesis: str
    seeds: list[int]
    tuning_seed: int = 0  # learning rate is picked from `lr_grid` on this seed only
    lr_grid: list[float]
    optimizers: list[str]
    reference_optimizers: list[str] = Field(default_factory=list)
    hparam_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    divergence_threshold: float = 1e12
    curve_dense_steps: int = 100  # every step is logged up to here, then every `curve_stride`
    curve_stride: int = 10
    trajectory_dir: str = "artifacts/toy"
    benchmarks: list[ToyBenchmarkConfig]


class EvalConfig(_Strict):
    in_distribution: str
    cross_dataset: list[str] = Field(default_factory=list)


class TuningConfig(_Strict):
    method: Literal["random", "grid", "pso", "ga"] = "random"
    trials: int = Field(20, ge=1)  # the same budget for every optimizer (TRD 7)
    epochs: int = Field(10, ge=1)  # shortened training per trial
    seed: int = 0


class TrainingConfig(_Strict):
    max_epochs: int = Field(30, ge=1)
    batch_size: int = Field(256, ge=1)
    patience: int = Field(5, ge=1)  # early stopping on val_tune video AUC
    clip_norm: float | None = None
    schedule: Literal["constant"] = "constant"


class ExperimentConfig(_Strict):
    name: str
    phase: Literal["baseline", "hybrid", "ablation"]
    description: str | None = None
    hypothesis: str
    train_dataset: str
    eval: EvalConfig
    model: str
    optimizers: list[str]
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    seeds: list[int]
    # 'simplex': w on the simplex (TRD 9). 'simplex_temperature': w on the simplex times a
    # learned temperature, so the weights no longer absorb logit scale (Gate 1 finding).
    fusion: Literal["simplex", "simplex_temperature"] = "simplex"
    fusion_ridge: float = Field(1e-3, ge=0.0)
    cv_folds: int | None = Field(None, ge=2)  # set for `optlab cv` (grouped by identity)
    cv_seed: int = 0
    manifests_dir: str = "manifests"
    features_dir: str = "features"
    artifacts_dir: str = "artifacts"
    device: str = "cpu"


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dir(directory: str | Path, model: type[T]) -> dict[str, T]:
    """Load every *.yaml in a directory into `model`, keyed by `name`."""
    items = [model.model_validate(_load(p)) for p in sorted(Path(directory).glob("*.yaml"))]
    return {item.name: item for item in items}  # type: ignore[attr-defined]


def load_optimizers(configs_dir: str | Path) -> dict[str, OptimizerConfig]:
    return load_dir(Path(configs_dir) / "optim", OptimizerConfig)


def load_models(configs_dir: str | Path) -> dict[str, ModelConfig]:
    return load_dir(Path(configs_dir) / "model", ModelConfig)


def load_datasets(configs_dir: str | Path) -> dict[str, DatasetConfig]:
    return load_dir(Path(configs_dir) / "data" / "datasets", DatasetConfig)


def load_splits(configs_dir: str | Path) -> SplitConfig:
    return SplitConfig.model_validate(_load(Path(configs_dir) / "data" / "splits.yaml"))


def load_extraction(configs_dir: str | Path) -> ExtractionConfig:
    return ExtractionConfig.model_validate(_load(Path(configs_dir) / "data" / "extraction.yaml"))


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load(Path(path)))


def load_toy_experiment(path: str | Path) -> ToyExperimentConfig:
    return ToyExperimentConfig.model_validate(_load(Path(path)))
