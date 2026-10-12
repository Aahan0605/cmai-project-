# Technical Requirements Document

**Project:** OptLab-DF (`optlab`)
**Companion docs:** `docs/PRD.md` (what and why), `docs/ARCHITECTURE.md` (diagrams), `docs/UIUX.md` (dashboard), `backend/schema.sql` (data model, source of truth)
**Status:** Draft v1 · 2026-10-12

---

## 1. Technology choices

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.11 | Type hints throughout |
| ML | PyTorch 2.x | CPU-capable for tests and toy suite; GPU for feature extraction |
| Visual backbone (frozen) | CLIP ViT-B/16 image encoder via `open_clip` (configurable) | Inspired by CLIP-feature deepfake detectors that generalize across generators; verify the reference before citing |
| Audio backbone (frozen) | wav2vec 2.0 base via `transformers` or `torchaudio` | Mean-pooled clip embedding, 16 kHz mono |
| Frequency stream | Azimuthally averaged log power spectrum of the grayscale face crop | No backbone; tiny hand-built feature vector; used when audio is unavailable |
| Face detection | MTCNN (`facenet-pytorch`) **or** InsightFace/RetinaFace | Pluggable; pick whichever installs cleanly on the target runtime |
| Optimizers | Own implementations + `torch.optim`, `lion-pytorch`, `adabelief-pytorch` (or `torch-optimizer`) | See §6 |
| Convex solving | `cvxpy` (OSQP/ECOS) | Fusion weights, SVM dual |
| Stats / metrics | `numpy`, `scipy`, `scikit-learn` | AUC via sklearn; EER from the ROC curve |
| Results store | SQLite via stdlib `sqlite3` + thin repository layer | `backend/schema.sql` is the source of truth |
| API | FastAPI + Pydantic v2, uvicorn | Read-only |
| Frontend | React 18, Vite, TypeScript, Tailwind, TanStack Query, React Router, Plotly (`react-plotly.js`) | See `docs/UIUX.md` |
| Tooling | `uv` or `pip`, `ruff`, `mypy` (soft), `pytest`, GitHub Actions, `Makefile` | |

## 2. Repository layout

```
cmai-project-/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── configs/
│   ├── data/                 # dataset + extraction configs
│   ├── model/                # model_configs rows as YAML
│   ├── optim/                # per-optimizer search spaces
│   └── experiments/          # toy.yaml, baseline_v1.yaml, hybrid_v1.yaml, ablation_*.yaml
├── manifests/                # committed split CSVs (video_id, label, identity_id, split, val_role)
├── src/optlab/
│   ├── cli.py
│   ├── config.py             # Pydantic config models
│   ├── db/                   # connection, repository, merge, migrations
│   ├── data/                 # preprocess, face crop, frame sampling, extractors, FeatureDataset
│   ├── models/               # modality heads, fusion (weighted_sum, concat_mlp, svm_head)
│   ├── optim/                # registry, sgd.py, momentum.py, nag.py, rmsprop.py, adam.py, sam.py, hybrid/
│   ├── train/                # trainer, early stopping, checkpointing, clipping
│   ├── tune/                 # random, grid, pso, ga
│   ├── fusion/               # kkt_closed_form.py, qp.py, svm_dual.py
│   ├── eval/                 # metrics, bootstrap, stats tests
│   ├── toy/                  # benchmark functions, runner, trajectories
│   └── export/               # snapshot JSON writer
├── backend/
│   ├── schema.sql
│   └── api/                  # FastAPI app, routers, response models
├── frontend/                 # Vite app
├── tests/
├── notebooks/                # Colab/Kaggle runner notebooks (thin wrappers around the CLI)
├── results/                  # results.db (or release asset), curves/, tables/
├── artifacts/                # checkpoints (git-ignored)
└── docs/                     # PRD, TRD, UIUX, ARCHITECTURE, report
```

`artifacts/`, raw data, feature caches and large checkpoints are git-ignored. `results/results.db` is committed only if small; otherwise attached to a GitHub release and fetched by `make fetch-results`.

## 3. Data pipeline

### 3.1 Inputs

| Dataset | Role | Audio | Access |
|---------|------|:-----:|--------|
| FaceForensics++ (c23 default) | Train / val / in-dist test | no | Access request required |
| Celeb-DF v2 | Cross-dataset test only | no | Access request required |
| FakeAVCeleb | True audio+video experiments | yes | Access request required |

Never commit raw videos, frames or face crops. Respect each dataset's terms.

### 3.2 Steps

1. **Manifests.** Build identity-disjoint train/val/test splits with a fixed split seed. The `val` split carries a `val_role` column: `tune` (checkpoint and hyperparameter selection) and `fusion` (fitting fusion weights). Commit CSVs under `manifests/`.
2. **Face crop.** Detect, expand the box by a fixed margin, resize to the backbone's input size.
3. **Frame sampling.** K = 16 frames per video, uniform in time (configurable).
4. **Feature extraction (the only expensive step).** Run frozen backbones once and cache. Visual: one embedding per frame. Audio: one embedding per video. Frequency: one vector per frame.
5. **Cache format.** Per `{dataset}/{backbone}`: float16 `.npy` shards plus an index CSV (`video_id, label, frame_idx, row`). Loaded with memmap and cast to float32 per batch.

### 3.3 `FeatureDataset`

Yields `(x_visual, x_audio | None, x_freq | None, label, video_id)` per **frame**; the video-level audio embedding is broadcast to that video's frames. Training uses a class-balanced sampler (FF++ has several fakes per real). Video-level scores are the mean of frame logits.

## 4. Model

Primary configuration is **late fusion**:

- One small MLP head per active modality (`d_in → 256 → 1`, ReLU, dropout 0.2, no batch norm so SAM and clipping behave cleanly).
- Per-modality logits `z_m(x)`; fused logit `z(x) = Σ_m w_m z_m(x)` with `w` on the simplex (§9).
- Heads are trained in a single loop on the summed binary cross-entropy `Σ_m BCE(z_m, y)`. Parameters are disjoint; global-norm operations (clipping, SAM's perturbation norm) are computed **per head** so heads stay independent.
- Alternatives stored in `model_configs.fusion_type`: `concat_mlp` (joint MLP on concatenated features) and `svm_head` (soft-margin SVM on concatenated features; dual in §9.3).
- Batch size is fixed across all optimizers (default 256) and is not tuned.
- Learning-rate schedule is **constant** by default so the optimizer is not confounded with a schedule; a cosine-schedule ablation is optional.
- No mixed precision for heads (they are tiny; keeps numerics clean).

## 5. Training specification

| Setting | Default |
|---------|---------|
| Loss | `BCEWithLogitsLoss` |
| Max epochs | 30 |
| Early stopping | patience 5 on `val_tune` AUC; best checkpoint kept |
| Gradient clipping | off in baselines; `clip_norm` configurable; one ablation at 1.0 |
| Seeds | 0–4 (5); 0–9 recommended because heads are cheap |
| Seed controls | Parameter init and data order. Splits are **not** reseeded. |
| Determinism | Seed `random`, `numpy`, `torch`; enable deterministic algorithms where available; record library versions in `runs` |
| Logged per epoch | `train_loss, val_loss, val_auc, lr, grad_norm, step_time_sec` → `run_epochs` |
| Divergence | NaN/Inf loss → `runs.status = 'diverged'`, kept in the DB |
| Resumability | A run key is `(experiment, optimizer, model_config, seed)`. Re-running a `completed` key is a no-op. |

## 6. Optimizers

### 6.1 Interface

```python
class OptimizerSpec(Protocol):
    name: str
    family: Literal["plain","momentum","adaptive","sign","sharpness_aware","second_order","hybrid"]
    def build(self, params: Iterable[Tensor], hparams: Mapping[str, float]) -> torch.optim.Optimizer: ...
    search_space: Mapping[str, SpaceDim]    # shared with the tuner
```

The registry maps `name → OptimizerSpec` and mirrors the `optimizers` table. SAM and any hybrid that needs a second forward/backward expose `step(closure)`; the trainer always calls `optimizer.step(closure)` so every optimizer shares one code path.

### 6.2 Update rules

Let `g_t = ∇L(θ_t)`, learning rate `η`, weight decay `λ`.

| Optimizer | Update |
|-----------|--------|
| SGD | `θ ← θ − η g` |
| Momentum | `v ← μ v + g;  θ ← θ − η v` |
| NAG (PyTorch form) | `v ← μ v + g;  θ ← θ − η (g + μ v)` |
| RMSProp | `s ← ρ s + (1−ρ) g²;  θ ← θ − η g / (√s + ε)` |
| Adam | `m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²;  θ ← θ − η m̂ / (√v̂ + ε)` with bias correction `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)` |
| AdamW | Adam step plus decoupled decay: `θ ← θ − η (m̂/(√v̂+ε) + λθ)` |
| Lion | `c = sign(β₁m + (1−β₁)g);  θ ← θ − η (c + λθ);  m ← β₂m + (1−β₂)g` |
| SAM | `ε̂ = ρ g/‖g‖;  g′ = ∇L(θ + ε̂);  restore θ;  apply the base optimizer step with g′` |
| RAdam, AdaBelief | Library implementations; cite the papers |
| Newton / L-BFGS | Toy benchmarks and small-head experiments only |

### 6.3 Implementation rules

- **FR-O2:** SGD, Momentum, NAG, RMSProp and Adam are implemented from scratch as `torch.optim.Optimizer` subclasses. Tests assert agreement with `torch.optim` (§14).
- SAM is implemented in-repo as a wrapper over any base optimizer.
- L2 weight decay conventions are documented per optimizer. Coupled L2 for SGD-family and RMSProp/Adam; decoupled decay for AdamW and Lion.

### 6.4 Hybrid optimizer (Phase 3 only)

Not implemented before Gate 1. Requirements for any hybrid:

1. A written update rule in `docs/hybrid_design.md` before code.
2. Validation on the toy suite first.
3. Registered as `is_hybrid = 1`; each ablation registered as a child (`parent_optimizer_id`, `ablation_note`).
4. A blended step must account for scale mismatch: Adam's normalized step and SGD's gradient-scaled step differ in magnitude, so use separate learning rates or an explicit norm-matching rule, and document which.
5. Design frozen and committed before any `test`-split evaluation (Gate 2).

## 7. Tuning protocol

| Item | Specification |
|------|---------------|
| Budget | **Same number of trials for every optimizer** (default N = 20), recorded in `tuning_trials` |
| Method | Random search, log-uniform for scale parameters (default). Grid for ≤2 dims. PSO and GA as additional tuners (course lab). |
| Training per trial | Shortened (default 10 epochs) with early stopping |
| Selection | Highest `val_tune` AUC; the chosen trial is flagged `is_selected = 1` and referenced by `runs.tuning_trial_id` |
| Tuned dimensions | `lr` (log-uniform 1e-5…1e0, narrowed per family in `configs/optim/`) and `weight_decay` (0 or log-uniform 1e-6…1e-2) for **all** optimizers. SAM adds `rho`. Momentum/β fixed at standard defaults. |
| Supplementary analysis | An "equal-dimension" pass where baselines also tune momentum/β₁, to compare against hybrids that have extra hyperparameters |
| Search spaces | Stored in `optimizers.default_search_space_json`; one tuner, one budget, enforced in code |

## 8. Evaluation and statistics

### 8.1 Protocols

| Protocol | Train | Test |
|----------|-------|------|
| `in_distribution` | FF++ train | FF++ test (identity-disjoint) |
| `cross_dataset` | FF++ train | Celeb-DF v2 test (no tuning or checkpoint selection on it) |
| Optional | FF++ minus one manipulation method | The held-out method |

Metrics at `frame` and `video` granularity: AUC, EER, accuracy (threshold fixed from validation), F1, average precision.

### 8.2 Test-set discipline (FR-E3)

`optlab evaluate --final` writes `split='test'` rows once per run. A second write requires `--force` and sets `evaluations.force_rewritten = 1`. Tuning, early stopping, fusion fitting and hybrid design may only read `val`.

### 8.3 Statistics

- Per optimizer: mean ± sample std across seeds, plus a bootstrap 95% CI (bootstrap over seeds for run-level uncertainty; over videos for within-run uncertainty).
- Pairwise comparisons: difference in means with bootstrap CI and Cohen's d. Welch's t-test as a secondary check, with **Holm** correction across comparisons.
- Do not rely on the Wilcoxon signed-rank test at n = 5: its smallest possible two-sided p-value is 0.0625. With few seeds, prefer CIs and effect sizes, and state low power plainly.
- Convergence metrics: epochs to reach a target validation AUC (e.g. 0.95 × the best value across optimizers), area under the validation-loss curve, wall-clock to target, divergence rate.
- Generalization gap: `AUC_in_dist − AUC_cross_dataset` per run (`v_generalisation_gap`).

## 9. Constrained fusion

### 9.1 Problem

Fit `w ∈ Δ = { w : w ≥ 0, 1ᵀw = 1 }` on the `val_fusion` subset:

```
minimize   L(w) = (1/n) Σ_i  log(1 + exp(−ỹ_i · wᵀ z_i)),      ỹ_i ∈ {−1, +1}
subject to w ≥ 0,  1ᵀ w = 1
```

`L` is convex in `w` (a convex loss composed with an affine map), so the problem is convex and Slater's condition holds (the uniform vector is strictly feasible).

### 9.2 KKT conditions

```
∇L(w)_m + λ − μ_m = 0,     μ_m ≥ 0,     μ_m w_m = 0,     1ᵀ w = 1
```

- **Solver A (`qp_cvxpy`)**: solve with `cvxpy`; recover `λ` and `μ` from the dual variables.
- **Solver B (`kkt_closed_form`)**: a squared-loss variant `L(w) = ‖Zw − y‖²/n` gives a closed form when the active set is empty: solve `[2ZᵀZ/n  1; 1ᵀ  0][w; λ] = [2Zᵀy/n; 1]`; if any `w_m < 0`, fix that weight at 0 and re-solve on the remaining modalities.
- **Solver C (`projected_gradient`)**: projection onto the simplex; used as a cross-check.
- Store `fusion_weights`, and in `fusion_solves` the objective, the **KKT residual** (norm of stationarity plus complementarity violation), `λ`, and the active set. Tests require residual < 1e-6 for A and B.

### 9.3 SVM head (optional)

Soft-margin SVM on concatenated features, solved through the dual (`cvxpy` or `sklearn`), with a short write-up of the dual, support vectors and Slater's condition.

## 10. Toy benchmarks

| Benchmark | Dim | Convex | Purpose |
|-----------|-----|:------:|---------|
| Ill-conditioned quadratic `½ xᵀAx`, κ(A) ∈ {10, 100, 1000} | 2–50 | yes | Condition-number effect, momentum vs. plain GD, Newton as reference |
| Rosenbrock | 2 | no | Narrow curved valley |
| Rastrigin | 2–10 | no | Many local minima; PSO/GA comparison |
| Himmelblau (optional) | 2 | no | Multiple global minima |

Runner: fixed start points per seed, `max_steps` per benchmark, tolerance on `|f − f*|`. Writes `toy_runs` (steps to tolerance, final value, final gradient norm), `toy_run_curves`, and a `.npz` trajectory for the 2-D viewer. The toy suite must run on CPU in minutes.

## 11. Database layer

- `backend/schema.sql` is applied by `optlab init-db`; `PRAGMA user_version` tracks the schema version. Migrations are numbered SQL files under `src/optlab/db/migrations/`.
- Writers use parameterized statements only. The API opens the file read-only (`file:results.db?mode=ro`, `uri=True`).
- Standard deviation is computed in Python from the `auc_var` and `eer_var` columns of `v_leaderboard`.
- `optlab merge-db a.db b.db --out merged.db` merges partial databases from separate Colab/Kaggle sessions using the `UNIQUE` keys; conflicting finished runs raise an error rather than overwrite.

## 12. API specification (read-only)

Base path `/api`. JSON responses, Pydantic models, pagination via `limit`/`offset` where lists can grow.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + schema version |
| GET | `/experiments` | List with phase and run counts |
| GET | `/experiments/{id}` | Details, hypothesis, hyperparameter budget |
| GET | `/optimizers` | Catalogue incl. family, update rule, hybrid/ablation links |
| GET | `/leaderboard` | Params: `experiment_id`, `eval_dataset`, `protocol`, `granularity`, `metric`. Returns mean, std, n_seeds, CI per optimizer |
| GET | `/convergence` | Params: `experiment_id`, `optimizer_ids[]`, `metric` (`val_loss`, `val_auc`, `train_loss`, `grad_norm`). Returns per-epoch mean and std across seeds |
| GET | `/generalisation` | Per-optimizer in-dist AUC, cross-dataset AUC, gap with CIs |
| GET | `/stats/pairwise` | Params: `experiment_id`, `metric`, `eval_dataset`. Mean difference, bootstrap CI, Cohen's d, Holm-adjusted p |
| GET | `/tuning/{experiment_id}/{optimizer_id}` | All trials, selected trial, LR-sensitivity data |
| GET | `/runs` | Filter by experiment, optimizer, seed, status |
| GET | `/runs/{id}` | Run summary, hyperparameters, evaluations |
| GET | `/runs/{id}/epochs` | Full curve |
| GET | `/fusion/{run_id}` | Weights, solver, objective, KKT residual, multiplier, active set |
| GET | `/toy/benchmarks` | Catalogue |
| GET | `/toy/runs` | Params: `benchmark_id`, `optimizer_ids[]` |
| GET | `/toy/runs/{id}/trajectory` | Path for the contour viewer |
| POST | `/infer` *(stretch)* | Multipart video upload; returns fused score and per-modality scores; 25 MB limit, temp file deleted after the request |

Errors: `404` for unknown ids, `422` for bad parameters, JSON body `{ "detail": "..." }`. CORS limited to the configured frontend origin.

### Leaderboard response shape

```json
{
  "experiment_id": 1,
  "eval_dataset": "celeb_df_v2",
  "protocol": "cross_dataset",
  "granularity": "video",
  "rows": [
    { "optimizer": "adamw", "family": "adaptive", "n_seeds": 5,
      "auc_mean": 0.0, "auc_std": 0.0, "auc_ci95": [0.0, 0.0],
      "eer_mean": 0.0, "wall_clock_mean_sec": 0.0 }
  ]
}
```

(Numbers above are placeholders for the contract only.)

## 13. Frontend technical notes

- Data access through TanStack Query hooks, one per endpoint, typed from generated TypeScript types (`openapi-typescript` against the FastAPI schema).
- **Two data modes**, chosen by `VITE_DATA_MODE`: `api` (FastAPI) and `snapshot` (static JSON from `optlab export-snapshot`, laid out under `frontend/public/data/` with the same shapes as the API responses).
- Charts: Plotly for line charts with std bands, scatter with error bars, contour + trajectory, bar charts.
- Theme tokens, layout and component rules are in `docs/UIUX.md`.

## 14. CLI

```
optlab init-db                                   # apply schema.sql, seed catalogues from configs
optlab prepare   --dataset ff++ --out manifests/ # splits
optlab extract   --dataset ff++ --backbone clip_vit_b16
optlab toy       --benchmarks all --optimizers all --seeds 0-4
optlab tune      --experiment baseline_v1 --optimizer adamw --method random --trials 20
optlab train     --experiment baseline_v1 --optimizer adamw --seed 0
optlab fuse      --run-id 12
optlab evaluate  --run-id 12 [--final] [--force]
optlab sweep     --experiment baseline_v1        # everything above, resumable
optlab stats     --experiment baseline_v1        # writes tables/ and plots
optlab merge-db  a.db b.db --out merged.db
optlab export-snapshot --out frontend/public/data
optlab serve                                     # FastAPI on :8000
```

`Makefile`: `setup`, `test`, `lint`, `toy`, `baseline`, `snapshot`, `dashboard`, `fetch-results`.

### Experiment config example

```yaml
name: baseline_v1
phase: baseline
hypothesis: >
  Adaptive optimizers converge faster but show a larger in-distribution to
  cross-dataset gap than SGD with momentum; SAM narrows the gap.
train_dataset: faceforensics_pp
eval:
  in_distribution: faceforensics_pp
  cross_dataset: [celeb_df_v2]
model: clip_vit_b16__freq__weighted_sum
optimizers: [sgd, sgd_momentum, nag, rmsprop, adam, adamw, radam, adabelief, lion, sam_sgd_momentum]
tuning: { method: random, trials: 20, epochs: 10 }
training: { max_epochs: 30, batch_size: 256, patience: 5, clip_norm: null, schedule: constant }
seeds: [0, 1, 2, 3, 4]
```

## 15. Testing and CI

| Area | Tests |
|------|-------|
| Optimizers | In float64, run 100 steps on a random quadratic and assert agreement with `torch.optim` (SGD, Momentum, NAG, RMSProp, Adam) within 1e-6; SAM restores parameters exactly after the perturbation; AdamW decay is decoupled |
| Fusion | Solutions lie on the simplex; KKT residual < 1e-6; closed form agrees with `cvxpy` on random problems |
| Metrics | AUC matches `sklearn`; EER on hand-made ROC cases; video aggregation correct |
| Tuner | Same trial budget for every optimizer; diverged trials recorded |
| DB | Schema applies cleanly; constraints reject bad rows; `v_leaderboard` variance matches NumPy `var(ddof=1)`; test-write guard blocks a second write without `--force` |
| API | Contract tests with `TestClient` against a small seeded in-memory DB |
| Frontend | Vitest for formatting and data utilities; optional Playwright smoke test of the leaderboard |
| CI (GitHub Actions) | `ruff`, `pytest` on CPU with tiny fixtures, frontend build; no datasets needed |

## 16. Reproducibility

- Every run stores config path, git commit, seed, and library versions.
- Toy suite reproducible with `make toy` on CPU.
- Feature extraction records backbone id, preprocessing parameters and manifest hash in an `artifacts` row of kind `feature_cache`.
- The README lists exact commands, expected runtimes and the dataset-access steps.

## 17. Compute planning

Feature extraction is the only heavy step; all training happens on cached features through small heads. Planning assumptions to be replaced by measured numbers in the README after the first full run:

- Extraction dominates GPU time and is done once per dataset and backbone.
- Head training is cheap enough that all tuning trials and all seeds fit in one or two sessions on a free GPU or even CPU.
- SAM costs about twice the per-step compute of its base optimizer.
- Colab sessions are ephemeral, so write the DB to a mounted Drive path or use `merge-db`.

## 18. Security and ethics

- Detection only; no generation code.
- Datasets are used under their licences; no redistribution; only manifests committed.
- Demo uploads: size limit, content-type check, processed in a temp directory, deleted after the request, never logged beyond scores.
- Read-only database connection for the API; parameterized SQL; no credentials in the repo.
- Reports state that detectors can fail on unseen generators and must not be used as sole evidence about a real person's media.

## 19. Implementation order

1. Repo skeleton, config models, `schema.sql` → `init-db`, CI.
2. Optimizers (from scratch) + tests → toy benchmarks → toy plots.
3. Data manifests, face crop, extraction, cache, `FeatureDataset`.
4. Trainer, logging to DB, early stopping, checkpointing.
5. Tuner (random first, then grid, PSO, GA), evaluation, test-write guard.
6. Fusion solvers and tests.
7. `sweep` for the baseline experiment; `stats`.
8. API, then dashboard views in the order of `docs/UIUX.md` §12.
9. Hybrid design document (after Gate 1), implementation, ablations.
10. README, report, snapshot export, optional demo.
