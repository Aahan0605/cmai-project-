# Development review — 2026-10-15

Scope: the local checkout of `Aahan0605/cmai-project-`, including its existing uncommitted
calibration/frontend work. The GitHub remote was checked for context; this document describes
local changes and verification, not a published GitHub release.

## Fixed issues

| Area | Failure | Resolution |
|---|---|---|
| Evaluation | Configured in-distribution dataset was ignored | Load and record the configured test target; validation remains on the training dataset |
| Generalization | Multiple external datasets were pooled; frontend could compare unmatched seeds | Calculate each dataset gap from paired run evaluations and expose paired seed counts |
| SAM | Failure in the second closure left model parameters perturbed | Restore original parameters in `finally` |
| SAM checkpoints | Base optimizer momentum/state was omitted | Share and reload the base optimizer state |
| Training | Experiment `device` was ignored | Move the model and batches to the selected device; load evaluation checkpoints onto CPU |
| Training/data | Single-class validation and non-finite/misaligned features could produce unusable results | Validate split dimensions, finite values, binary classes and finite AUC |
| Run ownership | A run/trial could be used with the wrong experiment object | Check experiment, optimizer/model and dataset ownership before use |
| Search | Empty grid dimensions divided by zero; `p_zero=1` failed at the upper endpoint | Preserve exact budgets for empty spaces; handle certain-zero dimensions |
| Resume | Changed deterministic search proposals silently reused old scores | Compare recorded and proposed hyperparameters; require a new experiment name on mismatch |
| Fusion | Projected Newton stalled on a zero-weight boundary | Use a feasible projected-gradient fallback; retain KKT acceptance checks |
| Calibration | Column-vector labels could broadcast into an incorrect loss | Require aligned 1D labels and at least one stream |
| Feature cache | Duplicate keys, float16 overflow, corrupt/reordered shards, inconsistent dimensions | Validate writes and reads; publish complete caches from staging |
| Modalities | Audio/frequency label mismatches and missing audio were discovered too late | Validate all active cache labels and audio coverage before loading |
| Manifests | Duplicate videos and identities crossing validation roles were accepted | Check identity role consistency when reading manifests |
| SQLite | Reserved URI characters broke read-only database paths | Use an escaped file URI |
| Migrations/evaluations | Failed migration or forced evaluation could leave partial changes | Transactional migrations and savepoint-protected replacement |
| Merge | Identical parent run rows could hide differing evaluations/weights; conflicts left partial inserts | Compare recorded children and catalogue definitions; rollback the operation |
| API/export | Old schemas failed later through opaque SQL errors | Check schema compatibility before serving/exporting |
| Snapshot | Failed exports could damage a previous snapshot | Build from a consistent database backup and stage before publishing |
| Frontend | Missing confidence intervals appeared as zero; null curve values were drawn incorrectly | Explicit missing-interval states and line breaks for missing observations |
| Frontend | New/running experiments and toy-only databases were awkward to navigate | Select available runs and make toy/protocol views independent of detector runs |
| Frontend | A foreign run URL could show details under the wrong experiment | Verify the selected run belongs to the current run list |
| Frontend | Generalization/run details did not refresh consistently in API mode | Share refresh behavior and refresh run details |
| Frontend | Theme toggle could disagree with the system default; icons polluted accessible names | Initialize from saved/system theme; hide decorative icons from accessibility names |
| Frontend | Tuning and toy views were absent | Add inspectable trial tables, objective curves, available 2D paths, per-seed outcomes and CSV |
| Build/CI | Frontend formatting/tests/build were missing from CI | Pin the package manager, use the lockfile and add a Node CI job |
| Documentation | Setup assumed a DB/active shell; architecture mixed current and planned systems | Rewrite the README/architecture, add contribution/security guidance and mark planning docs |

## Verification

- Python regression/integration/contract suite, Ruff lint and formatting.
- Frontend component/data-boundary tests, Prettier, TypeScript and Vite production build.
- Python wheel build and frozen frontend dependency installation.
- Fresh database initialization, empty snapshot export and API health/catalogue smoke test.
- Every committed manifest read through the new validation rules.
- Local SQLite `integrity_check` reports `ok`; `foreign_key_check` reports no violations.
- Every exported snapshot file parses as strict JSON; CV snapshot reports eight folds for all optimizers.
- Browser checks of toy curves/trajectories, tuning trials and matched-seed generalization with real records.
- The calibrated sweep completed 400 runs; its outputs remain recorded separately from historical baselines.

Synthetic saturated-AUC fixtures can emit SciPy precision-loss warnings for nearly identical
seed values. Those warnings do not indicate a production build or test failure, and the
project does not treat undetectable differences as equivalence.

## Remaining work and limits

- FF++/Celeb-DF evaluation needs author-granted access, extraction and real-layout verification.
- FakeAVCeleb preparation needs verified dataset layout and a dataset-backed parser.
- Hybrid design, confirmatory evaluation and ablations remain research milestones. Existing
  test results cannot become an untouched evaluation after observing them.
- Concat/SVM heads and backbone fine-tuning remain future model variants. Cached-head runs
  explicitly reject unsupported model/backbone settings.
- CPU paths are exercised locally; accelerator execution and external pretrained downloads
  need their own hardware/data validation.
- Configuration names are natural keys, not immutable fingerprints of all inputs. Use new
  names for changed scientific settings and preserve their original configs/caches.
- The API is for local/read-only research use. Public live serving needs an authenticated
  reverse proxy and a review of exported records; static hosting is the simpler public demo.

This review improves the implemented system and documents its limits. It is not a guarantee
that arbitrary future configurations, unavailable datasets or every hardware backend are bug-free.
