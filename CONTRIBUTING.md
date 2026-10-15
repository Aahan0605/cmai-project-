# Contributing

OptLab-DF is a research benchmark and a read-only results dashboard. Keep code changes,
experimental design, and claims about model quality traceable.

## Development

Use Python 3.11+, uv, Node 22.12+ (Node 24 is used in CI), and pnpm 10.28.1.
Run commands from the repository root unless a command explicitly uses `--dir frontend`.

```bash
make setup
make check
```

`make setup` installs the Python API extra and frontend dependencies. Extraction dependencies
are separate: `uv sync --frozen --extra api --extra extract`. That command needs a compatible
PyTorch/torchvision installation. No dataset downloads or pretrained weights are needed for
the core checks. Optional video tests skip when OpenCV is absent.

- Python: `uv run pytest`; lint/format with `uv run ruff check src tests` and
  `uv run ruff format src tests`.
- Frontend: `pnpm --dir frontend test`, `pnpm --dir frontend build`,
  `pnpm --dir frontend format`.
- UI: `make dev`. Live mode: `make api`, then
  `VITE_DATA_MODE=api pnpm --dir frontend dev` in a second terminal.

## Result integrity

1. Make a new experiment YAML and use a new `name` when changing training, model,
   optimizer, search, preprocessing, or fusion settings. A natural run key identifies a
   result; it is not a configuration fingerprint. Resume only unchanged experiments.
2. Split by connected identity components, including face-swap donors. Training,
   validation-tuning, validation-fusion, and test identities must remain disjoint.
3. Use validation for tuning, checkpoint selection, calibration, thresholds, and fusion.
   Freeze these choices before final evaluation. Do not refit fusion after test evaluation.
4. Test rows are written once. `evaluate --force` explicitly overwrites and flags them;
   it is for correcting a disclosed mistake, not selecting a model against test labels.
5. Preserve diverged runs, report seed/fold counts, and distinguish seed variability from
   uncertainty across people. The fold training sets overlap, so fold estimates are dependent.
6. Freeze a written hybrid design before evaluating it on a new unseen test. The hybrid and
   fine-tuning ablations remain research work; the cached-feature trainer does not unfreeze
   backbone layers.

After changing queries, test API/snapshot parity and regenerate `frontend/public/data` with
`make snapshot` from the intended results database. Snapshots contain real aggregate and
per-run records. Do not put raw datasets, faces, waveforms, feature arrays, credentials,
checkpoints, private file contents, or local database files in a commit.

## Pull requests

Explain the user-visible problem, the change, and the checks run. Add a regression test for
an actual failure or a meaningful boundary. Keep generated research outputs separate from
implementation descriptions, and identify the config that produced any reported number.

CI checks Python lint/format/tests, wheel creation, empty-database export, frontend format,
frontend tests, and the production build. A successful core test suite does not certify
external-dataset parsers or GPU execution against data/hardware that was unavailable.
