# OptLab-DF

**Optimizer comparison and hybrid optimizer design for multimodal deepfake detection.**

Innovative assignment for *Computational Mathematics for AI & ML* (B.Tech CSE-AIML, Nirma University).

> When training a multimodal deepfake detector, how do different optimizers compare in
> convergence speed, stability and, above all, generalization to deepfakes from unseen
> datasets? Can a hybrid optimizer designed from those findings do better?

Optimization is the subject here; deepfake detection is the testbed. Detectors score very
high on the dataset they were trained on and degrade on unseen manipulations, and that gap is
where the choice of optimizer (and flat versus sharp minima) can plausibly matter.

## Status

| Phase | Content | State |
|---|---|---|
| 0 Setup | Repo, configs, results database, data pipeline | Done |
| 1 Toy benchmarks | Optimizers verified on functions with known minima | Done ([summary](results/toy_summary.md)) |
| 2 Baselines | Fair tuning, 5 seeds, in-distribution and cross-dataset evaluation | Waiting for dataset access |
| 3 Hybrid design | Update rule designed from the Phase 2 findings, frozen before testing | Not started |
| 4 Hybrid evaluation | Equal-budget comparison and ablations | Not started |
| 5 Presentation | Read-only API and dashboard | Not started |

No deepfake-detection results exist yet. Every number in this repository is produced by a
command and stored in the results database; nothing is typed in by hand.

## Method

**Optimizers.** SGD, SGD + momentum, Nesterov (NAG), RMSProp and Adam are implemented from
scratch as `torch.optim.Optimizer` subclasses and tested against PyTorch: they match within
1e-6 over 100 float64 steps. AdamW, RAdam, AdaBelief and Lion use library implementations.
SAM (sharpness-aware minimization) is implemented here as a wrapper around any base optimizer.
Newton and L-BFGS serve as second-order references on the toy problems.

**Fair comparison** (Phase 2 onward):
- Every optimizer gets the same tuning budget over a documented search space.
- Selection uses validation AUC only.
- Each configuration runs with 5 seeds.
- Results are reported as mean ± std with bootstrap confidence intervals and Holm-corrected tests.
- Test-split evaluations are written once per run; a second write needs `--force` and is flagged
  in the database.

**Model.** Frozen backbones with cached features (CLIP ViT-B/16 for faces, wav2vec 2.0 for audio
when available, and a hand-built frequency spectrum), small per-modality heads, and late fusion
whose weights lie on the probability simplex. The fusion weights are solved as a constrained
problem via the KKT conditions or a QP.

**Data.** FaceForensics++ for training with identity-disjoint splits. Celeb-DF v2 is used only
as an unseen cross-dataset test set. FakeAVCeleb is used for true audio + video once access is
granted. Only split manifests are ever committed, never videos, frames or faces.

## Toy benchmark results (Phase 1)

Every optimizer runs on ill-conditioned quadratics (κ = 10, 100, 1000), Rosenbrock and
Rastrigin, with 5 seeds each. The full tables (steps to tolerance, final values, selected
learning rates), the protocol and the hypothesis recorded before the runs are in
[results/toy_summary.md](results/toy_summary.md).

![Rosenbrock trajectories](results/toy/rosenbrock_trajectories.png)

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is installed automatically.

```bash
make setup   # install dependencies
make test    # unit tests on CPU
make toy     # rerun the toy suite and regenerate results/toy_summary.md and plots
```

`make toy` runs entirely on CPU. It took 349 s on an Apple-silicon laptop.

## Command-line interface

```bash
optlab init-db                                   # create results/results.db, load catalogues
optlab toy [--force]                             # toy benchmarks (resumable)
optlab toy-report                                # tables, plots and summary from the DB
optlab prepare --dataset ff++ --root DATA/FF++   # identity-disjoint split manifests
optlab extract --dataset ff++ --backbone clip_vit_b16 --root DATA/FF++
optlab merge-db a.db b.db --out merged.db        # combine databases from separate GPU sessions
```

Feature extraction needs the optional extras (`uv sync --extra extract`) and a local copy of
each dataset. Access to FaceForensics++, Celeb-DF v2 and FakeAVCeleb is granted by their
authors on request.

## Repository layout

```
configs/          optimizer search spaces, model, dataset, split and experiment configs (YAML)
src/optlab/
  optim/          from-scratch optimizers, SAM, registry
  toy/            benchmark functions, runner, report
  data/           manifests, face crop, frame sampling, extractors, feature cache, dataset
  db/             SQLite connection, migrations, repository, merge
  cli.py          the optlab command
backend/schema.sql  results database schema (migrations in src/optlab/db/migrations/)
results/          generated toy summary, tables and plots
tests/            pytest suite (runs on CPU, no datasets needed)
docs/             PRD, TRD, architecture and dashboard design
```

## Reproducibility

- Configs are committed and runs are resumable.
- The database records each run's seed, hyperparameters, git commit and (from Phase 2)
  library versions.
- CI runs ruff and the full test suite on every push.

## Limitations

- Backbones are frozen, so findings may not transfer to full fine-tuning. Phase 4 includes a
  limited unfreeze-last-layers check.
- FaceForensics++ and Celeb-DF have no audio. Without FakeAVCeleb the model is multi-stream
  (visual + frequency) rather than truly multimodal.
- Toy learning rates are chosen on seed 0 and reused for every seed, so seed 0 is not held out.
- This project only classifies media; it contains no generation code. Detectors can fail on
  unseen generators and must not be used as the only evidence about a real person's media.

## References

- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization (AdamW).
- Liu et al. (2020). On the Variance of the Adaptive Learning Rate and Beyond (RAdam).
- Zhuang et al. (2020). AdaBelief Optimizer.
- Chen et al. (2023). Symbolic Discovery of Optimization Algorithms (Lion).
- Foret et al. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization.
- Rössler et al. (2019). FaceForensics++.
- Li et al. (2020). Celeb-DF.
- Khalid et al. (2021). FakeAVCeleb.

## License

MIT. See [LICENSE](LICENSE).
