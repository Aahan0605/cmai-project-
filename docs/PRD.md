# Product Requirements Document

> Planning specification: this document includes future research work. The [README status](../README.md#project-status) and [system architecture](ARCHITECTURE.md) describe the current implementation.

**Project:** OptLab-DF — Optimizer comparison and hybrid optimizer design for multimodal deepfake detection
**Package name:** `optlab`
**Course:** Computational Mathematics for AI & ML (Nirma University, B.Tech CSE-AIML) — innovative assignment
**Author:** Aahan Gajera
**Status:** Draft v1
**Last updated:** 2026-10-12

---

## 1. Summary

OptLab-DF is a reproducible experiment suite and results dashboard that answers one question with evidence:

> **When training a multimodal deepfake detector, how do different optimizers compare in convergence speed, stability and — most importantly — generalization to deepfakes from unseen datasets, and can a hybrid optimizer designed from those findings do better?**

The project treats optimization as the subject, not a tool. Deepfake detection is the testbed because it has a real, well-known generalization problem: detectors score very high on the data they were trained on and degrade on unseen manipulation methods. That gap is where optimizer choice (and flat vs. sharp minima) can plausibly matter.

The repository must work as a standalone portfolio piece: clone, read the README, reproduce the toy benchmarks in minutes, and inspect every number in the results table.

## 2. Problem and motivation

- Most "optimizer comparison" projects pick one dataset, tune one optimizer harder than the rest, report accuracy from a single seed, and declare a winner. The conclusion is usually an artifact of the protocol.
- Deepfake detectors are evaluated mostly in-distribution, which hides the failure mode that matters in practice.
- The course syllabus covers the maths (gradient descent variants, momentum, NAG, second-order methods, constrained and convex optimization, KKT, SVM duality, metaheuristics) but a taught algorithm is rarely tested on a realistic ML training problem with fair experimental design.

## 3. Goals

| ID | Goal |
|----|------|
| G1 | Compare a defined set of optimizers on a multimodal deepfake detector under a **fair, equal-budget tuning protocol** with **multiple seeds**. |
| G2 | Measure **in-distribution and cross-dataset** performance and convergence behaviour, and report whether differences are statistically distinguishable. |
| G3 | Verify the optimizer mathematics on **toy benchmarks** with known optima before any deepfake training. |
| G4 | Use **constrained optimization** (Lagrange/KKT, QP) to learn modality fusion weights on the simplex. |
| G5 | Design and evaluate a **hybrid optimizer** *after* baseline analysis, with a frozen design and the same tuning budget as baselines. |
| G6 | Ship a clean GitHub repo: reproducible CLI, committed configs, results database, read-only dashboard, honest README. |
| G7 | Map every component to the syllabus (units, labs, CLOs) so the assignment is evaluable as computational mathematics. |

## 4. Non-goals

- Building a state-of-the-art deepfake detector or beating published leaderboards.
- Training or fine-tuning large backbones end to end (a limited "unfreeze last N layers" ablation is in scope).
- Generating deepfakes. The project only classifies; no generation code is included.
- Hosting or redistributing any dataset. The repo holds manifests (video IDs and labels) only.
- Real-time or production-grade inference. A demo upload page is a stretch goal, not a requirement.
- Claiming the hybrid optimizer is "novel" without a literature check. Related prior work includes SWATS, Lookahead/Ranger and SAM variants.

## 5. Audience

| Persona | Needs |
|---------|-------|
| **Course evaluator** | See that syllabus concepts are implemented and analysed correctly; verify convergence analysis (CLO 4); find the maths quickly. |
| **Recruiter / reviewer of the GitHub repo** | Understand the question and result in 60 seconds from the README; trust the methodology; run something. |
| **Author (future self)** | One command per phase, resumable runs, no manual bookkeeping. |

## 6. Scope by phase

| Phase | Content | Exit criteria |
|-------|---------|---------------|
| **0 Setup** | Repo skeleton, config system, results DB, data access requests, face crop + frame sampling, feature extraction and cache, manifests with identity-disjoint splits | `optlab extract` produces a cache; DB initialises from `schema.sql`; unit tests pass |
| **1 Toy benchmarks** | All optimizers implemented and run on an ill-conditioned quadratic, Rosenbrock, Rastrigin (and optionally Himmelblau) | Steps-to-tolerance table; 2-D trajectory plots; L-BFGS and Newton reference results; optimizer unit tests |
| **2 Baselines** | Equal-budget LR/hyperparameter search per optimizer, 5 seeds each, in-dist + cross-dataset evaluation, convergence logging | All baseline runs completed; leaderboard with mean ± std and CIs; written baseline analysis (Gate 1) |
| **3 Hybrid design** | Written update rule(s), toy validation, validation-only tuning | Design document frozen (Gate 2) |
| **4 Hybrid evaluation** | Hybrid vs. best baselines at equal budget; ablations; "unfreeze last N layers" check | Ablation table; statistical comparison; honest limitations |
| **5 Presentation** | Dashboard, README, short report, optional demo | Dashboard covers all views in `docs/UIUX.md`; README has results table and reproduction steps |

Phases 0–2 are the minimum viable project. Phases 3–5 are added in order of value.

## 7. Optimizers in scope

| Family | Optimizer | In syllabus | Notes |
|--------|-----------|:-----------:|-------|
| Plain | SGD (mini-batch) | yes | Baseline |
| Momentum | SGD + Momentum | yes | |
| Momentum | Nesterov (NAG) | yes | |
| Adaptive | RMSProp | yes (lab 7) | |
| Adaptive | Adam | yes (lab 7) | |
| Adaptive | AdamW | no | Decoupled weight decay |
| Adaptive | RAdam | no | Variance rectification |
| Adaptive | AdaBelief | no | |
| Sign-based | Lion | no | |
| Sharpness-aware | SAM (wrapping SGD+M and/or AdamW) | partly (flat minima, unit II) | ~2x cost per step |
| Second-order | L-BFGS (small head / toy only) | yes (quasi-Newton) | Not used for mini-batch training |
| Hybrid | Candidates H1–H3 (see §9) | — | Phase 3 |

If compute is tight, drop to the six core optimizers before reducing seeds or tuning budget.

## 8. Functional requirements

### 8.1 Data and features

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-D1 | Ingest FaceForensics++ (primary training set) and Celeb-DF v2 (cross-dataset test set) from locally available copies. | Must |
| FR-D2 | Support FakeAVCeleb for true audio+video fusion when access is granted; otherwise fall back to visual + frequency streams on video-only datasets. | Should |
| FR-D3 | Detect and crop faces, sample K frames per video uniformly, and cache frozen-backbone features to disk. | Must |
| FR-D4 | Produce identity-disjoint train/val/test manifests (CSV) with a recorded split seed; commit manifests, never raw videos. | Must |
| FR-D5 | Record dataset metadata and split manifests in the results DB. | Must |

### 8.2 Optimizers and training

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-O1 | Every optimizer sits behind one registry interface and is created from a name + hyperparameter dict. | Must |
| FR-O2 | Core optimizers (SGD, Momentum, NAG, RMSProp, Adam) are **implemented from scratch** as `torch.optim.Optimizer` subclasses and unit-tested against the PyTorch reference implementations. | Must |
| FR-O3 | Extended optimizers (AdamW, RAdam, AdaBelief, Lion) may wrap library implementations; SAM is implemented in the repo as a wrapper over a base optimizer. | Must |
| FR-O4 | Trainer supports gradient clipping, early stopping, best-checkpoint saving, and logs per-epoch loss, validation AUC, learning rate, gradient norm and step time. | Must |
| FR-O5 | Runs are resumable and idempotent: re-running a finished (experiment, optimizer, seed) is a no-op. | Must |

### 8.3 Fair tuning

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-T1 | Each optimizer gets the **same number of tuning trials** over a documented search space; selection uses validation AUC only. | Must |
| FR-T2 | Tuning methods: random search (default), grid (small spaces), PSO and GA (as in the course lab). | Must / Should |
| FR-T3 | Diverged trials are recorded, not silently dropped. | Must |

### 8.4 Evaluation and statistics

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-E1 | Metrics: AUC, EER, accuracy, F1, average precision; frame-level and video-level (mean of frame scores). | Must |
| FR-E2 | In-distribution test (held-out FF++) and cross-dataset test (Celeb-DF v2); optional leave-one-manipulation-out on FF++. | Must / Should |
| FR-E3 | Test-split evaluations are written once per run; a second write requires `--force` and is flagged in the DB. | Must |
| FR-E4 | Report mean ± std across seeds, bootstrap 95% CIs, and pairwise tests with multiple-comparison correction. State plainly when differences are within noise. | Must |
| FR-E5 | Generalization gap (in-dist AUC − cross-dataset AUC) reported per optimizer. | Must |

### 8.5 Constrained fusion

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-F1 | Learn modality fusion weights on the simplex (w ≥ 0, Σw = 1) by minimizing validation loss, solved via KKT conditions and/or a QP solver. | Must |
| FR-F2 | Store the solution, the active set, the Lagrange multiplier and the KKT residual. | Should |
| FR-F3 | Provide an SVM (soft-margin) fusion head as an alternative, with the dual formulation discussed in the report. | Should |

### 8.6 Toy benchmarks

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-B1 | Run all optimizers on: ill-conditioned convex quadratic, Rosenbrock, Rastrigin. Record steps-to-tolerance, final value and gradient norm. | Must |
| FR-B2 | Save 2-D trajectories for contour visualisation. | Must |
| FR-B3 | Include Newton and L-BFGS as second-order references on the quadratic and Rosenbrock. | Should |

### 8.7 Hybrid optimizer

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-H1 | Write the update rule mathematically before implementing; validate on toy benchmarks. | Must |
| FR-H2 | Tune with the same budget, the same search space discipline and the same seed count as the baselines. | Must |
| FR-H3 | Ablate each component (full hybrid vs. each piece removed). Ablations are stored as child optimizers via `parent_optimizer_id`. | Must |
| FR-H4 | The design is frozen and committed (Gate 2) before any test-split evaluation of the hybrid. | Must |

### 8.8 Presentation

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-P1 | Read-only API over the results DB (spec in `docs/TRD.md`). | Must |
| FR-P2 | Dashboard with the views in `docs/UIUX.md`: overview, leaderboard, convergence, generalization, tuning, toy benchmarks, fusion, run detail, hybrid. | Must |
| FR-P3 | `optlab export-snapshot` writes static JSON so the dashboard can be hosted without a backend. | Should |
| FR-P4 | README with the research question, headline table, plots, reproduction commands, findings and limitations. | Must |
| FR-P5 | Demo page: upload a short clip, get a score and per-modality scores. Local only, no storage of uploads. | Could |

## 9. Hybrid optimizer candidates (Phase 3 — not to be implemented until Gate 1)

Candidates are starting points. The final design must be motivated by the Phase 2 findings.

| ID | Idea | Motivation to test |
|----|------|--------------------|
| H1 | **Phase switch:** AdamW early, SGD+Momentum late (related to SWATS) | Fast early progress, better late generalization |
| H2 | **Blended step:** `Δ = α·Δ_adam + (1−α)·Δ_sgdm`, α scheduled or adapted from gradient statistics | Smooth interpolation, one extra hyperparameter to ablate |
| H3 | **Late-stage SAM:** SAM only in final epochs | Flat-minima benefit at lower cost |
| H4 | **Layer-wise assignment:** AdamW on fusion head, SGD+M elsewhere | Different parameter groups, different optimizer needs |

## 10. Non-functional requirements

| Area | Requirement |
|------|-------------|
| Reproducibility | Every run records config, seed, git commit and library versions. One command reproduces the toy suite end to end on CPU. |
| Compute | Phase 2 must be feasible on a single free GPU session using cached features (documented estimate in the README). |
| Quality | Type hints, linting, unit tests for optimizers, fusion solver and metric code; CI runs tests on CPU. |
| Data governance | No raw dataset content, faces or downloaded videos committed. Respect each dataset's licence and access terms. |
| Privacy | Demo uploads are processed in memory or a temp directory and deleted; nothing is logged beyond scores. |
| Accessibility | Dashboard meets WCAG 2.1 AA contrast, keyboard navigation, and does not encode information by colour alone. |
| Licence | MIT (already in the repo). |

## 11. Success metrics

| Metric | Target |
|--------|--------|
| Baseline coverage | ≥ 6 optimizers × ≥ 3 seeds (target 5) completed under the same tuning budget |
| Statistical reporting | Every comparison in the README shows mean ± std and a CI or test result |
| Toy benchmark reproducibility | `make toy` runs on CPU without a GPU and regenerates the toy table and plots |
| Hybrid discipline | Design frozen before test evaluation; ablation table present |
| Syllabus mapping | Every unit (I–VI) and lab experiment 1–10 mapped to a repo component (§12) |
| Repo quality | Fresh clone → README → toy results in under 10 minutes of the reader's time |
| Honesty | Limitations section present; null results reported as findings |

A positive result for the hybrid is **not** a success criterion. A well-supported null result is.

## 12. Syllabus mapping

| Syllabus item | Where it appears |
|---------------|------------------|
| **CLO 1** explain optimization problems · Unit I | README problem formulation: detector training as empirical-risk minimization |
| **CLO 2** unconstrained + constrained techniques · Units III–IV | Optimizer baselines; simplex-constrained fusion via Lagrange/KKT |
| **CLO 3** implement gradient-based and heuristic algorithms · Units III, VI | From-scratch SGD/Momentum/NAG/RMSProp/Adam; PSO and GA tuners |
| **CLO 4** analyse convergence and performance · Unit III | Convergence plots, steps-to-tolerance, generalization gap, statistical tests |
| Unit II: loss functions, saddle points, plateaus, flat minima, generalization | BCE loss; SAM and flat-minima discussion; cross-dataset gap |
| Unit III: GD variants, momentum, NAG, clipping, second order, landscapes, implicit bias | Optimizer set; gradient clipping ablation; Newton/L-BFGS on toys; contour plots |
| Unit IV: Lagrange, KKT, penalty/barrier | Fusion weights on the simplex; optional penalty-method comparison |
| Unit V: convex sets/functions, duality, Slater, SVM | QP fusion; SVM head and dual |
| Unit VI: GA, SA, PSO, hyperparameter tuning | GA/PSO tuners vs. random search |
| Lab 1–2 (GD, logistic regression) | Toy suite and baseline head training |
| Lab 3–4 (Lagrangian, QP) | Fusion solver |
| Lab 5 (SVM) | SVM fusion head |
| Lab 6 (parameter tuning) | Tuner module |
| Lab 7 (SGD, Adam, RMSProp) | Baseline comparison |
| Lab 8 (early stopping, checkpointing) | Trainer |
| Lab 9–10 (GA feature selection, PSO) | GA over modality/feature subsets; PSO on non-convex toys and on hyperparameters |

## 13. Milestones

Dates are relative; set absolute dates once the submission deadline is known.

| Week | Milestone |
|------|-----------|
| 1 | Phase 0 complete; data access requested; toy benchmark code started |
| 2 | Phase 1 complete (toy suite, plots, unit tests) |
| 3–4 | Phase 2 tuning and seeds running; fusion solver done |
| 5 | Gate 1: written baseline analysis |
| 6 | Hybrid design frozen (Gate 2) |
| 7 | Hybrid evaluation and ablations |
| 8 | Dashboard, README, report, polish |

## 14. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dataset access delayed (request forms) | Blocks Phase 0 | Request FF++, Celeb-DF and FakeAVCeleb access on day 1; develop on a small public sample or synthetic features meanwhile |
| No audio in FF++ / Celeb-DF | "Multimodal" claim weakened | Primary multimodal config on FakeAVCeleb; fallback visual + frequency streams, described honestly as multi-stream |
| Optimizer differences are within seed noise | Weak headline result | Pre-register that null results are reported; more seeds on cheap heads; focus analysis on convergence and generalization gap |
| Frozen-backbone findings do not transfer to full fine-tuning | Narrow conclusions | Limited unfreeze-last-N ablation; state the limitation in the README |
| Compute too small for all optimizers | Incomplete grid | Cached features; cut optimizers before cutting seeds or tuning budget |
| Tuning-budget unfairness | Invalid comparison | One shared tuner; budget enforced in code and recorded in `tuning_trials` |
| Test-set leakage while designing the hybrid | Invalid hybrid claim | Test rows written once; design frozen at Gate 2; selection on validation only |
| Misuse concerns | Reputational | Detection only; no generation code; dataset terms respected |

## 15. Open questions

1. Submission deadline and demo format required by the instructor?
2. Must the assignment be mathematically focused, or is an applied ML project acceptable? (Plan assumes the maths-first framing above is safe either way.)
3. Individual or group work?
4. GPU availability (Colab free / Kaggle / lab machines) and typical session limits?
5. Status of dataset access requests, especially FakeAVCeleb for audio+video?
6. Should the report be a PDF in the repo, or is the README enough?

## 16. References (verify before citing in the report)

- Nesterov (1983), accelerated gradient method.
- Kingma & Ba (2015), Adam.
- Loshchilov & Hutter (2019), AdamW.
- Liu et al. (2020), RAdam.
- Zhuang et al. (2020), AdaBelief.
- Chen et al. (2023), Lion.
- Foret et al. (2021), Sharpness-Aware Minimization.
- Keskar & Socher (2017), SWATS.
- Zhang et al. (2019), Lookahead.
- Rössler et al. (2019), FaceForensics++.
- Li et al. (2020), Celeb-DF.
- Khalid et al. (2021), FakeAVCeleb.
- Course texts: Chong & Zak; Boyd & Vandenberghe; Nocedal & Wright; Deisenroth et al.
