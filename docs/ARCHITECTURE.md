# System Architecture

> Diagrams are Mermaid and render natively on GitHub. Names match `backend/schema.sql` and `docs/TRD.md`.

## 1. System context

Heavy compute (feature extraction, training) runs offline on a free GPU (Colab / Kaggle). Everything it produces lands in one SQLite file plus a few artifact folders. The dashboard only reads that file, so it can run locally or be deployed as a static site from a JSON snapshot.

```mermaid
flowchart LR
    subgraph DATA["Datasets - access request required"]
        FF["FaceForensics++"]
        CD["Celeb-DF v2"]
        FA["FakeAVCeleb - audio+video"]
    end

    subgraph PREP["optlab.data - offline, once"]
        FACE["Face detect + crop"]
        FRAMES["Frame sampling - K per video"]
        AUD["Audio extraction"]
        FEAT["Frozen feature extractors<br/>visual, audio, frequency"]
        CACHE[("Feature cache<br/>.npy shards + manifest CSVs")]
    end

    subgraph CORE["optlab core - Python and PyTorch"]
        REG["Optimizer registry<br/>SGD, Momentum, NAG, RMSProp, Adam, AdamW,<br/>RAdam, AdaBelief, Lion, SAM, Hybrid"]
        HEAD["Trainable heads<br/>per-modality MLP + fusion"]
        FUSE["Constrained fusion solver<br/>KKT / QP on the simplex"]
        TUNE["Tuner<br/>random, grid, PSO, GA"]
        TRAIN["Trainer<br/>clip, early stop, checkpoint"]
        EVAL["Evaluator<br/>AUC, EER, ACC, F1, AP"]
        STATS["Stats<br/>bootstrap CI, Holm-corrected tests"]
        TOY["Toy benchmarks<br/>quadratic, Rosenbrock, Rastrigin"]
    end

    DB[("results.db - SQLite<br/>schema.sql")]
    ART[("artifacts/<br/>checkpoints, curves, configs")]

    subgraph SERVE["Presentation"]
        API["FastAPI - read-only"]
        SNAP["Static JSON snapshot"]
        WEB["React dashboard"]
    end

    CLI["optlab CLI"]

    FF --> FACE
    CD --> FACE
    FA --> FACE
    FA --> AUD
    FACE --> FRAMES --> FEAT
    AUD --> FEAT
    FEAT --> CACHE

    CLI --> TUNE
    CLI --> TRAIN
    CLI --> EVAL
    CLI --> TOY
    CACHE --> TRAIN
    REG --> TRAIN
    REG --> TOY
    REG --> TUNE
    HEAD --> TRAIN
    FUSE --> HEAD
    TUNE --> TRAIN
    TRAIN --> EVAL
    TRAIN --> DB
    EVAL --> DB
    TUNE --> DB
    TOY --> DB
    TRAIN --> ART
    DB --> STATS
    STATS --> DB

    DB --> API
    API --> WEB
    DB -. "optlab export-snapshot" .-> SNAP
    SNAP -.-> WEB
```

## 2. Experiment phases and gates

The hybrid optimizer is designed only after baseline analysis, and its design is frozen before any test-split number is looked at.

```mermaid
flowchart TD
    P0["Phase 0 - Setup<br/>repo, data access, feature cache, DB"]
    P1["Phase 1 - Toy benchmarks<br/>optimizer maths on known functions"]
    P2["Phase 2 - Baselines<br/>fair LR search, 5 seeds, in-dist + cross-dataset"]
    G1{{"Gate 1<br/>baseline analysis written:<br/>where is each optimizer weak?"}}
    P3["Phase 3 - Hybrid design<br/>update rule written, toy-tested,<br/>tuned on validation only"]
    G2{{"Gate 2<br/>design frozen<br/>before test evaluation"}}
    P4["Phase 4 - Hybrid evaluation + ablations<br/>same budget, same seeds"]
    P5["Phase 5 - Dashboard, README, report"]

    P0 --> P1 --> P2 --> G1 --> P3 --> G2 --> P4 --> P5
    P1 -. "informs update-rule design" .-> P3
```

## 3. One training run

```mermaid
sequenceDiagram
    autonumber
    participant CLI as optlab CLI
    participant CFG as Config - YAML
    participant DS as Feature dataset
    participant REG as Optimizer registry
    participant TR as Trainer
    participant DB as results.db
    participant EV as Evaluator

    CLI->>CFG: load experiment + optimizer + seed
    CLI->>DB: insert runs row - status running
    CLI->>DS: open cached features - train, val
    CLI->>REG: build optimizer from hyperparams
    loop each epoch
        TR->>DS: next mini-batch
        TR->>REG: step - clip, momentum, adaptive, SAM
        TR->>DB: append run_epochs - loss, val AUC, lr, grad norm
        TR->>TR: early stopping + best checkpoint
    end
    TR->>DB: runs.status completed - best_epoch, wall_clock
    CLI->>EV: evaluate best checkpoint
    EV->>DB: evaluations - val rows
    Note over EV,DB: test and cross-dataset rows are written once per run
    EV->>DB: evaluations - test rows - frame and video level
```

## 4. Data model

Simplified view of `backend/schema.sql`.

```mermaid
erDiagram
    datasets ||--o{ dataset_splits : has
    datasets ||--o{ runs : "trained on"
    datasets ||--o{ evaluations : "evaluated on"
    experiments ||--o{ runs : contains
    experiments ||--o{ tuning_trials : contains
    optimizers ||--o{ runs : used_by
    optimizers ||--o{ tuning_trials : tuned
    optimizers ||--o{ toy_runs : used_by
    optimizers ||--o| optimizers : "ablation of"
    model_configs ||--o{ runs : uses
    tuning_trials ||--o{ runs : "hyperparams reused by"
    runs ||--o{ run_epochs : logs
    runs ||--o{ evaluations : produces
    runs ||--o{ fusion_weights : learns
    runs ||--o| fusion_solves : solved_by
    runs ||--o{ artifacts : stores
    toy_benchmarks ||--o{ toy_runs : contains
    toy_runs ||--o{ toy_run_curves : logs
    toy_runs ||--o{ artifacts : stores
```

## 5. Runtime and deployment

```mermaid
flowchart LR
    subgraph GPU["Free GPU - Colab or Kaggle"]
        RUN["optlab CLI<br/>extract, tune, train, evaluate"]
    end
    subgraph REPO["GitHub repository"]
        CODE["code + configs + manifests"]
        DBF["results.db - release asset or Git LFS"]
        SNAPF["frontend/public/data/*.json"]
    end
    subgraph LOCAL["Laptop"]
        DEV["FastAPI + Vite dev server"]
    end
    subgraph HOST["Static hosting - Vercel or GitHub Pages"]
        SITE["Read-only dashboard"]
    end

    CODE --> RUN
    RUN --> DBF
    DBF --> DEV
    DBF -- "optlab export-snapshot" --> SNAPF
    SNAPF --> SITE
```

## 6. Design decisions

| Decision | Reason |
|---|---|
| Frozen backbones with cached features | Optimizer runs train only small heads, so 10 optimizers x 5 seeds x tuning fits on a free GPU. Trade-off: findings may not transfer to full fine-tuning, so Phase 4 includes an "unfreeze last N layers" ablation. |
| SQLite results store | One file, no server, easy to version and share. Postgres-portable. |
| Read-only dashboard | Results are produced offline; no auth or write path to secure. |
| Static JSON snapshot | Lets the dashboard be hosted for free and keeps the repo demo-able without a backend. |
| Test splits written once | Prevents tuning the hybrid optimizer against the benchmark it is judged on. |
| Std-dev computed in the API layer | SQLite math functions are optional; views expose variance instead. |
