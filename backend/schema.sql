-- ============================================================================
-- optlab results store  |  schema v1
-- Target: SQLite 3.35+ (zero-setup, one committed .db file per release).
-- Portable to PostgreSQL with: INTEGER PRIMARY KEY -> BIGSERIAL, TEXT JSON -> JSONB,
-- datetime('now') -> now(), 0/1 flags -> BOOLEAN.
--
-- Conventions
--   * *_json columns hold JSON text (validated in the application layer).
--   * Std-dev is NOT computed in SQL (SQLite math functions are optional);
--     views expose mean + sample variance, the API layer takes the square root.
--   * Test-split evaluations are written once per run (see evaluations).
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

-- ---------------------------------------------------------------------------
-- Datasets
-- ---------------------------------------------------------------------------
CREATE TABLE datasets (
    id               INTEGER PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,   -- 'faceforensics_pp' | 'celeb_df_v2' | 'fakeavceleb'
    has_audio        INTEGER NOT NULL DEFAULT 0 CHECK (has_audio IN (0, 1)),
    num_real_videos  INTEGER,
    num_fake_videos  INTEGER,
    compression      TEXT,                      -- e.g. 'c23' for FaceForensics++
    access_notes     TEXT,                      -- how access was obtained; never store credentials
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE dataset_splits (
    id             INTEGER PRIMARY KEY,
    dataset_id     INTEGER NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    split          TEXT    NOT NULL CHECK (split IN ('train', 'val', 'test')),
    manifest_path  TEXT    NOT NULL,            -- CSV of video ids + labels, committed to the repo
    num_videos     INTEGER,
    split_seed     INTEGER,
    identity_disjoint INTEGER NOT NULL DEFAULT 1 CHECK (identity_disjoint IN (0, 1)),
    UNIQUE (dataset_id, split)
);

-- ---------------------------------------------------------------------------
-- Model and optimizer catalogues
-- ---------------------------------------------------------------------------
CREATE TABLE model_configs (
    id                    INTEGER PRIMARY KEY,
    name                  TEXT    NOT NULL UNIQUE,
    visual_backbone       TEXT    NOT NULL,                 -- frozen feature extractor id
    audio_backbone        TEXT,                             -- NULL when no audio branch
    uses_frequency_branch INTEGER NOT NULL DEFAULT 0 CHECK (uses_frequency_branch IN (0, 1)),
    fusion_type           TEXT    NOT NULL
                          CHECK (fusion_type IN ('concat_mlp', 'weighted_sum', 'svm_head')),
    backbone_frozen       INTEGER NOT NULL DEFAULT 1 CHECK (backbone_frozen IN (0, 1)),
    unfrozen_last_n_layers INTEGER NOT NULL DEFAULT 0 CHECK (unfrozen_last_n_layers >= 0),
    config_json           TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE optimizers (
    id                   INTEGER PRIMARY KEY,
    name                 TEXT    NOT NULL UNIQUE,           -- 'sgd', 'sgd_momentum', 'nag', 'adamw', 'sam_sgd', 'hybrid_h1', ...
    family               TEXT    NOT NULL
                         CHECK (family IN ('plain', 'momentum', 'adaptive', 'sign',
                                           'sharpness_aware', 'second_order', 'hybrid')),
    in_syllabus          INTEGER NOT NULL DEFAULT 0 CHECK (in_syllabus IN (0, 1)),
    is_hybrid            INTEGER NOT NULL DEFAULT 0 CHECK (is_hybrid IN (0, 1)),
    parent_optimizer_id  INTEGER REFERENCES optimizers (id), -- set for ablations of a hybrid
    ablation_note        TEXT,                               -- which component was removed
    update_rule          TEXT,                               -- LaTeX / plain-text update rule
    reference            TEXT,                               -- paper citation
    default_search_space_json TEXT NOT NULL DEFAULT '{}',    -- the tuning space used for ALL fair-budget searches
    CHECK (parent_optimizer_id IS NULL OR is_hybrid = 1)
);

-- ---------------------------------------------------------------------------
-- Experiments, tuning, runs
-- ---------------------------------------------------------------------------
CREATE TABLE experiments (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    phase        TEXT NOT NULL CHECK (phase IN ('toy', 'baseline', 'hybrid', 'ablation')),
    description  TEXT,
    hypothesis   TEXT,                    -- written BEFORE results are inspected
    git_commit   TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per hyper-parameter trial. Every optimizer gets the same trial budget.
CREATE TABLE tuning_trials (
    id               INTEGER PRIMARY KEY,
    experiment_id    INTEGER NOT NULL REFERENCES experiments (id) ON DELETE CASCADE,
    optimizer_id     INTEGER NOT NULL REFERENCES optimizers (id),
    model_config_id  INTEGER NOT NULL REFERENCES model_configs (id),
    search_method    TEXT    NOT NULL CHECK (search_method IN ('grid', 'random', 'pso', 'ga')),
    trial_index      INTEGER NOT NULL CHECK (trial_index >= 0),
    seed             INTEGER NOT NULL,
    hyperparams_json TEXT    NOT NULL,
    epochs_run       INTEGER,
    val_auc          REAL CHECK (val_auc IS NULL OR (val_auc >= 0 AND val_auc <= 1)),
    val_loss         REAL,
    diverged         INTEGER NOT NULL DEFAULT 0 CHECK (diverged IN (0, 1)),
    is_selected      INTEGER NOT NULL DEFAULT 0 CHECK (is_selected IN (0, 1)),
    UNIQUE (experiment_id, optimizer_id, model_config_id, search_method, trial_index)
);

CREATE TABLE runs (
    id                 INTEGER PRIMARY KEY,
    experiment_id      INTEGER NOT NULL REFERENCES experiments (id) ON DELETE CASCADE,
    optimizer_id       INTEGER NOT NULL REFERENCES optimizers (id),
    model_config_id    INTEGER NOT NULL REFERENCES model_configs (id),
    train_dataset_id   INTEGER NOT NULL REFERENCES datasets (id),
    tuning_trial_id    INTEGER REFERENCES tuning_trials (id),   -- trial whose hyper-params this run reuses
    seed               INTEGER NOT NULL,
    status             TEXT    NOT NULL DEFAULT 'queued'
                       CHECK (status IN ('queued', 'running', 'completed', 'failed', 'diverged')),
    hyperparams_json   TEXT    NOT NULL,
    max_epochs         INTEGER NOT NULL CHECK (max_epochs > 0),
    grad_clip_norm     REAL,
    early_stopped      INTEGER NOT NULL DEFAULT 0 CHECK (early_stopped IN (0, 1)),
    best_epoch         INTEGER,
    started_at         TEXT,
    finished_at        TEXT,
    wall_clock_sec     REAL,
    peak_gpu_mem_mb    REAL,
    git_commit         TEXT,
    config_path        TEXT,
    UNIQUE (experiment_id, optimizer_id, model_config_id, seed)
);

CREATE TABLE run_epochs (
    run_id         INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    epoch          INTEGER NOT NULL CHECK (epoch >= 0),
    train_loss     REAL,
    val_loss       REAL,
    val_auc        REAL,
    lr             REAL,
    grad_norm      REAL,
    step_time_sec  REAL,
    PRIMARY KEY (run_id, epoch)
);

-- Final metrics. Rows with split = 'test' are written once per run by
-- `optlab evaluate --final`; the CLI refuses a second write unless --force is given.
CREATE TABLE evaluations (
    id                INTEGER PRIMARY KEY,
    run_id            INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    eval_dataset_id   INTEGER NOT NULL REFERENCES datasets (id),
    split             TEXT    NOT NULL CHECK (split IN ('val', 'test')),
    protocol          TEXT    NOT NULL CHECK (protocol IN ('in_distribution', 'cross_dataset')),
    granularity       TEXT    NOT NULL CHECK (granularity IN ('frame', 'video')),
    checkpoint_epoch  INTEGER,
    n_samples         INTEGER,
    auc               REAL CHECK (auc IS NULL OR (auc >= 0 AND auc <= 1)),
    eer               REAL CHECK (eer IS NULL OR (eer >= 0 AND eer <= 1)),
    accuracy          REAL,
    f1                REAL,
    average_precision REAL,
    force_rewritten   INTEGER NOT NULL DEFAULT 0 CHECK (force_rewritten IN (0, 1)),
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (run_id, eval_dataset_id, split, granularity)
);

-- Constrained fusion: weights live on the probability simplex (w >= 0, sum w = 1).
CREATE TABLE fusion_weights (
    run_id    INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    modality  TEXT    NOT NULL CHECK (modality IN ('visual', 'audio', 'frequency')),
    weight    REAL    NOT NULL CHECK (weight >= 0 AND weight <= 1),
    PRIMARY KEY (run_id, modality)
);

CREATE TABLE fusion_solves (
    run_id           INTEGER PRIMARY KEY REFERENCES runs (id) ON DELETE CASCADE,
    solver           TEXT NOT NULL CHECK (solver IN ('kkt_closed_form', 'qp_cvxpy', 'projected_gradient')),
    objective_value  REAL,
    kkt_residual     REAL,                     -- stationarity / complementarity violation at the solution
    lagrange_multiplier REAL,                  -- multiplier of the sum-to-one constraint
    active_set_json  TEXT                      -- modalities whose w >= 0 constraint is active
);

-- ---------------------------------------------------------------------------
-- Toy benchmarks (Phase 1: optimizer maths on known functions)
-- ---------------------------------------------------------------------------
CREATE TABLE toy_benchmarks (
    id                INTEGER PRIMARY KEY,
    name              TEXT    NOT NULL UNIQUE,  -- 'quadratic_ill_conditioned', 'rosenbrock', 'rastrigin', 'himmelblau'
    dimension         INTEGER NOT NULL CHECK (dimension >= 1),
    is_convex         INTEGER NOT NULL CHECK (is_convex IN (0, 1)),
    condition_number  REAL,                     -- for quadratics
    global_min_value  REAL    NOT NULL DEFAULT 0,
    description       TEXT
);

CREATE TABLE toy_runs (
    id                INTEGER PRIMARY KEY,
    benchmark_id      INTEGER NOT NULL REFERENCES toy_benchmarks (id) ON DELETE CASCADE,
    optimizer_id      INTEGER NOT NULL REFERENCES optimizers (id),
    seed              INTEGER NOT NULL,
    hyperparams_json  TEXT    NOT NULL,
    max_steps         INTEGER NOT NULL CHECK (max_steps > 0),
    tolerance         REAL    NOT NULL,         -- |f - f*| threshold defining "converged"
    steps_to_tol      INTEGER,                  -- NULL = never reached tolerance
    final_f           REAL,
    final_grad_norm   REAL,
    trajectory_path   TEXT,                     -- .npz with full (x_t) path for the 2-D viewer
    UNIQUE (benchmark_id, optimizer_id, seed)
);

CREATE TABLE toy_run_curves (
    toy_run_id   INTEGER NOT NULL REFERENCES toy_runs (id) ON DELETE CASCADE,
    step         INTEGER NOT NULL CHECK (step >= 0),
    f_value      REAL    NOT NULL,
    grad_norm    REAL,
    PRIMARY KEY (toy_run_id, step)
);

-- ---------------------------------------------------------------------------
-- Artifacts (checkpoints, plots, logs). Files live on disk / releases, not in the DB.
-- ---------------------------------------------------------------------------
CREATE TABLE artifacts (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER REFERENCES runs (id) ON DELETE CASCADE,
    toy_run_id  INTEGER REFERENCES toy_runs (id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('checkpoint', 'curve_png', 'config', 'log', 'feature_cache')),
    path        TEXT NOT NULL,
    sha256      TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (run_id IS NOT NULL OR toy_run_id IS NOT NULL)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_runs_experiment_opt   ON runs (experiment_id, optimizer_id);
CREATE INDEX idx_runs_status           ON runs (status);
CREATE INDEX idx_epochs_run            ON run_epochs (run_id);
CREATE INDEX idx_eval_run              ON evaluations (run_id);
CREATE INDEX idx_eval_dataset_split    ON evaluations (eval_dataset_id, split);
CREATE INDEX idx_trials_lookup         ON tuning_trials (experiment_id, optimizer_id, search_method);
CREATE INDEX idx_toy_runs_bench_opt    ON toy_runs (benchmark_id, optimizer_id);
CREATE INDEX idx_artifacts_run         ON artifacts (run_id);

-- ---------------------------------------------------------------------------
-- Views used by the API
-- ---------------------------------------------------------------------------

-- One row per run with the headline numbers.
CREATE VIEW v_run_summary AS
SELECT
    r.id                AS run_id,
    r.experiment_id,
    o.name              AS optimizer_name,
    o.family            AS optimizer_family,
    r.seed,
    r.status,
    r.best_epoch,
    r.early_stopped,
    r.wall_clock_sec,
    r.peak_gpu_mem_mb,
    (SELECT MAX(val_auc) FROM run_epochs re WHERE re.run_id = r.id) AS best_val_auc,
    (SELECT COUNT(*)     FROM run_epochs re WHERE re.run_id = r.id) AS epochs_logged
FROM runs r
JOIN optimizers o ON o.id = r.optimizer_id;

-- Per optimizer x eval dataset: mean and sample variance across seeds (completed runs only).
CREATE VIEW v_leaderboard AS
SELECT
    r.experiment_id,
    r.optimizer_id,
    o.name            AS optimizer_name,
    o.family          AS optimizer_family,
    e.eval_dataset_id,
    d.name            AS eval_dataset_name,
    e.split,
    e.protocol,
    e.granularity,
    COUNT(*)          AS n_seeds,
    AVG(e.auc)        AS auc_mean,
    CASE WHEN COUNT(*) > 1
         THEN (SUM(e.auc * e.auc) - SUM(e.auc) * SUM(e.auc) / COUNT(*)) / (COUNT(*) - 1)
    END               AS auc_var,
    AVG(e.eer)        AS eer_mean,
    CASE WHEN COUNT(*) > 1
         THEN (SUM(e.eer * e.eer) - SUM(e.eer) * SUM(e.eer) / COUNT(*)) / (COUNT(*) - 1)
    END               AS eer_var,
    AVG(e.accuracy)   AS accuracy_mean,
    AVG(e.f1)         AS f1_mean,
    AVG(r.wall_clock_sec) AS wall_clock_mean_sec
FROM evaluations e
JOIN runs       r ON r.id = e.run_id AND r.status = 'completed'
JOIN optimizers o ON o.id = r.optimizer_id
JOIN datasets   d ON d.id = e.eval_dataset_id
GROUP BY r.experiment_id, r.optimizer_id, e.eval_dataset_id, e.split, e.protocol, e.granularity;

-- Generalisation gap per run: in-distribution test AUC minus cross-dataset test AUC (video level).
CREATE VIEW v_generalisation_gap AS
SELECT
    r.id            AS run_id,
    r.experiment_id,
    r.optimizer_id,
    r.seed,
    ind.auc         AS auc_in_distribution,
    xd.auc          AS auc_cross_dataset,
    ind.auc - xd.auc AS auc_gap
FROM runs r
JOIN evaluations ind ON ind.run_id = r.id AND ind.split = 'test'
                    AND ind.protocol = 'in_distribution' AND ind.granularity = 'video'
JOIN evaluations xd  ON xd.run_id  = r.id AND xd.split  = 'test'
                    AND xd.protocol  = 'cross_dataset'   AND xd.granularity  = 'video'
WHERE r.status = 'completed';
