"""Phase 2 orchestration: tune -> train -> fuse -> evaluate, all recorded in the results DB.

Every step is resumable and idempotent on its natural key:
  tuning trial  (experiment, optimizer, model, method, trial_index)
  run           (experiment, optimizer, model, seed); a completed or diverged run is a no-op
  test rows     written once per run; a rewrite needs force=True and is flagged (TRD 8.2)
Selection, early stopping and fusion only ever read validation data.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from functools import cached_property
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optlab.config import ExperimentConfig, ModelConfig, OptimizerConfig
from optlab.db.repository import (
    TestSplitAlreadyWritten,
    git_commit,
    id_of,
    insert,
    insert_many,
    upsert_experiment,
    write_evaluation,
)
from optlab.eval.metrics import detection_metrics, eer_and_threshold, video_scores
from optlab.fusion import kkt_closed_form, projected_gradient, qp, temperature
from optlab.models.heads import LateFusion, modalities
from optlab.train.data import Split, load_split
from optlab.train.trainer import build_model, train
from optlab.tune.search import search

KKT_TOL = 1e-6  # a fusion solution is accepted only below this KKT residual (TRD 9.2)
SOLVERS = {
    "qp_cvxpy": qp,
    "kkt_closed_form": kkt_closed_form,
    "projected_gradient": projected_gradient,
}
LIBRARIES = [
    "torch",
    "numpy",
    "scipy",
    "scikit-learn",
    "cvxpy",
    "lion-pytorch",
    "adabelief-pytorch",
]


def library_versions() -> str:
    versions = {"python": sys.version.split()[0]}
    for lib in LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return json.dumps(versions, sort_keys=True)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


class Experiment:
    def __init__(
        self,
        conn: sqlite3.Connection,
        cfg: ExperimentConfig,
        config_path: Path,
        optimizers: dict[str, OptimizerConfig],
        models: dict[str, ModelConfig],
        log: Any = print,
    ) -> None:
        self.conn, self.cfg, self.config_path, self.log = conn, cfg, config_path, log
        self.optimizers = optimizers
        self.model_cfg = models[cfg.model]
        if self.model_cfg.fusion_type != "weighted_sum":
            raise NotImplementedError(
                f"fusion_type {self.model_cfg.fusion_type!r}: only weighted_sum is implemented"
            )
        self.mods = modalities(self.model_cfg)
        self.model_extra = {
            "head_hidden": self.model_cfg.head_hidden,
            "dropout": self.model_cfg.dropout,
        }
        self.exp_id = upsert_experiment(
            conn, cfg.name, cfg.phase, cfg.description, cfg.hypothesis, git_commit()
        )
        self.model_id = id_of(conn, "model_configs", cfg.model)
        self.train_ds_id = id_of(conn, "datasets", cfg.train_dataset)
        conn.commit()

    # --- data ----------------------------------------------------------------------------

    def _caches(self, dataset: str) -> dict[str, Path]:
        backbone = {
            "visual": self.model_cfg.visual_backbone,
            "audio": self.model_cfg.audio_backbone,
            "frequency": "freq_spectrum",
        }
        return {m: Path(self.cfg.features_dir) / dataset / backbone[m] for m in self.mods}

    def _split(self, dataset: str, split: str, val_role: str | None = None) -> Split:
        manifest = Path(self.cfg.manifests_dir) / f"{dataset}.csv"
        return load_split(manifest, split, self._caches(dataset), val_role)

    @cached_property
    def train_split(self) -> Split:
        return self._split(self.cfg.train_dataset, "train")

    @cached_property
    def val_tune(self) -> Split:
        return self._split(self.cfg.train_dataset, "val", "tune")

    @cached_property
    def val_fusion(self) -> Split:
        return self._split(self.cfg.train_dataset, "val", "fusion")

    # --- tuning --------------------------------------------------------------------------

    def tune(self, optimizer: str) -> sqlite3.Row:
        """Equal-budget search on val_tune; returns the selected tuning_trials row."""
        t = self.cfg.tuning
        opt_cfg = self.optimizers[optimizer]
        opt_id = id_of(self.conn, "optimizers", optimizer)
        key = (self.exp_id, opt_id, self.model_id, t.method)

        def objective(index: int, hp: dict[str, Any]) -> float:
            done = self.conn.execute(
                "SELECT val_auc, diverged FROM tuning_trials WHERE experiment_id = ?"
                " AND optimizer_id = ? AND model_config_id = ? AND search_method = ?"
                " AND trial_index = ?",
                (*key, index),
            ).fetchone()
            if done is not None:  # resume: deterministic proposals, reuse the recorded score
                return -np.inf if done["diverged"] else done["val_auc"]
            res = train(
                self.train_split,
                self.val_tune,
                opt_cfg,
                hp,
                t.seed,
                t.epochs,
                self.cfg.training.batch_size,
                self.cfg.training.patience,
                self.cfg.training.clip_norm,
                self.model_extra,
            )
            diverged = res.status == "diverged"
            insert(
                self.conn,
                "tuning_trials",
                {
                    "experiment_id": self.exp_id,
                    "optimizer_id": opt_id,
                    "model_config_id": self.model_id,
                    "search_method": t.method,
                    "trial_index": index,
                    "seed": t.seed,
                    "hyperparams_json": json.dumps(hp, sort_keys=True),
                    "epochs_run": len(res.epochs),
                    "val_auc": None if diverged else res.best_val_auc,
                    "val_loss": None if diverged else res.best_val_loss,
                    "diverged": int(diverged),
                },
            )
            self.conn.commit()
            self.log(
                f"tune {optimizer} trial {index}: {'diverged' if diverged else res.best_val_auc}"
            )
            return -np.inf if diverged else res.best_val_auc

        search(t.method, opt_cfg.search_space, objective, t.trials, t.seed)
        trials = self.conn.execute(
            "SELECT * FROM tuning_trials WHERE experiment_id = ? AND optimizer_id = ?"
            " AND model_config_id = ? AND search_method = ?",
            key,
        ).fetchall()
        valid = [r for r in trials if not r["diverged"]]
        if not valid:
            raise RuntimeError(f"every tuning trial of {optimizer} diverged")
        best = max(valid, key=lambda r: (r["val_auc"], -r["trial_index"]))
        self.conn.execute(
            "UPDATE tuning_trials SET is_selected = (id = ?) WHERE experiment_id = ?"
            " AND optimizer_id = ? AND model_config_id = ? AND search_method = ?",
            (best["id"], *key),
        )
        self.conn.commit()
        return best

    # --- training ------------------------------------------------------------------------

    def run_row(self, optimizer: str, seed: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? AND optimizer_id = ?"
            " AND model_config_id = ? AND seed = ?",
            (self.exp_id, id_of(self.conn, "optimizers", optimizer), self.model_id, seed),
        ).fetchone()

    def checkpoint_path(self, optimizer: str, seed: int) -> Path:
        return Path(self.cfg.artifacts_dir) / self.cfg.name / optimizer / f"seed{seed}.pt"

    def train_run(self, optimizer: str, seed: int, trial: sqlite3.Row) -> int:
        existing = self.run_row(optimizer, seed)
        if existing is not None and existing["status"] in ("completed", "diverged"):
            return existing["id"]  # idempotent (FR-O5)
        if existing is not None:  # queued / running / failed: start over
            self.conn.execute("DELETE FROM runs WHERE id = ?", (existing["id"],))
        tr = self.cfg.training
        hp = json.loads(trial["hyperparams_json"])
        run_id = insert(
            self.conn,
            "runs",
            {
                "experiment_id": self.exp_id,
                "optimizer_id": id_of(self.conn, "optimizers", optimizer),
                "model_config_id": self.model_id,
                "train_dataset_id": self.train_ds_id,
                "tuning_trial_id": trial["id"],
                "seed": seed,
                "status": "running",
                "hyperparams_json": trial["hyperparams_json"],
                "max_epochs": tr.max_epochs,
                "grad_clip_norm": tr.clip_norm,
                "started_at": _now(),
                "git_commit": git_commit(),
                "config_path": self.config_path.as_posix(),
                "library_versions_json": library_versions(),
            },
        )
        self.conn.commit()

        def on_epoch(e) -> None:
            insert_many(
                self.conn,
                "run_epochs",
                [
                    "run_id",
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_auc",
                    "lr",
                    "grad_norm",
                    "step_time_sec",
                ],
                [
                    (
                        run_id,
                        e.epoch,
                        e.train_loss,
                        e.val_loss,
                        e.val_auc,
                        e.lr,
                        e.grad_norm,
                        e.step_time_sec,
                    )
                ],
            )
            self.conn.commit()

        try:
            res = train(
                self.train_split,
                self.val_tune,
                self.optimizers[optimizer],
                hp,
                seed,
                tr.max_epochs,
                tr.batch_size,
                tr.patience,
                tr.clip_norm,
                self.model_extra,
                on_epoch=on_epoch,
            )
        except Exception:
            self.conn.execute(
                "UPDATE runs SET status = 'failed', finished_at = ? WHERE id = ?", (_now(), run_id)
            )
            self.conn.commit()
            raise
        if res.status == "completed":
            path = self.checkpoint_path(optimizer, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            dims = {m: x.shape[1] for m, x in self.train_split.x.items()}
            torch.save({"state_dict": res.state_dict, "dims": dims, "epoch": res.best_epoch}, path)
            insert(
                self.conn,
                "artifacts",
                {
                    "run_id": run_id,
                    "kind": "checkpoint",
                    "path": path.as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            )
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        self.conn.execute(
            "UPDATE runs SET status = ?, early_stopped = ?, best_epoch = ?, finished_at = ?,"
            " wall_clock_sec = ?, peak_gpu_mem_mb = ? WHERE id = ?",
            (
                res.status,
                int(res.early_stopped),
                res.best_epoch,
                _now(),
                res.wall_clock_sec,
                peak,
                run_id,
            ),
        )
        self.conn.commit()
        self.log(f"train {optimizer} seed {seed}: {res.status}, best val AUC {res.best_val_auc}")
        return run_id

    # --- fusion and evaluation -----------------------------------------------------------

    def _load_model(self, run_id: int) -> tuple[LateFusion, int]:
        row = self.conn.execute(
            "SELECT path FROM artifacts WHERE run_id = ? AND kind = 'checkpoint'", (run_id,)
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"run {run_id} has no checkpoint (not completed?)")
        ckpt = torch.load(row["path"], weights_only=True)
        model = build_model(ckpt["dims"], self.model_extra)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt["epoch"]

    @torch.no_grad()
    def _logits(self, model: LateFusion, split: Split) -> np.ndarray:
        return model(split.x).numpy().astype(np.float64)

    def fuse(self, run_id: int, solver: str = "qp_cvxpy") -> None:
        """Fit fusion weights on val_fusion; store weights, contributions and KKT diagnostics.

        `cfg.fusion` picks the formulation: plain simplex (TRD 9) or simplex with a learned
        temperature (separates logit scale from mixing; Gate 1 finding).
        """
        model, _ = self._load_model(run_id)
        z = self._logits(model, self.val_fusion)
        y_pm = self.val_fusion.y.numpy().astype(np.float64) * 2 - 1
        if self.cfg.fusion == "simplex_temperature":
            t = temperature.solve(z, y_pm, lam=self.cfg.fusion_ridge, tol=KKT_TOL)
            self._store_fusion(
                run_id,
                t.solver,
                "simplex_temperature",
                t.weights,
                t.contribution,
                t.objective,
                t.kkt_residual,
                None,
                t.temperature,
                t.active_set,
            )
            return
        sol = SOLVERS[solver].solve(z, y_pm)
        if sol.kkt_residual > KKT_TOL and solver != "projected_gradient":
            # Very confident logits push the conic solver to its precision limit; projected
            # gradient (with a Newton polish) on the same convex problem converges to ~1e-11.
            # The stored `solver` column records which method produced the solution.
            self.log(
                f"fuse run {run_id}: {solver} KKT residual {sol.kkt_residual:.1e} > {KKT_TOL:.0e};"
                " using projected_gradient"
            )
            sol = projected_gradient.solve(z, y_pm)
        if sol.kkt_residual > KKT_TOL:
            raise RuntimeError(f"fusion for run {run_id} did not converge: {sol.kkt_residual:.1e}")
        spread = sol.weights * z.std(axis=0)
        contribution = spread / spread.sum() if spread.sum() > 0 else sol.weights
        self._store_fusion(
            run_id,
            sol.solver,
            "simplex",
            sol.weights,
            contribution,
            sol.objective,
            sol.kkt_residual,
            sol.multiplier,
            None,
            sol.active_set,
        )

    def _store_fusion(
        self,
        run_id: int,
        solver: str,
        formulation: str,
        weights: np.ndarray,
        contribution: np.ndarray,
        objective: float,
        residual: float,
        multiplier: float | None,
        temp: float | None,
        active: list[int],
    ) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM fusion_weights WHERE run_id = ?", (run_id,))
            self.conn.execute("DELETE FROM fusion_solves WHERE run_id = ?", (run_id,))
            insert_many(
                self.conn,
                "fusion_weights",
                ["run_id", "modality", "weight", "contribution"],
                [
                    (run_id, m, float(w), float(np.clip(c, 0.0, 1.0)))
                    for m, w, c in zip(self.mods, weights, contribution, strict=True)
                ],
            )
            insert(
                self.conn,
                "fusion_solves",
                {
                    "run_id": run_id,
                    "solver": solver,
                    "formulation": formulation,
                    "objective_value": objective,
                    "kkt_residual": residual,
                    "lagrange_multiplier": multiplier,
                    "temperature": temp,
                    "active_set_json": json.dumps([self.mods[i] for i in active]),
                },
            )

    def _weights(self, run_id: int) -> np.ndarray:
        rows = {
            r["modality"]: r["weight"]
            for r in self.conn.execute(
                "SELECT modality, weight FROM fusion_weights WHERE run_id = ?", (run_id,)
            )
        }
        if set(rows) != set(self.mods):
            raise RuntimeError(f"run {run_id} has no fusion weights; run `optlab fuse` first")
        return np.array([rows[m] for m in self.mods])

    def has_test_rows(self, run_id: int) -> bool:
        return bool(
            self.conn.execute(
                "SELECT 1 FROM evaluations WHERE run_id = ? AND split = 'test'", (run_id,)
            ).fetchone()
        )

    def evaluate(self, run_id: int, final: bool = False, force: bool = False) -> None:
        """Validation rows always; with `final`, test rows for in-distribution and every
        cross-dataset set. Accuracy/F1 thresholds come from val_tune (EER point)."""
        if final and not force and self.has_test_rows(run_id):
            raise TestSplitAlreadyWritten(
                f"run {run_id} already has test evaluations; "
                "pass --force to overwrite and flag them"
            )
        model, epoch = self._load_model(run_id)
        w = self._weights(run_id)

        def scores(split: Split) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            frame = self._logits(model, split) @ w
            labels = split.y.numpy()
            vs, vl, _ = video_scores(frame, labels, split.video_ids)
            return {"frame": (frame, labels), "video": (vs, vl)}

        val = scores(self.val_tune)
        thresholds = {g: eer_and_threshold(s, y)[1] for g, (s, y) in val.items()}
        targets = [(self.cfg.train_dataset, "val", "in_distribution", val)]
        if final:
            targets.append(
                (
                    self.cfg.train_dataset,
                    "test",
                    "in_distribution",
                    scores(self._split(self.cfg.train_dataset, "test")),
                )
            )
            for ds in self.cfg.eval.cross_dataset:
                targets.append((ds, "test", "cross_dataset", scores(self._split(ds, "test"))))
        with self.conn:
            for ds, split, protocol, by_gran in targets:
                for gran, (s, y) in by_gran.items():
                    write_evaluation(
                        self.conn,
                        {
                            "run_id": run_id,
                            "eval_dataset_id": id_of(self.conn, "datasets", ds),
                            "split": split,
                            "protocol": protocol,
                            "granularity": gran,
                            "checkpoint_epoch": epoch,
                            "n_samples": len(y),
                            **detection_metrics(s, y, thresholds[gran]),
                        },
                        force=force,
                    )

    # --- sweep ---------------------------------------------------------------------------

    def sweep(self) -> None:
        """Everything for every optimizer and seed; safe to rerun after an interruption."""
        for optimizer in self.cfg.optimizers:
            trial = self.tune(optimizer)
            for seed in self.cfg.seeds:
                run_id = self.train_run(optimizer, seed, trial)
                if (
                    self.conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()[
                        0
                    ]
                    != "completed"
                ):
                    continue  # diverged runs stay in the DB, without fusion or evaluation
                if not self.conn.execute(
                    "SELECT 1 FROM fusion_solves WHERE run_id = ?", (run_id,)
                ).fetchone():
                    self.fuse(run_id)
                if not self.has_test_rows(run_id):
                    self.evaluate(run_id, final=True)
