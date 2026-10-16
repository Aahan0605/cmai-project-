"""Limited raw-CLIP ablation with cached frozen prefix tokens and a trainable final block.

Caching is strictly upstream of the unfrozen block. It never caches outputs from parameters
being optimized. The paired frozen control uses the same images, head, budget and seeds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import roc_auc_score
from torch import nn

from optlab.config import ModelConfig, load_optimizers
from optlab.data.faces import MTCNNDetector, crop_face
from optlab.data.manifest import read_manifest
from optlab.data.preprocess import read_sampled_frames
from optlab.db.connection import connect
from optlab.db.repository import (
    git_commit,
    id_of,
    insert,
    seed_catalogues,
    upsert_experiment,
    write_evaluation,
)
from optlab.eval.metrics import detection_metrics, eer_and_threshold
from optlab.fusion.calibration import fit
from optlab.optim.registry import build_from_config
from optlab.pipeline import library_versions
from optlab.train.checkpoint import save_checkpoint
from optlab.train.trainer import seed_everything


class FineTuneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str
    manifest: str
    video_root: str
    cache: str = "artifacts/finetune/prefix.pt"
    seeds: list[int] = [0, 1, 2]
    optimizers: list[str] = ["adamw", "hybrid_blend"]
    last_n: list[int] = [0, 1]
    epochs: int = Field(3, ge=1)
    batch_size: int = Field(16, ge=1)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    device: str = "cpu"


class TailClassifier(nn.Module):
    def __init__(self, blocks, ln_post, projection, thaw: bool, hidden=256):
        super().__init__()
        self.blocks = copy.deepcopy(blocks)
        self.ln_post = copy.deepcopy(ln_post)
        self.projection = nn.Parameter(projection.detach().clone(), requires_grad=thaw)
        self.blocks.requires_grad_(thaw)
        self.ln_post.requires_grad_(thaw)
        self.head = nn.Sequential(
            nn.Linear(projection.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1)
        )

    def encode(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return self.ln_post(tokens[:, 0]) @ self.projection

    def forward(self, tokens):
        return self.head(self.encode(tokens)).flatten()


def prefix_tokens(visual, images, last_n: int):
    """OpenCLIP ViT-B/16 uses batch-first residual blocks and CLS pooling."""
    if (
        not visual.transformer.batch_first
        or visual.pool_type != "tok"
        or visual.attn_pool is not None
    ):
        raise ValueError("this ablation supports the configured batch-first CLS-pooled CLIP ViT")
    tokens = visual._embeds(images)
    for block in visual.transformer.resblocks[:-last_n]:
        tokens = block(tokens)
    return tokens


def run_finetune(config_path: Path, db: Path, log=print):
    cfg = FineTuneConfig.model_validate(yaml.safe_load(config_path.read_text()))
    if set(cfg.last_n) != {0, 1} or not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("the limited paired ablation requires last_n=[0,1] and unique seeds")
    import open_clip
    from PIL import Image

    torch.set_num_threads(1)
    clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=cfg.device
    )
    visual = clip.visual.eval().requires_grad_(False)
    manifest = Path(cfg.manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache = Path(cfg.cache)
    if cache.exists():
        packed = torch.load(cache, weights_only=True, map_location="cpu")
        if (
            packed["manifest_sha256"] != digest
            or packed["open_clip_version"] != open_clip.__version__
        ):
            raise ValueError("prefix cache provenance changed; use a new cache path")
    else:
        detector = MTCNNDetector("cpu")
        tokens, labels, roles, ids = [], [], [], []
        rows = read_manifest(manifest)
        with torch.no_grad():
            for i, row in enumerate(rows):
                _, frames = read_sampled_frames(Path(cfg.video_root) / row.path, 1)
                boxes = detector.detect(frames) if frames else []
                if not boxes or boxes[0] is None:
                    continue
                crop = crop_face(frames[0], boxes[0], 0.3, 224)
                x = preprocess(Image.fromarray(crop)).unsqueeze(0).to(cfg.device)
                prefix = prefix_tokens(visual, x, 1)
                tokens.append(prefix.cpu().half())
                labels.append(row.label)
                roles.append(row.val_role if row.split == "val" else row.split)
                ids.append(row.video_id)
                if (i + 1) % 50 == 0:
                    log(f"prefix cache {i + 1}/{len(rows)}")
        packed = {
            "tokens": torch.cat(tokens),
            "labels": labels,
            "roles": roles,
            "video_ids": ids,
            "manifest_sha256": digest,
            "open_clip_version": open_clip.__version__,
            "pretrained": "ViT-B-16/openai",
            "frames_per_video": 1,
            "cached_before_last_blocks": 1,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(packed, cache)
    blocks = copy.deepcopy(visual.transformer.resblocks[-1:]).cpu()
    ln = copy.deepcopy(visual.ln_post).cpu()
    proj = visual.proj.detach().cpu().clone()
    del clip, visual
    if cfg.device.startswith("mps"):
        torch.mps.empty_cache()
    x = packed["tokens"]
    y = torch.tensor(packed["labels"], dtype=torch.float32)
    splits = {
        role: torch.tensor([i for i, r in enumerate(packed["roles"]) if r == role])
        for role in ("train", "tune", "fusion", "test")
    }
    if any(len(idx) == 0 or set(y[idx].tolist()) != {0, 1} for idx in splits.values()):
        raise ValueError("all raw ablation roles need both classes after face detection")
    conn = connect(db)
    opts = load_optimizers("configs")
    model_configs = [
        ModelConfig(
            name=f"clip_vit_b16__raw_last{n}",
            visual_backbone="clip_vit_b16",
            fusion_type="weighted_sum",
            backbone_frozen=n == 0,
            unfrozen_last_n_layers=n,
        )
        for n in cfg.last_n
    ]
    seed_catalogues(conn, [], model_configs, [], [])
    exp_id = upsert_experiment(
        conn,
        cfg.name,
        "ablation",
        (
            "Paired raw CLIP control vs final block fine-tuning. One frame/video; "
            "fixed common LR, three epochs, no hyperparameter search. Exploratory TIMIT only."
        ),
        (
            "Final-block fine-tuning changes detection relative to a matched frozen control; "
            "superiority is unknown."
        ),
        git_commit(),
    )
    ds_id = id_of(conn, "datasets", "optlab_timit_hq")
    conn.commit()
    for n in cfg.last_n:
        model_id = id_of(conn, "model_configs", f"clip_vit_b16__raw_last{n}")
        for name in cfg.optimizers:
            for seed in cfg.seeds:
                existing = conn.execute(
                    "SELECT id,status FROM runs WHERE experiment_id=? AND model_config_id=?"
                    " AND optimizer_id=? AND seed=?",
                    (exp_id, model_id, id_of(conn, "optimizers", name), seed),
                ).fetchone()
                if existing and existing["status"] == "completed":
                    continue
                if existing:
                    conn.execute("DELETE FROM runs WHERE id=?", (existing["id"],))
                seed_everything(seed)
                model = TailClassifier(blocks, ln, proj, thaw=n > 0).to(cfg.device)
                parameters = [p for p in model.parameters() if p.requires_grad]
                hp = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
                optimizer = build_from_config(opts[name], parameters, hp)
                run = insert(
                    conn,
                    "runs",
                    {
                        "experiment_id": exp_id,
                        "optimizer_id": id_of(conn, "optimizers", name),
                        "model_config_id": model_id,
                        "train_dataset_id": ds_id,
                        "seed": seed,
                        "status": "running",
                        "hyperparams_json": json.dumps(hp),
                        "max_epochs": cfg.epochs,
                        "config_path": str(config_path),
                        "git_commit": git_commit(),
                        "library_versions_json": library_versions(),
                    },
                )
                conn.commit()
                start = time.perf_counter()
                initial = {
                    k: v.detach().cpu().clone() for k, v in model.blocks.state_dict().items()
                }
                generator = torch.Generator().manual_seed(seed)
                train_idx = splits["train"]
                weights = torch.where(
                    y[train_idx] == 1,
                    1 / y[train_idx].sum(),
                    1 / (len(train_idx) - y[train_idx].sum()),
                )
                best_auc = -1.0
                best_state = None
                best_epoch = 0

                def scores(indices, model=model):
                    model.eval()
                    with torch.no_grad():
                        return torch.cat(
                            [
                                model(x[idx].to(cfg.device, dtype=torch.float32)).cpu()
                                for idx in indices.split(cfg.batch_size)
                            ]
                        ).numpy()

                try:
                    for epoch in range(cfg.epochs):
                        model.train()
                        order = train_idx[
                            torch.multinomial(
                                weights, len(train_idx), replacement=True, generator=generator
                            )
                        ]
                        losses = []
                        norms = []
                        times = []
                        for idx in order.split(cfg.batch_size):
                            xb = x[idx].to(cfg.device, dtype=torch.float32)
                            yb = y[idx].to(cfg.device)

                            def closure(
                                xb=xb,
                                yb=yb,
                                optimizer=optimizer,
                                model=model,
                                norms=norms,
                                parameters=parameters,
                            ):
                                optimizer.zero_grad()
                                loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
                                loss.backward()
                                norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, 1.0)))
                                return loss

                            before = time.perf_counter()
                            loss = optimizer.step(closure)
                            if not torch.isfinite(loss):
                                raise ValueError("fine-tuning diverged")
                            losses.append(float(loss.detach()))
                            times.append(time.perf_counter() - before)
                        val = scores(splits["tune"])
                        auc = float(roc_auc_score(y[splits["tune"]], val))
                        val_loss = float(
                            nn.functional.binary_cross_entropy_with_logits(
                                torch.tensor(val), y[splits["tune"]]
                            )
                        )
                        insert(
                            conn,
                            "run_epochs",
                            {
                                "run_id": run,
                                "epoch": epoch,
                                "train_loss": float(np.mean(losses)),
                                "val_loss": val_loss,
                                "val_auc": auc,
                                "lr": cfg.lr,
                                "grad_norm": float(np.mean(norms)),
                                "step_time_sec": float(np.mean(times)),
                            },
                        )
                        conn.commit()
                        if auc > best_auc:
                            best_auc = auc
                            best_epoch = epoch
                            best_state = {
                                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            }
                    model.load_state_dict(best_state)
                    changed = any(
                        not torch.equal(v.cpu(), initial[k])
                        for k, v in model.blocks.state_dict().items()
                    )
                    if n == 0 and changed:
                        raise RuntimeError("frozen control backbone changed")
                    path = Path("artifacts") / cfg.name / name / f"last{n}_seed{seed}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(
                        {
                            "state_dict": best_state,
                            "last_n": n,
                            "backbone_changed": changed,
                            "prefix_manifest_sha256": digest,
                        },
                        path,
                    )
                    tune = scores(splits["tune"])
                    fusion = scores(splits["fusion"])
                    test = scores(splits["test"])
                    calibration = fit(tune[:, None], y[splits["tune"]].numpy())
                    calibration["modalities"] = ["visual"]
                    scale = calibration["scales"][0]
                    threshold = eer_and_threshold(tune * scale, y[splits["tune"]].numpy())[1]
                    with conn:
                        insert(
                            conn,
                            "artifacts",
                            {
                                "run_id": run,
                                "kind": "checkpoint",
                                "path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            },
                        )
                        insert(
                            conn,
                            "fusion_weights",
                            {
                                "run_id": run,
                                "modality": "visual",
                                "weight": 1.0,
                                "contribution": 1.0,
                            },
                        )
                        insert(
                            conn,
                            "fusion_solves",
                            {
                                "run_id": run,
                                "solver": "projected_gradient",
                                "objective_value": float(
                                    np.mean(
                                        np.logaddexp(0, fusion * scale)
                                        - y[splits["fusion"]].numpy() * fusion * scale
                                    )
                                ),
                                "kkt_residual": 0.0,
                                "formulation": "simplex",
                                "active_set_json": "[]",
                                "calibration_json": json.dumps(calibration),
                            },
                        )
                        for role, values in [("val", tune), ("test", test)]:
                            labels = y[splits["tune" if role == "val" else "test"]].numpy()
                            for granularity in ("frame", "video"):
                                write_evaluation(
                                    conn,
                                    {
                                        "run_id": run,
                                        "eval_dataset_id": ds_id,
                                        "split": role,
                                        "protocol": "in_distribution",
                                        "granularity": granularity,
                                        "checkpoint_epoch": best_epoch,
                                        "n_samples": len(labels),
                                        **detection_metrics(values * scale, labels, threshold),
                                    },
                                )
                        conn.execute(
                            "UPDATE runs SET status='completed',best_epoch=?,wall_clock_sec=?,"
                            "finished_at=datetime('now') WHERE id=?",
                            (best_epoch, time.perf_counter() - start, run),
                        )
                    log(
                        f"finetune last{n} {name} seed{seed}: val AUC {best_auc:.4f}, "
                        f"backbone_changed={changed}"
                    )
                except BaseException:
                    conn.execute("UPDATE runs SET status='failed' WHERE id=?", (run,))
                    conn.commit()
                    raise
                finally:
                    del model, optimizer
                    if cfg.device.startswith("mps"):
                        torch.mps.empty_cache()
    conn.close()
