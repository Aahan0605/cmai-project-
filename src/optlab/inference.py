"""Portable local inference bundles; input media is never added to the results database."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from optlab import config
from optlab.db.connection import connect
from optlab.eval.metrics import eer_and_threshold, video_scores
from optlab.fusion.calibration import apply
from optlab.pipeline import Experiment
from optlab.train.checkpoint import save_checkpoint
from optlab.train.trainer import build_model


def export_model(db: Path, run_id: int, out: Path, configs_dir=Path("configs")) -> dict:
    conn = connect(db, readonly=True)
    try:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None or row["status"] != "completed":
            raise ValueError("select a completed run with a saved checkpoint")
        cfg = config.load_experiment(row["config_path"])
        # Experiment construction writes catalogue provenance, so use a non-writing reader.
        model_cfg = config.load_models(configs_dir)[cfg.model]
        artifact = conn.execute(
            "SELECT path FROM artifacts WHERE run_id=? AND kind='checkpoint'", (run_id,)
        ).fetchone()
        fusion = conn.execute("SELECT * FROM fusion_solves WHERE run_id=?", (run_id,)).fetchone()
        if artifact is None or fusion is None:
            raise ValueError("checkpoint and fitted fusion are required")
        ckpt = torch.load(artifact["path"], weights_only=True, map_location="cpu")
        extra = {
            "head_hidden": model_cfg.head_hidden,
            "dropout": model_cfg.dropout,
            "fusion_type": model_cfg.fusion_type,
            "svm_c": model_cfg.svm_c,
        }
        model = build_model(ckpt["dims"], extra).eval()
        model.load_state_dict(ckpt["state_dict"])
        mods = list(ckpt["dims"])
        score_mods = mods if model_cfg.fusion_type == "weighted_sum" else ["joint"]
        weights_by_name = {
            r["modality"]: r["weight"]
            for r in conn.execute("SELECT * FROM fusion_weights WHERE run_id=?", (run_id,))
        }
        weights = [float(weights_by_name[m]) for m in score_mods]
        cal = json.loads(fusion["calibration_json"]) if fusion["calibration_json"] else None
        scale = fusion["temperature"] if fusion["temperature"] is not None else 1.0
        from optlab.train.data import load_split

        backbones = {
            "visual": model_cfg.visual_backbone,
            "audio": model_cfg.audio_backbone,
            "frequency": "freq_spectrum",
        }
        val = load_split(
            Path(cfg.manifests_dir) / f"{cfg.train_dataset}.csv",
            "val",
            {m: Path(cfg.features_dir) / cfg.train_dataset / backbones[m] for m in mods},
            "tune",
        )
        with torch.no_grad():
            z = Experiment._logits(None, model, val)
        scores = (apply(z, cal) @ np.array(weights)) * scale
        vs, vy, _ = video_scores(scores, val.y.numpy(), val.video_ids)
        thresholds = {
            "frame": eer_and_threshold(scores, val.y.numpy())[1],
            "video": eer_and_threshold(vs, vy)[1],
        }
        extraction = config.load_extraction(configs_dir).model_dump(mode="json")
        # Use recorded cache preprocessing rather than silently changing the training crop/sampling.
        for modality in mods:
            meta_path = (
                Path(cfg.features_dir) / cfg.train_dataset / backbones[modality] / "meta.json"
            )
            metadata = json.loads(meta_path.read_text())
            for key, value in metadata.get("preprocessing", {}).items():
                if key in extraction:
                    extraction[key] = value
        bundle = {
            "format_version": 1,
            "state_dict": ckpt["state_dict"],
            "dims": ckpt["dims"],
            "model_extra": extra,
            "model_config": model_cfg.model_dump(mode="json"),
            "input_modalities": mods,
            "score_modalities": score_mods,
            "weights": weights,
            "calibration": cal,
            "scale": scale,
            "thresholds": thresholds,
            "extraction": extraction,
            "source_run": {
                "id": run_id,
                "experiment": cfg.name,
                "optimizer_id": row["optimizer_id"],
                "seed": row["seed"],
                "git_commit": row["git_commit"],
                "train_dataset": cfg.train_dataset,
            },
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(bundle, out)
        return {"bundle": str(out), "source_run": bundle["source_run"], "input_modalities": mods}
    finally:
        conn.close()


def load_bundle(path: Path) -> dict:
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    if bundle.get("format_version") != 1:
        raise ValueError("unsupported inference bundle version")
    return bundle


@torch.no_grad()
def predict_features(bundle: dict, features: dict[str, np.ndarray], granularity="video") -> dict:
    if set(features) != set(bundle["input_modalities"]):
        raise ValueError(
            "input modalities must match the trained model; missing audio is not imputed"
        )
    arrays = {m: np.asarray(x, dtype=np.float32) for m, x in features.items()}
    n = len(next(iter(arrays.values())))
    if n < 1 or any(
        x.ndim != 2 or len(x) != n or x.shape[1] != bundle["dims"][m] or not np.isfinite(x).all()
        for m, x in arrays.items()
    ):
        raise ValueError("finite, aligned features with the trained dimensions are required")
    model = build_model(bundle["dims"], bundle["model_extra"]).eval()
    model.load_state_dict(bundle["state_dict"])
    logits = model({m: torch.from_numpy(x) for m, x in arrays.items()}).numpy().astype(float)
    calibrated = apply(logits, bundle["calibration"])
    score = float(np.mean((calibrated @ np.array(bundle["weights"])) * bundle["scale"]))
    threshold = float(bundle["thresholds"][granularity])
    return {
        "prediction": "manipulated" if score >= threshold else "real",
        "score": float(expit(score)),
        "logit": score,
        "threshold_logit": threshold,
        "frames_analyzed": n,
        "streams": {
            m: float(expit(calibrated[:, i].mean()))
            for i, m in enumerate(bundle["score_modalities"])
        },
        "source_run": bundle["source_run"],
        "notice": (
            "Research demo. Scores are not verified authenticity probabilities. "
            "Image-dataset accuracy and unseen-generator robustness are unmeasured."
        ),
    }


def predict_media(bundle: dict, path: Path, device="cpu", detector=None, extractors=None) -> dict:
    from PIL import Image, ImageOps

    from optlab.data.extractors import (
        ClipVisualExtractor,
        FrequencyExtractor,
        Wav2Vec2AudioExtractor,
    )
    from optlab.data.faces import MTCNNDetector, crop_face
    from optlab.data.preprocess import load_audio, read_sampled_frames

    ext = path.suffix.lower()
    image = ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if image and "audio" in bundle["input_modalities"]:
        raise ValueError(
            "images require an exported visual/frequency model without an audio branch"
        )
    cfg = bundle["extraction"]
    if image:
        with Image.open(path) as im:
            frames = [np.array(ImageOps.exif_transpose(im).convert("RGB"))]
    elif ext in {".mp4", ".mov", ".avi", ".webm", ".mkv"}:
        _, frames = read_sampled_frames(path, cfg["frames_per_video"])
    else:
        raise ValueError("unsupported image/video file type")
    # MTCNN uses area interpolation unsupported for arbitrary sizes on MPS.
    detector = detector or MTCNNDetector("cpu" if device.startswith("mps") else device)
    boxes = detector.detect(frames) if frames else []
    crops = [
        crop_face(f, b, cfg["face_margin"], cfg["face_size"])
        for f, b in zip(frames, boxes, strict=True)
        if b is not None
    ]
    if not crops:
        raise ValueError("no usable face detected; a prediction cannot be produced")
    if extractors is None:
        name = bundle["model_config"]["visual_backbone"]
        spec = cfg["visual_backbones"][name]
        extractors = {"visual": ClipVisualExtractor(name, spec["arch"], spec["pretrained"], device)}
        if "frequency" in bundle["input_modalities"]:
            extractors["frequency"] = FrequencyExtractor(cfg["freq_size"], cfg["freq_bins"])
        if "audio" in bundle["input_modalities"]:
            name = bundle["model_config"]["audio_backbone"]
            spec = cfg["audio_backbones"][name]
            extractors["audio"] = Wav2Vec2AudioExtractor(
                name, spec["hf_id"], cfg["audio_sample_rate"], device
            )
    features = {m: extractors[m](crops) for m in bundle["input_modalities"] if m != "audio"}
    if "audio" in bundle["input_modalities"]:
        embedding = extractors["audio"]([load_audio(path, cfg["audio_sample_rate"])])
        features["audio"] = np.repeat(embedding, len(crops), axis=0)
    out = predict_features(bundle, features, "frame" if image else "video")
    return {**out, "media_type": "image" if image else "video", "sampled_frames": len(frames)}
