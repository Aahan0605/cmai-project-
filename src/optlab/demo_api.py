"""Opt-in local prediction service. Separate from the read-only results API."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from optlab.inference import load_bundle, predict_media

MAX_BYTES = 25 * 1024 * 1024


def create_demo_app(bundle_path: Path, device="cpu", cors_origin="http://localhost:5173"):
    bundle = load_bundle(bundle_path)
    app = FastAPI(title="OptLab local media demo")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[cors_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    lock = Lock()

    @app.get("/demo/health")
    def health():
        return {
            "status": "ok",
            "input_modalities": bundle["input_modalities"],
            "source_run": bundle["source_run"],
            "max_bytes": MAX_BYTES,
        }

    def analyze(raw: bytes, suffix: str):
        with lock, TemporaryDirectory(prefix="optlab-media-") as directory:
            path = Path(directory) / ("input" + suffix)
            path.write_bytes(raw)
            return predict_media(bundle, path, device)

    @app.post("/demo/predict")
    async def predict(request: Request):
        suffix = Path(request.headers.get("x-filename", "")).suffix.lower()
        if suffix not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".mp4",
            ".mov",
            ".avi",
            ".webm",
            ".mkv",
        }:
            raise HTTPException(415, "unsupported image/video extension")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > MAX_BYTES:
                raise HTTPException(413, "media exceeds the 25 MiB limit")
            raw.extend(chunk)
        if not raw:
            raise HTTPException(400, "empty media file")
        try:
            return await run_in_threadpool(analyze, bytes(raw), suffix)
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    return app
