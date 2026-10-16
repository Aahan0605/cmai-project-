import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from optlab.demo_api import create_demo_app
from optlab.inference import predict_features, predict_media
from optlab.train.trainer import build_model


@pytest.fixture
def bundle():
    dims = {"visual": 4, "frequency": 2}
    extra = {"head_hidden": 8, "dropout": 0}
    model = build_model(dims, extra)
    return {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "dims": dims,
        "model_extra": extra,
        "input_modalities": list(dims),
        "score_modalities": list(dims),
        "weights": [0.6, 0.4],
        "calibration": None,
        "scale": 1.0,
        "thresholds": {"frame": 0.0, "video": 0.0},
        "extraction": {"face_margin": 0.0, "face_size": 16, "frames_per_video": 2},
        "source_run": {"id": 1, "train_dataset": "synthetic"},
    }


def test_predict_features_aggregates_fixed_model_without_refitting(bundle):
    x = {"visual": np.ones((3, 4)), "frequency": np.ones((3, 2))}
    before = {k: v.clone() for k, v in bundle["state_dict"].items()}
    result = predict_features(bundle, x)
    assert 0 <= result["score"] <= 1 and result["frames_analyzed"] == 3
    assert all(torch.equal(v, bundle["state_dict"][k]) for k, v in before.items())
    with pytest.raises(ValueError, match="modalities"):
        predict_features(bundle, {"visual": x["visual"]})


def test_image_prediction_with_injected_extractors(bundle, tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8)).save(path)

    class Detector:
        def detect(self, frames):
            return [(0.0, 0.0, 32.0, 32.0) for _ in frames]

    extractors = {
        "visual": lambda crops: np.ones((len(crops), 4)),
        "frequency": lambda crops: np.ones((len(crops), 2)),
    }
    result = predict_media(bundle, path, detector=Detector(), extractors=extractors)
    assert result["media_type"] == "image" and result["frames_analyzed"] == 1
    bundle["input_modalities"].append("audio")
    with pytest.raises(ValueError, match="images require"):
        predict_media(bundle, path, detector=Detector(), extractors=extractors)


def test_demo_rejects_empty_and_unsupported_uploads(bundle, tmp_path):
    path = tmp_path / "bundle.pt"
    torch.save(bundle, path)
    client = TestClient(create_demo_app(path))
    assert client.get("/demo/health").status_code == 200
    assert (
        client.post("/demo/predict", content=b"x", headers={"x-filename": "a.exe"}).status_code
        == 415
    )
    assert (
        client.post("/demo/predict", content=b"", headers={"x-filename": "a.png"}).status_code
        == 400
    )


def test_demo_deletes_temporary_media(bundle, tmp_path, monkeypatch):
    from optlab import demo_api

    paths = []

    def predict(b, path, device):
        paths.append(path)
        assert path.read_bytes() == b"fakeimage"
        return {"score": 0.5}

    monkeypatch.setattr(demo_api, "predict_media", predict)
    path = tmp_path / "bundle.pt"
    torch.save(bundle, path)
    client = TestClient(create_demo_app(path))
    assert client.post(
        "/demo/predict", content=b"fakeimage", headers={"x-filename": "../../a.png"}
    ).json() == {"score": 0.5}
    assert not paths[0].exists()
