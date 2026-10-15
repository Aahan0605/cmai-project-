import numpy as np
import pytest

from optlab.fusion.calibration import apply, fit


def test_calibration_reduces_overconfidence_and_preserves_ranking():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-latent))).astype(float)
    z = np.column_stack([latent * 20, latent * 0.1])
    cal = fit(z, y)
    assert np.all(np.array(cal["nll_after"]) < cal["nll_before"])
    calibrated = apply(z, cal)
    assert np.allclose(calibrated[:, 0], calibrated[:, 1], atol=1e-4)
    assert np.array_equal(np.argsort(z[:, 0]), np.argsort(calibrated[:, 0]))
    assert cal["fit_split"] == "val_tune"


def test_calibration_rejects_missing_class():
    with pytest.raises(ValueError, match="both binary classes"):
        fit(np.ones((5, 2)), np.zeros(5))


@pytest.mark.parametrize(
    "z,y",
    [
        (np.ones((4, 2)), np.array([[0], [1], [0], [1]])),
        (np.ones((4, 0)), np.array([0, 1, 0, 1])),
    ],
)
def test_calibration_rejects_broadcasting_labels_and_missing_streams(z, y):
    with pytest.raises(ValueError, match="aligned logits"):
        fit(z, y)
