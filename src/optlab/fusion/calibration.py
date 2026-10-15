"""Per-stream temperature scaling, fitted on val_tune before val_fusion is used.

Positive scales preserve each head's ranking. They reduce scale confounding, but mixing
weights still reflect redundancy and validation sampling, not causal importance.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def fit(z: np.ndarray, y: np.ndarray) -> dict:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if (
        z.ndim != 2
        or y.ndim != 1
        or z.shape[1] == 0
        or len(z) != len(y)
        or not np.isfinite(z).all()
    ):
        raise ValueError("calibration requires finite aligned logits")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("calibration requires both binary classes")
    scales, before, after = [], [], []
    for col in z.T:

        def loss(log_scale, col=col):
            scores = col * np.exp(log_scale)
            return float(np.mean(np.logaddexp(0, scores) - y * scores))

        result = minimize_scalar(loss, bounds=(-12, 12), method="bounded")
        if not result.success:
            raise RuntimeError("stream calibration did not converge")
        best = min((0.0, -12.0, 12.0, float(result.x)), key=loss)
        scales.append(float(np.exp(best)))
        before.append(loss(0))
        after.append(loss(best))
    return {
        "method": "temperature",
        "fit_split": "val_tune",
        "scales": scales,
        "nll_before": before,
        "nll_after": after,
    }


def apply(z: np.ndarray, calibration: dict | None) -> np.ndarray:
    return z if calibration is None else z * np.asarray(calibration["scales"])
