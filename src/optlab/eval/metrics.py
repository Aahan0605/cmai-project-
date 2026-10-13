"""Detection metrics (TRD 8.1). Scores are logits or probabilities; higher = more likely fake."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def video_scores(
    scores: np.ndarray, labels: np.ndarray, video_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Video-level score = mean of its frame scores (TRD 3.3)."""
    order = list(dict.fromkeys(video_ids))
    index = {v: i for i, v in enumerate(order)}
    rows = np.array([index[v] for v in video_ids])
    sums = np.bincount(rows, weights=scores, minlength=len(order))
    counts = np.bincount(rows, minlength=len(order))
    vid_labels = np.zeros(len(order))
    vid_labels[rows] = labels
    return sums / counts, vid_labels, order


def eer_and_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate (where false-positive rate = false-negative rate) and its threshold.

    The ROC is piecewise linear, so the crossing is interpolated between the two ROC points
    that bracket it.
    """
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    diff = fpr - fnr
    i = int(np.flatnonzero(diff >= 0)[0])
    if i == 0 or diff[i] == 0:
        return float((fpr[i] + fnr[i]) / 2), float(thr[i])
    a = diff[i - 1] / (diff[i - 1] - diff[i])  # fraction of the way from point i-1 to i
    eer = fpr[i - 1] + a * (fpr[i] - fpr[i - 1])
    return float(eer), float(thr[i])


def detection_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """AUC, EER, AP, plus accuracy and F1 at a threshold fixed from validation."""
    pred = scores >= threshold
    tp = float(np.sum(pred & (labels == 1)))
    fp = float(np.sum(pred & (labels == 0)))
    fn = float(np.sum(~pred & (labels == 1)))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": eer_and_threshold(scores, labels)[0],
        "accuracy": float(np.mean(pred == (labels == 1))),
        "f1": f1,
        "average_precision": float(average_precision_score(labels, scores)),
    }
