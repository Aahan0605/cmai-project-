import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from optlab.eval.metrics import detection_metrics, eer_and_threshold, video_scores
from optlab.eval.stats import (
    area_under_curve,
    bootstrap_ci,
    cohens_d,
    epochs_to_target,
    holm,
    pairwise,
)


def test_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(size=500)
    assert detection_metrics(s, y, 0.5)["auc"] == pytest.approx(roc_auc_score(y, s))


def test_eer_hand_made_cases() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert eer_and_threshold(np.arange(8.0), y)[0] == 0.0  # perfectly separated
    assert eer_and_threshold(-np.arange(8.0), y)[0] == 1.0  # perfectly inverted
    # One negative above one positive among 4 + 4: FPR = FNR = 0.25 at the crossing.
    s = np.array([0.0, 1.0, 2.0, 5.0, 4.0, 6.0, 7.0, 8.0])
    assert eer_and_threshold(s, y)[0] == pytest.approx(0.25)


def test_video_aggregation_is_mean_of_frames() -> None:
    scores = np.array([1.0, 3.0, -2.0, 0.0, 4.0])
    labels = np.array([1, 1, 0, 0, 1])
    vids = ["a", "a", "b", "b", "c"]
    vs, vl, order = video_scores(scores, labels, vids)
    assert order == ["a", "b", "c"]
    np.testing.assert_allclose(vs, [2.0, -1.0, 4.0])
    np.testing.assert_array_equal(vl, [1, 0, 1])


def test_threshold_metrics() -> None:
    m = detection_metrics(np.array([0.1, 0.6, 0.4, 0.9]), np.array([0, 0, 1, 1]), 0.5)
    assert m["accuracy"] == 0.5 and m["f1"] == pytest.approx(0.5)


def test_holm_against_hand_computation() -> None:
    # Sorted p: 0.01, 0.02, 0.04 -> x3, x2, x1 -> 0.03, 0.04, 0.04 (monotone).
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm([float("nan"), 0.01]) == pytest.approx([1.0, 0.02])


def test_bootstrap_and_effect_size() -> None:
    lo, hi = bootstrap_ci([0.80, 0.82, 0.81, 0.79, 0.83])
    assert lo < 0.81 < hi
    assert cohens_d([1, 2, 3], [1, 2, 3]) == 0.0


def test_pairwise_marks_noise_as_indistinguishable() -> None:
    rng = np.random.default_rng(0)
    rows = pairwise(
        {
            "a": list(0.80 + rng.normal(0, 0.01, 5)),
            "b": list(0.80 + rng.normal(0, 0.01, 5)),
            "c": list(0.60 + rng.normal(0, 0.01, 5)),
        }
    )
    verdict = {(r["a"], r["b"]): r["distinguishable"] for r in rows}
    assert verdict == {("a", "b"): False, ("a", "c"): True, ("b", "c"): True}


def test_convergence_helpers() -> None:
    assert epochs_to_target([0.5, 0.7, 0.9, 0.95], 0.9) == 2
    assert epochs_to_target([0.5], 0.9) is None
    assert area_under_curve([1.0, 0.5, 0.0]) == pytest.approx(1.0)


def test_video_aggregation_rejects_inconsistent_labels():
    with pytest.raises(ValueError, match="same label"):
        video_scores(np.array([1.0, 2.0]), np.array([0, 1]), ["a", "a"])


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([1.0, np.inf], [0, 1]),
        ([1.0, 2.0], [1, 1]),
        ([], []),
        ([1.0], [0, 1]),
    ],
)
def test_roc_rejects_invalid_inputs(scores, labels):
    with pytest.raises(ValueError):
        eer_and_threshold(np.array(scores), np.array(labels))
