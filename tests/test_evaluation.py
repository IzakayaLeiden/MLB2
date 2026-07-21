from __future__ import annotations

import math

import pytest

from mlb_predictor.evaluation import (
    binary_classification_metrics,
    calibration_bins,
    evaluate_prediction_set,
)


def test_binary_classification_metrics_returns_expected_values() -> None:
    metrics = binary_classification_metrics(
        [1, 0, 1, 0],
        [0.9, 0.2, 0.6, 0.4],
    )

    assert metrics == pytest.approx(
        {
            "log_loss": -(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.6)) / 4,
            "brier_score": 0.0925,
            "accuracy": 1.0,
            "row_count": 4,
            "positive_rate": 0.5,
            "mean_probability": 0.525,
        }
    )


def test_binary_classification_metrics_uses_half_as_positive_threshold() -> None:
    metrics = binary_classification_metrics([1, 0], [0.5, 0.499])

    assert metrics["accuracy"] == 1.0


def test_binary_classification_metrics_clips_log_loss_after_validation() -> None:
    metrics = binary_classification_metrics([1, 0], [0.0, 1.0])

    assert math.isfinite(metrics["log_loss"])
    assert metrics["log_loss"] > 0
    assert metrics["brier_score"] == 1.0


def test_calibration_bins_uses_left_closed_bins_and_includes_one_in_last_bin() -> None:
    bins = calibration_bins(
        [0, 1, 1, 0, 1],
        [0.0, 0.25, 0.5, 0.75, 1.0],
        n_bins=2,
    )

    assert bins == [
        {
            "lower_bound": 0.0,
            "upper_bound": 0.5,
            "count": 2,
            "mean_probability": pytest.approx(0.125),
            "observed_rate": pytest.approx(0.5),
        },
        {
            "lower_bound": 0.5,
            "upper_bound": 1.0,
            "count": 3,
            "mean_probability": pytest.approx(0.75),
            "observed_rate": pytest.approx(2 / 3),
        },
    ]


def test_calibration_bins_preserves_empty_bins_with_none_rates() -> None:
    bins = calibration_bins([0, 1], [0.1, 0.9], n_bins=4)

    assert len(bins) == 4
    assert bins[1] == {
        "lower_bound": 0.25,
        "upper_bound": 0.5,
        "count": 0,
        "mean_probability": None,
        "observed_rate": None,
    }
    assert bins[2] == {
        "lower_bound": 0.5,
        "upper_bound": 0.75,
        "count": 0,
        "mean_probability": None,
        "observed_rate": None,
    }


def test_calibration_bins_assigns_decimal_edge_to_following_bin() -> None:
    bins = calibration_bins([1], [0.3], n_bins=10)

    assert bins[2]["count"] == 0
    assert bins[3]["count"] == 1


def test_evaluate_prediction_set_combines_metrics_and_calibration() -> None:
    result = evaluate_prediction_set([0, 1], [0.2, 0.8], n_bins=2)

    assert set(result) == {"metrics", "calibration"}
    assert result["metrics"] == binary_classification_metrics([0, 1], [0.2, 0.8])
    assert result["calibration"] == calibration_bins([0, 1], [0.2, 0.8], n_bins=2)


@pytest.mark.parametrize(
    ("y_true", "probabilities"),
    [
        ([0], [0.2, 0.8]),
        ([], []),
        ([0, 2], [0.2, 0.8]),
        ([0, 0.5], [0.2, 0.8]),
        ([0, float("nan")], [0.2, 0.8]),
        ([0, 1], [-0.01, 0.8]),
        ([0, 1], [0.2, 1.01]),
        ([0, 1], [float("nan"), 0.8]),
        ([0, 1], [float("inf"), 0.8]),
        ([0, 1], [float("-inf"), 0.8]),
    ],
)
@pytest.mark.parametrize(
    "evaluator",
    [binary_classification_metrics, calibration_bins, evaluate_prediction_set],
)
def test_evaluators_reject_invalid_inputs(evaluator, y_true, probabilities) -> None:
    with pytest.raises(ValueError):
        evaluator(y_true, probabilities)


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
def test_calibration_evaluators_require_positive_integer_bin_count(n_bins) -> None:
    with pytest.raises(ValueError):
        calibration_bins([0, 1], [0.2, 0.8], n_bins=n_bins)
    with pytest.raises(ValueError):
        evaluate_prediction_set([0, 1], [0.2, 0.8], n_bins=n_bins)
