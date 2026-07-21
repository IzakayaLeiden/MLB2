from __future__ import annotations

import json

import numpy as np
import pytest

from mlb_predictor.modeling import (
    DEFAULT_FEATURE_SPECS,
    FeatureSpec,
    ModelBundle,
    PlattCalibrator,
    StandardizedLogisticRegression,
    extract_feature_matrix,
)


def _feature_row(*, day_night: str = "day") -> dict[str, float | int | str]:
    return {
        "home_elo_minus_away": 25.0,
        "season_win_pct_difference": 0.1,
        "recent_win_pct_difference": 0.2,
        "recent_run_margin_difference": 1.5,
        "rest_days_difference": -1,
        "home_games_before": 10,
        "away_games_before": 11,
        "home_recent_games_count": 8,
        "away_recent_games_count": 9,
        "home_has_prior_history": 1,
        "away_has_prior_history": 1,
        "day_night": day_night,
    }


def test_feature_spec_serialization_and_default_schema() -> None:
    spec = FeatureSpec(
        name="is_night_game",
        source="day_night",
        transform="equals",
        value="night",
    )

    assert FeatureSpec.from_dict(spec.to_dict()) == spec
    assert spec.to_dict() == {
        "name": "is_night_game",
        "source": "day_night",
        "transform": "equals",
        "value": "night",
    }
    assert [item.name for item in DEFAULT_FEATURE_SPECS] == [
        "home_elo_minus_away",
        "season_win_pct_difference",
        "recent_win_pct_difference",
        "recent_run_margin_difference",
        "rest_days_difference",
        "home_games_before",
        "away_games_before",
        "home_recent_games_count",
        "away_recent_games_count",
        "home_has_prior_history",
        "away_has_prior_history",
        "is_night_game",
    ]
    assert DEFAULT_FEATURE_SPECS[-1] == spec


def test_extract_feature_matrix_encodes_night_and_preserves_spec_order() -> None:
    day_row = _feature_row(day_night="day")
    night_row = _feature_row(day_night="night")
    night_row["home_elo_minus_away"] = -10.0

    matrix = extract_feature_matrix([day_row, night_row])

    assert matrix.shape == (2, len(DEFAULT_FEATURE_SPECS))
    assert matrix.dtype == np.dtype(float)
    assert matrix[:, 0].tolist() == [25.0, -10.0]
    assert matrix[:, -1].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("home_elo_minus_away"),
        lambda row: row.__setitem__("recent_win_pct_difference", float("nan")),
        lambda row: row.__setitem__("recent_run_margin_difference", float("inf")),
        lambda row: row.__setitem__("home_games_before", None),
        lambda row: row.pop("day_night"),
        lambda row: row.__setitem__("day_night", float("nan")),
    ],
)
def test_extract_feature_matrix_rejects_missing_or_non_finite_values(mutation) -> None:
    row = _feature_row()
    mutation(row)

    with pytest.raises(ValueError):
        extract_feature_matrix([row])


def test_logistic_regression_is_monotonic_and_standardizes_from_training_only() -> None:
    matrix = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    target = np.asarray([0, 0, 0, 1, 1, 1])
    model = StandardizedLogisticRegression(l2=0.01).fit(
        matrix,
        target,
        feature_names=["signal"],
    )
    before = model.to_dict()

    probabilities = model.predict_proba(np.asarray([[-4.0], [0.0], [4.0], [400.0]]))

    assert np.all(np.diff(probabilities) > 0)
    assert model.feature_names == ("signal",)
    assert model.mean == pytest.approx([0.0])
    assert model.to_dict() == before


def test_logistic_l2_does_not_penalize_intercept() -> None:
    matrix = np.zeros((4, 1))
    target = np.asarray([0, 1, 1, 1])

    model = StandardizedLogisticRegression(l2=1_000.0).fit(matrix, target)

    assert model.coefficients == pytest.approx([0.0], abs=1e-10)
    assert model.predict_proba(matrix) == pytest.approx([0.75] * 4, abs=1e-8)


def test_logistic_serialization_round_trip_is_json_safe() -> None:
    matrix = np.asarray([[-2.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    target = np.asarray([0, 0, 1, 1])
    model = StandardizedLogisticRegression(l2=0.05).fit(
        matrix,
        target,
        feature_names=["signal", "flag"],
    )

    payload = model.to_dict()
    restored = StandardizedLogisticRegression.from_dict(json.loads(json.dumps(payload)))

    assert set(payload) == {
        "feature_names",
        "mean",
        "scale",
        "coefficients",
        "intercept",
        "l2",
    }
    assert restored.to_dict() == payload
    assert restored.predict_proba(matrix) == pytest.approx(model.predict_proba(matrix))


def test_logistic_prediction_fails_closed_when_standardization_overflows() -> None:
    model = StandardizedLogisticRegression.from_dict(
        {
            "feature_names": ["signal"],
            "mean": [-1e308],
            "scale": [1.0],
            "coefficients": [0.0],
            "intercept": 0.0,
            "l2": 0.0,
        }
    )

    with pytest.raises(ValueError):
        model.predict_proba([[1e308]])


def test_platt_calibrator_uses_logit_and_round_trips() -> None:
    raw_probabilities = np.asarray([0.05, 0.2, 0.8, 0.95])
    target = np.asarray([0, 0, 1, 1])
    calibrator = PlattCalibrator().fit(raw_probabilities, target, l2=0.01)

    calibrated = calibrator.predict(np.asarray([0.0, 0.1, 0.5, 0.9, 1.0]))
    payload = calibrator.to_dict()
    restored = PlattCalibrator.from_dict(json.loads(json.dumps(payload)))

    assert calibrator.coefficient > 0
    assert np.all(np.diff(calibrated) > 0)
    assert set(payload) == {"coefficient", "intercept", "l2"}
    assert restored.to_dict() == payload
    assert restored.predict(raw_probabilities) == pytest.approx(
        calibrator.predict(raw_probabilities)
    )


def test_model_bundle_predicts_rows_and_serializes() -> None:
    rows = [_feature_row(day_night=value) for value in ("day", "day", "night", "night")]
    rows[0]["home_elo_minus_away"] = -60.0
    rows[1]["home_elo_minus_away"] = -20.0
    rows[2]["home_elo_minus_away"] = 20.0
    rows[3]["home_elo_minus_away"] = 60.0
    matrix = extract_feature_matrix(rows)
    target = np.asarray([0, 0, 1, 1])
    logistic = StandardizedLogisticRegression(l2=0.1).fit(
        matrix,
        target,
        feature_names=[spec.name for spec in DEFAULT_FEATURE_SPECS],
    )
    raw = logistic.predict_proba(matrix)
    calibrator = PlattCalibrator().fit(raw, target, l2=0.1)
    bundle = ModelBundle(DEFAULT_FEATURE_SPECS, logistic, calibrator)

    payload = bundle.to_dict()
    restored = ModelBundle.from_dict(json.loads(json.dumps(payload)))

    assert bundle.predict_rows(rows) == pytest.approx(logistic.predict_proba(matrix))
    assert bundle.predict_rows(rows, calibrated=True) == pytest.approx(
        calibrator.predict(logistic.predict_proba(matrix))
    )
    assert bundle.predict_rows(rows) == pytest.approx(bundle.predict_matrix(matrix))
    assert restored.to_dict() == payload
    assert restored.predict_rows(rows) == pytest.approx(bundle.predict_rows(rows))


def test_model_bundle_rejects_feature_name_or_order_mismatch() -> None:
    matrix = np.asarray([[-1.0, 0.0], [1.0, 1.0]])
    target = np.asarray([0, 1])
    logistic = StandardizedLogisticRegression(l2=0.1).fit(
        matrix,
        target,
        feature_names=["second", "first"],
    )
    calibrator = PlattCalibrator().fit(
        logistic.predict_proba(matrix),
        target,
        l2=0.1,
    )
    specs = (
        FeatureSpec("first", "first", "numeric", None),
        FeatureSpec("second", "second", "numeric", None),
    )

    with pytest.raises(ValueError):
        ModelBundle(specs, logistic, calibrator)


def test_feature_spec_normalizes_json_numeric_scalars_and_rejects_objects() -> None:
    spec = FeatureSpec("flag", "flag", "equals", np.int64(1))

    assert type(spec.value) is int
    json.dumps(spec.to_dict(), allow_nan=False)
    with pytest.raises(ValueError):
        FeatureSpec("flag", "flag", "equals", object())
