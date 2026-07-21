from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pytest

from mlb_predictor.io import sha256_file, write_rows_csv
from mlb_predictor.modeling import DEFAULT_FEATURE_SPECS, ModelBundle
from mlb_predictor.training import (
    chronological_date_split,
    train_model_artifacts,
    validate_training_rows,
)


def _training_rows(days: int = 10, games_per_day: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    first = date(2024, 4, 1)
    game_id = 1
    for day_index in range(days):
        official_date = first + timedelta(days=day_index)
        history_date = None if day_index == 0 else (official_date - timedelta(days=1)).isoformat()
        for game_index in range(games_per_day):
            signal = float((game_index % 3) - 1) + day_index * 0.02
            row: dict[str, object] = {
                "game_id": game_id,
                "official_date": official_date.isoformat(),
                "game_start_utc": f"{official_date.isoformat()}T{18 + game_index:02d}:00:00Z",
                "feature_cutoff_policy": "prior_official_date_only",
                "feature_version": "pregame-v1",
                "home_history_through_date": history_date,
                "away_history_through_date": history_date,
                "home_win": int((day_index + game_index) % 2 == 0),
                "day_night": "night" if game_index % 2 else "day",
                "elo_expected_home_win_probability": 0.55 + signal * 0.03,
            }
            for spec in DEFAULT_FEATURE_SPECS:
                if spec.transform == "numeric":
                    row[spec.source] = (
                        signal
                        if "difference" in spec.name or "minus" in spec.name
                        else float(day_index + 1)
                    )
            rows.append(row)
            game_id += 1
    return rows


def test_chronological_date_split_keeps_dates_atomic_and_ordered() -> None:
    rows = list(reversed(_training_rows()))
    split = chronological_date_split(rows)
    train_dates = {row["official_date"] for row in split.train}
    validation_dates = {row["official_date"] for row in split.validation}
    test_dates = {row["official_date"] for row in split.test}

    assert len(split.train) == 36
    assert len(split.validation) == 12
    assert len(split.test) == 12
    assert train_dates.isdisjoint(validation_dates | test_dates)
    assert validation_dates.isdisjoint(test_dates)
    assert max(train_dates) < min(validation_dates) < min(test_dates)
    assert split.to_metadata()["date_atomic"] is True


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0.0, 0.2), (1.0, 0.2), (0.8, 0.2), (0.6, -0.1)],
)
def test_chronological_date_split_rejects_invalid_fractions(
    train_fraction: float,
    validation_fraction: float,
) -> None:
    with pytest.raises(ValueError):
        chronological_date_split(
            _training_rows(),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )


def test_training_validation_rejects_lookahead_and_duplicate_games() -> None:
    rows = _training_rows()
    rows[0]["home_history_through_date"] = rows[0]["official_date"]
    with pytest.raises(ValueError, match="엄격히 이전"):
        validate_training_rows(rows)

    rows = _training_rows()
    rows[1]["game_id"] = rows[0]["game_id"]
    with pytest.raises(ValueError, match="중복 game_id"):
        validate_training_rows(rows)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("home_score", 5, "결과 누수 위험"),
        ("game_id", 0, "양의 정수"),
        ("elo_expected_home_win_probability", "0.5", "0~1"),
        ("feature_cutoff_policy", "same_day", "prior_official_date_only"),
    ],
)
def test_training_validation_fails_closed_on_contract_violations(
    column: str,
    value: object,
    message: str,
) -> None:
    rows = _training_rows()
    rows[0][column] = value

    with pytest.raises(ValueError, match=message):
        validate_training_rows(rows)


def test_training_validation_requires_both_classes_in_fit_partitions() -> None:
    rows = _training_rows()
    for row in rows[:36]:
        row["home_win"] = 1

    with pytest.raises(ValueError, match="train 분할"):
        validate_training_rows(rows)


def test_train_model_artifacts_writes_sites_ready_fail_closed_contract(tmp_path: Path) -> None:
    source = write_rows_csv(tmp_path / "features.csv", _training_rows())
    output = tmp_path / "model-run"

    manifest = train_model_artifacts(source, output)
    model_payload = json.loads((output / "model.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["build_status"] == "succeeded"
    assert manifest["artifacts_valid"] is True
    assert manifest["performance_gate_passed"] is None
    assert manifest["deployment_approved"] is False
    assert saved_manifest["deployment_approved"] is False
    assert model_payload["schema_version"] == "mlb-win-probability-model-v1"
    assert model_payload["default_output"] == "raw_logistic_home_win_probability"
    assert model_payload["feature_cutoff_policy"] == "prior_official_date_only"
    assert model_payload["runtime_contract"]["implementation"] == "portable-json"
    assert model_payload["runtime_contract"]["probability_epsilon"] == 1e-12
    assert "coefficients[i]" in model_payload["runtime_contract"]["raw_formula"]
    assert evaluation["assessment"] == "sample_only_not_performance_evidence"
    assert evaluation["test_set_usage"] == "evaluation_only_not_used_for_selection_or_tuning"
    assert evaluation["default_output_basis"] == (
        "raw_probability_until_calibration_has_independent_support"
    )
    assert set(evaluation["test_results"]) == {
        "constant_home_rate",
        "elo_baseline",
        "logistic_raw",
        "logistic_platt_calibrated",
    }
    for name in ("model.json", "evaluation.json"):
        assert manifest["artifacts"][name]["sha256"] == sha256_file(output / name)
        assert manifest["artifacts"][name]["bytes"] == (output / name).stat().st_size

    bundle = ModelBundle.from_dict(model_payload["runtime_model"])
    prediction_row = _training_rows()[0]
    probability = float(bundle.predict_rows([prediction_row])[0])
    runtime = model_payload["runtime_model"]
    values = [
        float(prediction_row[spec["source"]])
        if spec["transform"] == "numeric"
        else float(prediction_row[spec["source"]] == spec["value"])
        for spec in runtime["feature_specs"]
    ]
    logistic = runtime["logistic"]
    score = logistic["intercept"] + sum(
        coefficient * ((value - mean) / scale)
        for value, mean, scale, coefficient in zip(
            values,
            logistic["mean"],
            logistic["scale"],
            logistic["coefficients"],
            strict=True,
        )
    )
    portable_probability = 1.0 / (1.0 + math.exp(-score))

    assert 0.0 < probability < 1.0
    assert portable_probability == pytest.approx(probability)


def test_train_model_artifacts_never_overwrites_existing_output(tmp_path: Path) -> None:
    source = write_rows_csv(tmp_path / "features.csv", _training_rows())
    output = tmp_path / "model-run"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        train_model_artifacts(source, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_train_model_artifacts_rejects_bad_bin_count_before_reading_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_bins"):
        train_model_artifacts(
            tmp_path / "missing.csv",
            tmp_path / "model-run",
            n_bins=0,
        )
