from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evaluation import evaluate_prediction_set
from .io import write_json
from .modeling import (
    DEFAULT_FEATURE_SPECS,
    ModelBundle,
    PlattCalibrator,
    StandardizedLogisticRegression,
    extract_feature_matrix,
)


SELECTION_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025
DEFAULT_L2_VALUES = (0.01, 0.1, 1.0, 10.0)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _rows_for_season(rows: Sequence[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["season"]) == season]


def _rows_before(rows: Sequence[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["season"]) < season]


def _target(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["home_win"]) for row in rows], dtype=float)


def _fit_logistic(rows: Sequence[dict[str, Any]], l2: float) -> StandardizedLogisticRegression:
    if not rows:
        raise ValueError("로지스틱 회귀 학습 행이 비어 있습니다.")
    return StandardizedLogisticRegression(l2=l2).fit(
        extract_feature_matrix(rows, DEFAULT_FEATURE_SPECS),
        _target(rows),
        feature_names=[spec.name for spec in DEFAULT_FEATURE_SPECS],
    )


def _candidate_key(name: str, l2: float | None = None) -> str:
    return name if l2 is None else f"{name}:l2={l2:g}"


def _metric_pair(result: Mapping[str, Any]) -> tuple[float, float]:
    metrics = result["metrics"]
    return float(metrics["log_loss"]), float(metrics["brier_score"])


def _selection_fingerprint(selection: Mapping[str, Any]) -> str:
    canonical = json.dumps(selection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_model(
    rows: Iterable[dict[str, Any]],
    *,
    l2_values: Sequence[float] = DEFAULT_L2_VALUES,
) -> dict[str, Any]:
    """2022~2024 워크포워드에서 Log Loss, Brier 순으로 후보를 선택합니다."""

    materialized = sorted((dict(row) for row in rows), key=lambda row: (str(row["official_date"]), int(row["game_id"])))
    candidates: dict[str, dict[str, Any]] = {
        "constant": {"model_type": "constant", "probabilities": [], "targets": [], "folds": []},
        "elo": {"model_type": "elo", "probabilities": [], "targets": [], "folds": []},
    }
    for raw_l2 in l2_values:
        l2 = float(raw_l2)
        if l2 < 0 or not np.isfinite(l2):
            raise ValueError("l2 값은 0 이상의 유한한 수여야 합니다.")
        candidates[_candidate_key("logistic", l2)] = {
            "model_type": "logistic",
            "l2": l2,
            "probabilities": [],
            "targets": [],
            "folds": [],
        }
        candidates[_candidate_key("logistic_platt", l2)] = {
            "model_type": "logistic_platt",
            "l2": l2,
            "probabilities": [],
            "targets": [],
            "folds": [],
        }

    for validation_season in SELECTION_SEASONS:
        validation_rows = _rows_for_season(materialized, validation_season)
        training_rows = _rows_before(materialized, validation_season)
        if not validation_rows or not training_rows:
            raise ValueError(f"워크포워드 시즌 데이터가 부족합니다: {validation_season}")
        targets = _target(validation_rows)
        constant_probability = float(np.mean(_target(training_rows)))
        for key, probabilities in (
            ("constant", np.full(len(validation_rows), constant_probability)),
            ("elo", np.asarray([float(row["elo_expected_home_win_probability"]) for row in validation_rows])),
        ):
            candidates[key]["probabilities"].extend(float(value) for value in probabilities)
            candidates[key]["targets"].extend(int(value) for value in targets)
            candidates[key]["folds"].append({"season": validation_season, "rows": len(validation_rows)})

        calibration_rows = _rows_for_season(materialized, validation_season - 1)
        pre_calibration_rows = [row for row in materialized if int(row["season"]) < validation_season - 1]
        if not calibration_rows or not pre_calibration_rows:
            raise ValueError(f"Platt 보정 시즌 데이터가 부족합니다: {validation_season - 1}")
        for raw_l2 in l2_values:
            l2 = float(raw_l2)
            raw_key = _candidate_key("logistic", l2)
            raw_model = _fit_logistic(training_rows, l2)
            raw_probability = raw_model.predict_proba(extract_feature_matrix(validation_rows, DEFAULT_FEATURE_SPECS))
            candidates[raw_key]["probabilities"].extend(float(value) for value in raw_probability)
            candidates[raw_key]["targets"].extend(int(value) for value in targets)
            candidates[raw_key]["folds"].append({"season": validation_season, "rows": len(validation_rows)})

            platt_key = _candidate_key("logistic_platt", l2)
            platt_model = _fit_logistic(pre_calibration_rows, l2)
            calibration_raw = platt_model.predict_proba(extract_feature_matrix(calibration_rows, DEFAULT_FEATURE_SPECS))
            calibrator = PlattCalibrator().fit(calibration_raw, _target(calibration_rows))
            calibrated_probability = calibrator.predict(
                platt_model.predict_proba(extract_feature_matrix(validation_rows, DEFAULT_FEATURE_SPECS))
            )
            candidates[platt_key]["probabilities"].extend(float(value) for value in calibrated_probability)
            candidates[platt_key]["targets"].extend(int(value) for value in targets)
            candidates[platt_key]["folds"].append(
                {"season": validation_season, "rows": len(validation_rows), "calibration_season": validation_season - 1}
            )

    results: dict[str, Any] = {}
    for key, candidate in candidates.items():
        result = evaluate_prediction_set(candidate.pop("targets"), candidate.pop("probabilities"))
        results[key] = {**candidate, "evaluation": result}
    selected_key = min(results, key=lambda key: (*_metric_pair(results[key]["evaluation"]), key))
    selected = {"candidate": selected_key, **results[selected_key]}
    return {
        "schema_version": "model-selection-v1",
        "selection_seasons": list(SELECTION_SEASONS),
        "ranking_policy": ["log_loss", "brier_score"],
        "candidates": results,
        "selected": selected,
    }


def _fit_selected_runtime(
    rows: Sequence[dict[str, Any]],
    selected: Mapping[str, Any],
    *,
    prediction_season: int,
) -> tuple[ModelBundle | None, dict[str, Any]]:
    model_type = str(selected["model_type"])
    if model_type in {"constant", "elo"}:
        return None, {"model_type": model_type}
    l2 = float(selected["l2"])
    if model_type == "logistic":
        training_rows = _rows_before(rows, prediction_season)
        logistic = _fit_logistic(training_rows, l2)
        calibration_source = _rows_for_season(rows, prediction_season - 1)
        if not calibration_source:
            calibration_source = training_rows
        calibrator = PlattCalibrator().fit(
            logistic.predict_proba(extract_feature_matrix(calibration_source, DEFAULT_FEATURE_SPECS)),
            _target(calibration_source),
        )
        return ModelBundle(DEFAULT_FEATURE_SPECS, logistic, calibrator), {"model_type": model_type, "l2": l2}
    if model_type == "logistic_platt":
        calibration_rows = _rows_for_season(rows, prediction_season - 1)
        training_rows = [row for row in rows if int(row["season"]) < prediction_season - 1]
        if not calibration_rows or not training_rows:
            raise ValueError("Platt 런타임 학습·보정 구간이 부족합니다.")
        logistic = _fit_logistic(training_rows, l2)
        calibrator = PlattCalibrator().fit(
            logistic.predict_proba(extract_feature_matrix(calibration_rows, DEFAULT_FEATURE_SPECS)),
            _target(calibration_rows),
        )
        return ModelBundle(DEFAULT_FEATURE_SPECS, logistic, calibrator), {"model_type": model_type, "l2": l2}
    raise ValueError(f"지원하지 않는 모델 유형입니다: {model_type}")


def evaluate_sealed_holdout(
    rows: Iterable[dict[str, Any]],
    selection: Mapping[str, Any],
    *,
    state_path: str | Path,
) -> dict[str, Any]:
    """선택 완료 후 2025 홀드아웃을 한 번만 평가하고 사용 상태를 봉인합니다."""

    state = Path(state_path)
    if os.path.lexists(state):
        raise RuntimeError(f"2025 홀드아웃이 이미 사용되었습니다: {state}")
    materialized = sorted((dict(row) for row in rows), key=lambda row: (str(row["official_date"]), int(row["game_id"])))
    holdout_rows = _rows_for_season(materialized, HOLDOUT_SEASON)
    prior_rows = _rows_before(materialized, HOLDOUT_SEASON)
    if not holdout_rows or not prior_rows:
        raise ValueError("2025 홀드아웃 또는 이전 학습 데이터가 비어 있습니다.")
    selected = selection["selected"]
    bundle, runtime = _fit_selected_runtime(materialized, selected, prediction_season=HOLDOUT_SEASON)
    targets = _target(holdout_rows)
    constant_probability = np.full(len(holdout_rows), float(np.mean(_target(prior_rows))))
    elo_probability = np.asarray([float(row["elo_expected_home_win_probability"]) for row in holdout_rows])
    if runtime["model_type"] == "constant":
        selected_probability = constant_probability
    elif runtime["model_type"] == "elo":
        selected_probability = elo_probability
    elif runtime["model_type"] == "logistic":
        assert bundle is not None
        selected_probability = bundle.predict_rows(holdout_rows, calibrated=False)
    else:
        assert bundle is not None
        selected_probability = bundle.predict_rows(holdout_rows, calibrated=True)
    evaluations = {
        "constant": evaluate_prediction_set(targets, constant_probability),
        "elo": evaluate_prediction_set(targets, elo_probability),
        "selected": evaluate_prediction_set(targets, selected_probability),
    }
    selected_pair = _metric_pair(evaluations["selected"])
    constant_pair = _metric_pair(evaluations["constant"])
    elo_pair = _metric_pair(evaluations["elo"])
    beats_constant = all(left < right for left, right in zip(selected_pair, constant_pair))
    beats_elo_required = runtime["model_type"] in {"logistic", "logistic_platt"}
    beats_elo = all(left < right for left, right in zip(selected_pair, elo_pair))
    passed = beats_constant and (beats_elo if beats_elo_required else True)
    evaluated_at = _now()
    report = {
        "schema_version": "sealed-holdout-evaluation-v1",
        "holdout_season": HOLDOUT_SEASON,
        "evaluated_at_utc": evaluated_at,
        "usage_count": 1,
        "selection_fingerprint": _selection_fingerprint(selection),
        "selected": runtime,
        "row_count": len(holdout_rows),
        "evaluations": evaluations,
        "criteria": {
            "beats_constant_on_both": beats_constant,
            "beats_elo_required": beats_elo_required,
            "beats_elo_on_both": beats_elo,
        },
        "passed": passed,
    }
    state.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{state.name}.", suffix=".tmp", dir=state.parent)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        write_json(temporary, report)
        if os.path.lexists(state):
            raise RuntimeError(f"2025 홀드아웃이 동시에 사용되었습니다: {state}")
        temporary.replace(state)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report


def freeze_model_v1(
    rows: Iterable[dict[str, Any]],
    selection: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    cutoff_date: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """통과 모델을 기준일까지 재학습하고 model-v1 JSON으로 동결합니다."""

    if not bool(holdout.get("passed")):
        raise RuntimeError("2025 봉인 홀드아웃을 통과하지 않아 모델을 동결할 수 없습니다.")
    eligible = [dict(row) for row in rows if str(row["official_date"]) <= cutoff_date]
    if not eligible:
        raise ValueError("동결 학습 데이터가 비어 있습니다.")
    selected = selection["selected"]
    model_type = str(selected["model_type"])
    runtime_model: dict[str, Any] | None = None
    training: dict[str, Any] = {"rows": len(eligible), "cutoff_date": cutoff_date}
    training["constant_home_win_rate"] = float(np.mean(_target(eligible)))
    if model_type in {"logistic", "logistic_platt"}:
        l2 = float(selected["l2"])
        latest_season = max(int(row["season"]) for row in eligible)
        if model_type == "logistic":
            logistic = _fit_logistic(eligible, l2)
            calibration_rows = _rows_for_season(eligible, latest_season)
        else:
            calibration_rows = _rows_for_season(eligible, latest_season)
            raw_training_rows = [row for row in eligible if int(row["season"]) < latest_season]
            logistic = _fit_logistic(raw_training_rows, l2)
            training["raw_training_rows"] = len(raw_training_rows)
        calibrator = PlattCalibrator().fit(
            logistic.predict_proba(extract_feature_matrix(calibration_rows, DEFAULT_FEATURE_SPECS)),
            _target(calibration_rows),
        )
        runtime_model = ModelBundle(DEFAULT_FEATURE_SPECS, logistic, calibrator).to_dict()
        training.update({"l2": l2, "calibration_rows": len(calibration_rows)})
    payload = {
        "schema_version": "model-v1",
        "model_version": "model-v1",
        "created_at_utc": _now(),
        "frozen": True,
        "retraining_during_shadow": False,
        "model_type": model_type,
        "selected_candidate": selected["candidate"],
        "selection_fingerprint": _selection_fingerprint(selection),
        "holdout_evaluation": {
            "season": HOLDOUT_SEASON,
            "passed": True,
            "evaluated_at_utc": holdout["evaluated_at_utc"],
        },
        "training": training,
        "runtime_model": runtime_model,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["model_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(output_path, payload)
    return payload


def run_backtest(
    rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    cutoff_date: str = "2026-07-20",
    l2_values: Sequence[float] = DEFAULT_L2_VALUES,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    selection = select_model(materialized, l2_values=l2_values)
    write_json(destination / "selection.json", selection)
    holdout = evaluate_sealed_holdout(materialized, selection, state_path=destination / "holdout-state.json")
    write_json(destination / "holdout-evaluation.json", holdout)
    model = None
    if holdout["passed"]:
        model = freeze_model_v1(
            materialized,
            selection,
            holdout,
            cutoff_date=cutoff_date,
            output_path=destination / "model-v1.json",
        )
    status = {
        "schema_version": "historical-gate-v1",
        "created_at_utc": _now(),
        "passed": bool(holdout["passed"]),
        "model_frozen": model is not None,
        "selected_candidate": selection["selected"]["candidate"],
        "holdout_state": "used_once",
    }
    write_json(destination / "status.json", status)
    return status


__all__ = [
    "DEFAULT_L2_VALUES",
    "HOLDOUT_SEASON",
    "SELECTION_SEASONS",
    "evaluate_sealed_holdout",
    "freeze_model_v1",
    "run_backtest",
    "select_model",
]
