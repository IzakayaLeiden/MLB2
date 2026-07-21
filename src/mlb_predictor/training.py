from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import evaluate_prediction_set
from .io import read_rows, sha256_file, write_json
from .modeling import (
    DEFAULT_FEATURE_SPECS,
    ModelBundle,
    PlattCalibrator,
    StandardizedLogisticRegression,
    extract_feature_matrix,
)
from .quality import BLOCKED_FEATURE_COLUMNS


_CUTOFF_POLICY = "prior_official_date_only"
_DEFAULT_OUTPUT = "raw_logistic_home_win_probability"
_CALIBRATED_OUTPUT = "platt_calibrated_home_win_probability"
_PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ChronologicalDateSplit:
    """공식 날짜를 쪼개지 않는 시간 순서 학습/검증/테스트 분할입니다."""

    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
    test_rows: tuple[dict[str, Any], ...]
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    target_train_fraction: float
    target_validation_fraction: float
    squared_fraction_error: float

    @property
    def train(self) -> tuple[dict[str, Any], ...]:
        return self.train_rows

    @property
    def validation(self) -> tuple[dict[str, Any], ...]:
        return self.validation_rows

    @property
    def test(self) -> tuple[dict[str, Any], ...]:
        return self.test_rows

    def to_metadata(self) -> dict[str, Any]:
        row_counts = {
            "train": len(self.train_rows),
            "validation": len(self.validation_rows),
            "test": len(self.test_rows),
        }
        total_rows = sum(row_counts.values())
        target_test_fraction = 1.0 - self.target_train_fraction - self.target_validation_fraction
        return {
            "strategy": "chronological_official_date_atomic",
            "date_atomic": True,
            "selection_objective": "minimum_squared_row_fraction_error_over_all_two_boundary_pairs",
            "target_fractions": {
                "train": self.target_train_fraction,
                "validation": self.target_validation_fraction,
                "test": target_test_fraction,
            },
            "actual_fractions": {
                name: count / total_rows for name, count in row_counts.items()
            },
            "row_counts": row_counts,
            "date_counts": {
                "train": len(self.train_dates),
                "validation": len(self.validation_dates),
                "test": len(self.test_dates),
            },
            "date_ranges": {
                "train": {"start": self.train_dates[0], "end": self.train_dates[-1]},
                "validation": {
                    "start": self.validation_dates[0],
                    "end": self.validation_dates[-1],
                },
                "test": {"start": self.test_dates[0], "end": self.test_dates[-1]},
            },
            "dates": {
                "train": list(self.train_dates),
                "validation": list(self.validation_dates),
                "test": list(self.test_dates),
            },
            "squared_fraction_error": self.squared_fraction_error,
        }


# 짧고 일반적인 이름도 제공하되 직렬화 계약에는 위의 명시적 이름을 사용합니다.
ChronologicalSplit = ChronologicalDateSplit
TrainingSplit = ChronologicalDateSplit


def _validated_fraction(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}은 0과 1 사이의 유한한 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name}은 0과 1 사이의 유한한 숫자여야 합니다.")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name}은 1 이상의 정수여야 합니다.")
    return int(value)


def _canonical_official_date(value: Any, *, row_index: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"행 {row_index}의 official_date는 YYYY-MM-DD ISO 날짜여야 합니다.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"행 {row_index}의 official_date는 YYYY-MM-DD ISO 날짜여야 합니다: {value!r}"
        ) from exc
    if parsed.isoformat() != value:
        raise ValueError(f"행 {row_index}의 official_date는 YYYY-MM-DD ISO 날짜여야 합니다: {value!r}")
    return value


def _materialize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"행 {row_index}는 열 이름을 가진 매핑이어야 합니다.")
        materialized.append(dict(row))
    return materialized


def chronological_date_split(
    rows: Iterable[Mapping[str, Any]],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> ChronologicalDateSplit:
    """모든 두 날짜 경계를 비교해 목표 행 비율에 가장 가까운 분할을 고릅니다."""

    target_train = _validated_fraction(train_fraction, name="train_fraction")
    target_validation = _validated_fraction(validation_fraction, name="validation_fraction")
    if target_train + target_validation >= 1.0:
        raise ValueError("train_fraction과 validation_fraction의 합은 1보다 작아야 합니다.")

    materialized = _materialize_rows(rows)
    if not materialized:
        raise ValueError("날짜 분할할 학습 행이 비어 있습니다.")

    rows_by_date: dict[str, list[dict[str, Any]]] = {}
    for row_index, row in enumerate(materialized):
        official_date = _canonical_official_date(row.get("official_date"), row_index=row_index)
        rows_by_date.setdefault(official_date, []).append(row)

    for date_rows in rows_by_date.values():
        date_rows.sort(
            key=lambda row: (
                str(row.get("game_start_utc") or ""),
                str(row.get("game_id") or ""),
            )
        )

    ordered_dates = tuple(sorted(rows_by_date))
    if len(ordered_dates) < 3:
        raise ValueError(
            "시간 순서 train/validation/test 분할에는 서로 다른 official_date가 최소 3개 필요합니다."
        )

    counts = [len(rows_by_date[value]) for value in ordered_dates]
    cumulative = [0]
    for count in counts:
        cumulative.append(cumulative[-1] + count)
    total_rows = cumulative[-1]
    target_test = 1.0 - target_train - target_validation

    best: tuple[tuple[float, float, float, int, int], int, int, float] | None = None
    for train_boundary in range(1, len(ordered_dates) - 1):
        for validation_boundary in range(train_boundary + 1, len(ordered_dates)):
            train_count = cumulative[train_boundary]
            validation_count = cumulative[validation_boundary] - train_count
            test_count = total_rows - cumulative[validation_boundary]
            actual = (
                train_count / total_rows,
                validation_count / total_rows,
                test_count / total_rows,
            )
            squared_error = sum(
                (observed - target) ** 2
                for observed, target in zip(
                    actual,
                    (target_train, target_validation, target_test),
                    strict=True,
                )
            )
            # 동일 오차이면 train, validation 목표 오차와 더 이른 경계 순으로 결정합니다.
            key = (
                squared_error,
                abs(actual[0] - target_train),
                abs(actual[1] - target_validation),
                train_boundary,
                validation_boundary,
            )
            candidate = (key, train_boundary, validation_boundary, squared_error)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:  # 최소 날짜 수 검증 덕분에 도달하지 않지만 계약을 명시합니다.
        raise ValueError("비어 있지 않은 세 분할을 만들 날짜 경계를 찾지 못했습니다.")

    _, train_boundary, validation_boundary, squared_error = best
    train_dates = ordered_dates[:train_boundary]
    validation_dates = ordered_dates[train_boundary:validation_boundary]
    test_dates = ordered_dates[validation_boundary:]

    def rows_for(dates: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        return tuple(row for value in dates for row in rows_by_date[value])

    return ChronologicalDateSplit(
        train_rows=rows_for(train_dates),
        validation_rows=rows_for(validation_dates),
        test_rows=rows_for(test_dates),
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
        target_train_fraction=target_train,
        target_validation_fraction=target_validation,
        squared_fraction_error=float(squared_error),
    )


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _raise_training_issues(issues: list[str]) -> None:
    if not issues:
        return
    shown = issues[:20]
    suffix = f"; 그 밖의 오류 {len(issues) - len(shown)}개" if len(issues) > len(shown) else ""
    raise ValueError("학습 행 검증 실패: " + "; ".join(shown) + suffix)


def validate_training_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> ChronologicalDateSplit:
    """학습 안전 계약을 검증하고 실제로 사용할 시간 순서 분할을 반환합니다."""

    materialized = _materialize_rows(rows)
    if not materialized:
        raise ValueError("학습 행 검증 실패: 입력 데이터가 비어 있습니다.")

    issues: list[str] = []
    seen_game_ids: dict[Any, int] = {}
    valid_dates: dict[int, date] = {}
    distinct_dates: set[str] = set()

    available_columns = {
        str(column).lower()
        for row in materialized
        for column in row
    }
    blocked_columns = sorted(available_columns & BLOCKED_FEATURE_COLUMNS)
    if blocked_columns:
        issues.append(f"결과 누수 위험 열이 포함되었습니다: {blocked_columns}")

    for row_index, row in enumerate(materialized):
        game_id = row.get("game_id")
        if (
            isinstance(game_id, bool)
            or not isinstance(game_id, Integral)
            or int(game_id) < 1
        ):
            issues.append(f"행 {row_index}의 game_id는 양의 정수여야 합니다")
        else:
            try:
                prior_index = seen_game_ids.get(game_id)
                if prior_index is not None:
                    issues.append(
                        f"중복 game_id={game_id!r} (행 {prior_index}, {row_index})"
                    )
                else:
                    seen_game_ids[game_id] = row_index
            except TypeError:
                issues.append(f"행 {row_index}의 game_id는 해시 가능한 값이어야 합니다")

        official_value = row.get("official_date")
        try:
            canonical_date = _canonical_official_date(official_value, row_index=row_index)
            valid_dates[row_index] = date.fromisoformat(canonical_date)
            distinct_dates.add(canonical_date)
        except ValueError as exc:
            issues.append(str(exc))

        target = row.get("home_win")
        if isinstance(target, bool) or not isinstance(target, Integral) or int(target) not in {0, 1}:
            issues.append(f"행 {row_index}의 home_win은 이진 정수 0 또는 1이어야 합니다")

        if row.get("feature_cutoff_policy") != _CUTOFF_POLICY:
            issues.append(
                f"행 {row_index}의 feature_cutoff_policy는 {_CUTOFF_POLICY!r}여야 합니다"
            )

        current_date = valid_dates.get(row_index)
        for side in ("home", "away"):
            field = f"{side}_history_through_date"
            history_value = row.get(field)
            if _is_null(history_value):
                continue
            if not isinstance(history_value, str):
                issues.append(f"행 {row_index}의 {field}는 YYYY-MM-DD ISO 날짜 또는 null이어야 합니다")
                continue
            try:
                history_date = date.fromisoformat(history_value)
            except ValueError:
                issues.append(f"행 {row_index}의 {field}는 YYYY-MM-DD ISO 날짜 또는 null이어야 합니다")
                continue
            if history_date.isoformat() != history_value:
                issues.append(f"행 {row_index}의 {field}는 YYYY-MM-DD ISO 날짜 또는 null이어야 합니다")
            elif current_date is not None and history_date >= current_date:
                issues.append(
                    f"행 {row_index}의 {field}={history_value}는 official_date={official_value}보다 엄격히 이전이어야 합니다"
                )

        elo_probability = row.get("elo_expected_home_win_probability")
        if (
            isinstance(elo_probability, bool)
            or not isinstance(elo_probability, Real)
            or not math.isfinite(float(elo_probability))
            or not 0.0 <= float(elo_probability) <= 1.0
        ):
            issues.append(
                f"행 {row_index}의 elo_expected_home_win_probability는 0~1의 유한한 숫자여야 합니다"
            )

    if len(distinct_dates) < 3:
        issues.append("서로 다른 official_date가 최소 3개 필요합니다")

    required_sources = {spec.source for spec in DEFAULT_FEATURE_SPECS}
    for row_index, row in enumerate(materialized):
        missing_sources = sorted(source for source in required_sources if source not in row)
        if missing_sources:
            issues.append(f"행 {row_index}의 기본 모델 피처가 누락되었습니다: {missing_sources}")

    if not any("기본 모델 피처가 누락" in issue for issue in issues):
        try:
            matrix = extract_feature_matrix(materialized, DEFAULT_FEATURE_SPECS)
            if not np.all(np.isfinite(matrix)):
                issues.append("DEFAULT_FEATURE_SPECS로 만든 모델 입력은 모두 유한해야 합니다")
        except ValueError as exc:
            issues.append(f"DEFAULT_FEATURE_SPECS 모델 입력 오류: {exc}")

    _raise_training_issues(issues)

    split = chronological_date_split(
        materialized,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    class_issues: list[str] = []
    for name, partition in (
        ("train", split.train_rows),
        ("validation", split.validation_rows),
    ):
        classes = sorted({int(row["home_win"]) for row in partition})
        if classes != [0, 1]:
            class_issues.append(
                f"{name} 분할에는 home_win 0과 1이 모두 필요합니다; 관측 클래스={classes}"
            )
    _raise_training_issues(class_issues)
    return split


def _target(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    return [int(row["home_win"]) for row in rows]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def train_model_artifacts(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    l2: float = 1.0,
    n_bins: int = 10,
) -> dict[str, Any]:
    """누수 없는 시간 분할로 모델을 학습하고 세 JSON 산출물을 원자적으로 공개합니다."""

    source = Path(features_path)
    destination = Path(output_dir)
    if os.path.lexists(destination):
        raise FileExistsError(f"출력 디렉터리가 이미 존재하여 덮어쓰지 않습니다: {destination}")
    bin_count = _positive_integer(n_bins, name="n_bins")

    dataset_sha256_before = sha256_file(source)
    rows = read_rows(source)
    dataset_sha256_after = sha256_file(source)
    if dataset_sha256_before != dataset_sha256_after:
        raise RuntimeError("학습 입력 파일이 읽는 동안 변경되었습니다.")

    split = validate_training_rows(
        rows,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    feature_names = [spec.name for spec in DEFAULT_FEATURE_SPECS]
    train_matrix = extract_feature_matrix(split.train_rows, DEFAULT_FEATURE_SPECS)
    validation_matrix = extract_feature_matrix(split.validation_rows, DEFAULT_FEATURE_SPECS)
    test_matrix = extract_feature_matrix(split.test_rows, DEFAULT_FEATURE_SPECS)
    train_target = _target(split.train_rows)
    validation_target = _target(split.validation_rows)
    test_target = _target(split.test_rows)

    logistic = StandardizedLogisticRegression(l2=l2).fit(
        train_matrix,
        train_target,
        feature_names=feature_names,
    )
    validation_raw_probability = logistic.predict_proba(validation_matrix)
    calibrator = PlattCalibrator().fit(validation_raw_probability, validation_target)
    bundle = ModelBundle(DEFAULT_FEATURE_SPECS, logistic, calibrator)

    test_raw_probability = logistic.predict_proba(test_matrix)
    test_calibrated_probability = calibrator.predict(test_raw_probability)
    train_positive_rate = float(np.mean(np.asarray(train_target, dtype=float)))
    constant_probability = np.full(len(test_target), train_positive_rate, dtype=float)
    elo_probability = np.asarray(
        [float(row["elo_expected_home_win_probability"]) for row in split.test_rows],
        dtype=float,
    )
    test_results = {
        "constant_home_rate": evaluate_prediction_set(
            test_target, constant_probability, n_bins=bin_count
        ),
        "elo_baseline": evaluate_prediction_set(
            test_target, elo_probability, n_bins=bin_count
        ),
        "logistic_raw": evaluate_prediction_set(
            test_target, test_raw_probability, n_bins=bin_count
        ),
        "logistic_platt_calibrated": evaluate_prediction_set(
            test_target, test_calibrated_probability, n_bins=bin_count
        ),
    }

    created_at_utc = _utc_now()
    split_metadata = split.to_metadata()
    dataset_metadata = {
        "source_file": source.name,
        "sha256": dataset_sha256_after,
        "bytes": source.stat().st_size,
        "row_count": len(rows),
    }
    model_payload = {
        "schema_version": "mlb-win-probability-model-v1",
        "created_at_utc": created_at_utc,
        "default_output": _DEFAULT_OUTPUT,
        "feature_cutoff_policy": _CUTOFF_POLICY,
        "runtime_contract": {
            "format": "portable-json",
            "implementation": "portable-json",
            "schema_version": "mlb-win-probability-runtime-v1",
            "input": "mapping_with_feature_spec_sources",
            "output": _DEFAULT_OUTPUT,
            "available_outputs": [
                _DEFAULT_OUTPUT,
                _CALIBRATED_OUTPUT,
            ],
            "raw_formula": (
                "sigmoid(intercept + sum(coefficients[i] * "
                "((x[i] - mean[i]) / scale[i])))"
            ),
            "calibrated_formula": (
                "sigmoid(calibrator.intercept + calibrator.coefficient * "
                "logit(clamp(raw_probability, epsilon, 1 - epsilon)))"
            ),
            "probability_epsilon": _PROBABILITY_EPSILON,
        },
        "runtime_model": bundle.to_dict(),
        "dataset": dataset_metadata,
        "split": split_metadata,
        "training": {
            "fit_partition": "train_only",
            "algorithm": "standardized_l2_logistic_regression",
            "l2": float(logistic.l2),
            "calibration": {
                "method": "platt_scaling",
                "fit_partition": "validation_raw_probability_only",
                "status": "evaluation_only_not_selected_as_default",
            },
        },
    }
    small_sample_warning = (
        f"경고: 테스트 표본은 {len(split.test_rows)}행, {len(split.test_dates)}개 날짜뿐입니다. "
        "이 결과는 배선 확인용 소표본 평가이며 성능, 일반화, 수익성 또는 운영 준비의 증거가 아닙니다."
    )
    evaluation_payload = {
        "schema_version": "mlb-win-probability-evaluation-v1",
        "created_at_utc": created_at_utc,
        "assessment": "sample_only_not_performance_evidence",
        "performance_gate_passed": None,
        "test_set_usage": "evaluation_only_not_used_for_selection_or_tuning",
        "calibration_fit": "validation_raw_probability_only",
        "default_output_basis": "raw_probability_until_calibration_has_independent_support",
        "dataset": dataset_metadata,
        "split": split_metadata,
        "warnings": [small_sample_warning],
        "small_sample_warning": small_sample_warning,
        "test_results": test_results,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        model_path = write_json(staging / "model.json", model_payload)
        evaluation_path = write_json(staging / "evaluation.json", evaluation_payload)
        manifest_payload = {
            "schema_version": "mlb-win-probability-artifact-manifest-v1",
            "created_at_utc": created_at_utc,
            "status": "succeeded",
            "build_status": "succeeded",
            "artifacts_valid": True,
            "performance_gate_passed": None,
            "deployment_approved": False,
            "assessment": "sample_only_not_performance_evidence",
            "dataset": dataset_metadata,
            "artifacts": {
                "model.json": {
                    "sha256": sha256_file(model_path),
                    "bytes": model_path.stat().st_size,
                },
                "evaluation.json": {
                    "sha256": sha256_file(evaluation_path),
                    "bytes": evaluation_path.stat().st_size,
                },
            },
        }
        write_json(staging / "manifest.json", manifest_payload)

        if os.path.lexists(destination):
            raise FileExistsError(f"출력 디렉터리가 이미 존재하여 덮어쓰지 않습니다: {destination}")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        **manifest_payload,
        "manifest_path": str(destination / "manifest.json"),
    }


__all__ = [
    "ChronologicalDateSplit",
    "ChronologicalSplit",
    "TrainingSplit",
    "chronological_date_split",
    "train_model_artifacts",
    "validate_training_rows",
]
