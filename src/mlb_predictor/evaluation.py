from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np


def _as_float_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    try:
        materialized: Any = values if isinstance(values, np.ndarray) else list(values)
        array = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}은 숫자로 구성된 1차원 입력이어야 합니다.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name}은 1차원 입력이어야 합니다.")
    return array


def _validated_inputs(
    y_true: Iterable[float],
    probabilities: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    targets = _as_float_vector(y_true, name="y_true")
    predicted_probabilities = _as_float_vector(probabilities, name="probabilities")

    if targets.size != predicted_probabilities.size:
        raise ValueError("y_true와 probabilities의 길이가 같아야 합니다.")
    if targets.size == 0:
        raise ValueError("평가할 입력이 비어 있습니다.")
    if not np.all(np.isfinite(targets)) or not np.all(np.isin(targets, (0.0, 1.0))):
        raise ValueError("y_true는 유한한 이진 타깃 0 또는 1만 포함해야 합니다.")
    if not np.all(np.isfinite(predicted_probabilities)):
        raise ValueError("probabilities는 유한한 값만 포함해야 합니다.")
    if np.any((predicted_probabilities < 0.0) | (predicted_probabilities > 1.0)):
        raise ValueError("probabilities는 0부터 1 사이여야 합니다.")

    return targets, predicted_probabilities


def _validated_bin_count(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
        raise ValueError("n_bins는 1 이상의 정수여야 합니다.")
    return int(n_bins)


def _metrics_from_validated(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    epsilon = np.finfo(float).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    log_loss = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
    predicted_classes = probabilities >= 0.5

    return {
        "log_loss": float(log_loss),
        "brier_score": float(np.mean(np.square(probabilities - targets))),
        "accuracy": float(np.mean(predicted_classes == targets)),
        "row_count": int(targets.size),
        "positive_rate": float(np.mean(targets)),
        "mean_probability": float(np.mean(probabilities)),
    }


def binary_classification_metrics(
    y_true: Iterable[float],
    probabilities: Iterable[float],
) -> dict[str, float | int]:
    """이진 타깃과 양성 클래스 확률의 핵심 평가 지표를 계산합니다."""

    targets, predicted_probabilities = _validated_inputs(y_true, probabilities)
    return _metrics_from_validated(targets, predicted_probabilities)


def _calibration_from_validated(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> list[dict[str, float | int | None]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Multiplication avoids decimal-edge drift from ``linspace`` (for example,
    # 0.3 must belong to the [0.3, 0.4) bin, not the preceding bin).
    bin_indices = np.floor(probabilities * n_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    result: list[dict[str, float | int | None]] = []

    for index in range(n_bins):
        mask = bin_indices == index
        count = int(np.count_nonzero(mask))
        result.append(
            {
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "count": count,
                "mean_probability": float(np.mean(probabilities[mask])) if count else None,
                "observed_rate": float(np.mean(targets[mask])) if count else None,
            }
        )

    return result


def calibration_bins(
    y_true: Iterable[float],
    probabilities: Iterable[float],
    n_bins: int = 10,
) -> list[dict[str, float | int | None]]:
    """동일 너비 구간별 평균 예측 확률과 실제 양성률을 반환합니다."""

    bin_count = _validated_bin_count(n_bins)
    targets, predicted_probabilities = _validated_inputs(y_true, probabilities)
    return _calibration_from_validated(targets, predicted_probabilities, bin_count)


def evaluate_prediction_set(
    y_true: Iterable[float],
    probabilities: Iterable[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """한 예측 집합의 지표와 보정 구간을 함께 계산합니다."""

    bin_count = _validated_bin_count(n_bins)
    targets, predicted_probabilities = _validated_inputs(y_true, probabilities)
    return {
        "metrics": _metrics_from_validated(targets, predicted_probabilities),
        "calibration": _calibration_from_validated(targets, predicted_probabilities, bin_count),
    }
