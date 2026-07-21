from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Self

import numpy as np


_FEATURE_TRANSFORMS = frozenset({"numeric", "equals"})
_PROBABILITY_EPSILON = 1e-12


def _json_scalar(
    value: Any,
    *,
    name: str,
    allow_none: bool,
) -> str | int | float | bool | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name}은 필수값입니다.")
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        try:
            result = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}은 JSON 호환 유한 스칼라여야 합니다.") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name}은 JSON 호환 유한 스칼라여야 합니다.")
        return result
    raise ValueError(f"{name}은 JSON 호환 유한 스칼라여야 합니다.")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """한 모델 입력 피처를 원본 행에서 계산하는 직렬화 가능한 명세입니다."""

    name: str
    source: str
    transform: str
    value: str | int | float | bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("피처 name은 비어 있지 않은 문자열이어야 합니다.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("피처 source는 비어 있지 않은 문자열이어야 합니다.")
        if not isinstance(self.transform, str) or self.transform not in _FEATURE_TRANSFORMS:
            raise ValueError(f"지원하지 않는 피처 transform입니다: {self.transform!r}")
        normalized_value = _json_scalar(
            self.value,
            name="피처 value",
            allow_none=True,
        )
        object.__setattr__(self, "value", normalized_value)
        if self.transform == "numeric" and self.value is not None:
            raise ValueError("numeric 피처의 value는 None이어야 합니다.")
        if self.transform == "equals" and self.value is None:
            raise ValueError("equals 피처에는 비교할 value가 필요합니다.")

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "name": self.name,
            "source": self.source,
            "transform": self.transform,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("피처 명세는 매핑이어야 합니다.")
        try:
            return cls(
                name=payload["name"],
                source=payload["source"],
                transform=payload["transform"],
                value=payload["value"],
            )
        except KeyError as exc:
            raise ValueError(f"피처 명세 필드가 누락되었습니다: {exc.args[0]}") from exc


_NUMERIC_FEATURE_NAMES = (
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
)

DEFAULT_FEATURE_SPECS: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(name=name, source=name, transform="numeric", value=None)
    for name in _NUMERIC_FEATURE_NAMES
) + (
    FeatureSpec(
        name="is_night_game",
        source="day_night",
        transform="equals",
        value="night",
    ),
)


def _materialize_specs(specs: Sequence[FeatureSpec]) -> tuple[FeatureSpec, ...]:
    materialized = tuple(specs)
    if not materialized:
        raise ValueError("피처 명세가 비어 있습니다.")
    if not all(isinstance(spec, FeatureSpec) for spec in materialized):
        raise ValueError("specs는 FeatureSpec만 포함해야 합니다.")
    names = [spec.name for spec in materialized]
    if len(names) != len(set(names)):
        raise ValueError("피처 name은 중복될 수 없습니다.")
    return materialized


def _feature_value(row: Mapping[str, Any], spec: FeatureSpec, row_index: int) -> float:
    if spec.source not in row or row[spec.source] is None:
        raise ValueError(
            f"행 {row_index}에 필수 피처 값이 없습니다: {spec.source}"
        )
    raw_value = row[spec.source]
    if spec.transform == "equals":
        comparable = _json_scalar(
            raw_value,
            name=f"행 {row_index}의 피처 {spec.source}",
            allow_none=False,
        )
        return 1.0 if comparable == spec.value else 0.0

    try:
        value = float(raw_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"행 {row_index}의 피처 {spec.source}는 숫자여야 합니다."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"행 {row_index}의 피처 {spec.source}는 유한해야 합니다."
        )
    return value


def extract_feature_matrix(
    rows: Iterable[Mapping[str, Any]],
    specs: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPECS,
) -> np.ndarray:
    """행 목록을 명세 순서의 유한한 2차원 float 행렬로 변환합니다."""

    feature_specs = _materialize_specs(specs)
    matrix_rows: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"행 {row_index}는 매핑이어야 합니다.")
        matrix_rows.append(
            [_feature_value(row, spec, row_index) for spec in feature_specs]
        )
    if not matrix_rows:
        return np.empty((0, len(feature_specs)), dtype=float)
    return np.asarray(matrix_rows, dtype=float)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}은 유한한 숫자여야 합니다.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}은 유한한 숫자여야 합니다.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}은 유한한 숫자여야 합니다.")
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name}은 0 이상이어야 합니다.")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name}은 1 이상의 정수여야 합니다.")
    return int(value)


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name}은 양수여야 합니다.")
    return result


def _as_matrix(
    matrix: Iterable[Iterable[float]],
    *,
    name: str,
    expected_features: int | None = None,
    require_rows: bool = True,
) -> np.ndarray:
    try:
        materialized: Any = matrix if isinstance(matrix, np.ndarray) else list(matrix)
        array = np.asarray(materialized, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}은 숫자로 구성된 2차원 행렬이어야 합니다.") from exc
    if array.ndim != 2:
        raise ValueError(f"{name}은 2차원 행렬이어야 합니다.")
    if require_rows and array.shape[0] == 0:
        raise ValueError(f"{name}에는 하나 이상의 행이 필요합니다.")
    if array.shape[1] == 0:
        raise ValueError(f"{name}에는 하나 이상의 피처가 필요합니다.")
    if expected_features is not None and array.shape[1] != expected_features:
        raise ValueError(
            f"{name}의 피처 수가 모델과 다릅니다: "
            f"expected={expected_features}, actual={array.shape[1]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}은 유한한 값만 포함해야 합니다.")
    return array


def _as_target(target: Iterable[float], *, expected_rows: int) -> np.ndarray:
    try:
        materialized: Any = target if isinstance(target, np.ndarray) else list(target)
        array = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target은 이진 값으로 구성된 1차원 입력이어야 합니다.") from exc
    if array.ndim != 1:
        raise ValueError("target은 1차원 입력이어야 합니다.")
    if array.size != expected_rows:
        raise ValueError("matrix 행 수와 target 길이가 같아야 합니다.")
    if not np.all(np.isfinite(array)) or not np.all(np.isin(array, (0.0, 1.0))):
        raise ValueError("target은 유한한 이진 값 0 또는 1만 포함해야 합니다.")
    if np.unique(array).size != 2:
        raise ValueError("target에는 0과 1이 모두 있어야 합니다.")
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _penalized_logistic_loss(
    design: np.ndarray,
    target: np.ndarray,
    parameters: np.ndarray,
    penalty: np.ndarray,
) -> float:
    scores = design @ parameters
    data_loss = np.mean(np.logaddexp(0.0, scores) - target * scores)
    regularization = 0.5 * np.dot(penalty * parameters, parameters)
    return float(data_loss + regularization)


def _fit_logistic_parameters(
    design: np.ndarray,
    target: np.ndarray,
    *,
    penalty: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    positive_rate = float(np.mean(target))
    parameters = np.zeros(design.shape[1], dtype=float)
    parameters[0] = math.log(positive_rate / (1.0 - positive_rate))

    for _ in range(max_iter):
        scores = design @ parameters
        probabilities = _sigmoid(scores)
        gradient = design.T @ (probabilities - target) / target.size
        gradient += penalty * parameters
        if float(np.max(np.abs(gradient))) <= tolerance:
            break

        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * weights[:, np.newaxis]) / target.size
        hessian.flat[:: hessian.shape[0] + 1] += penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        current_loss = _penalized_logistic_loss(
            design, target, parameters, penalty
        )
        directional_decrease = float(np.dot(gradient, step))
        step_size = 1.0
        accepted = False
        for _ in range(40):
            candidate = parameters - step_size * step
            candidate_loss = _penalized_logistic_loss(
                design, target, candidate, penalty
            )
            if candidate_loss <= current_loss - 1e-4 * step_size * directional_decrease:
                parameters = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(step_size * step))) <= tolerance:
            break

    if not np.all(np.isfinite(parameters)):
        raise ValueError("유한한 로지스틱 회귀 계수를 학습하지 못했습니다.")
    return parameters


class StandardizedLogisticRegression:
    """학습 집합 통계만 사용하는 L2 정규화 이진 로지스틱 회귀입니다."""

    def __init__(
        self,
        *,
        l2: float = 1.0,
        max_iter: int = 100,
        tolerance: float = 1e-9,
    ) -> None:
        self.l2 = _nonnegative_float(l2, name="l2")
        self.max_iter = _positive_integer(max_iter, name="max_iter")
        self.tolerance = _positive_float(tolerance, name="tolerance")
        self.feature_names: tuple[str, ...] = ()
        self.mean: tuple[float, ...] = ()
        self.scale: tuple[float, ...] = ()
        self.coefficients: tuple[float, ...] = ()
        self.intercept: float | None = None

    def fit(
        self,
        matrix: Iterable[Iterable[float]],
        target: Iterable[float],
        feature_names: Sequence[str] | None = None,
        *,
        l2: float | None = None,
    ) -> Self:
        features = _as_matrix(matrix, name="matrix")
        targets = _as_target(target, expected_rows=features.shape[0])
        if l2 is not None:
            self.l2 = _nonnegative_float(l2, name="l2")

        if feature_names is None:
            names = tuple(f"feature_{index}" for index in range(features.shape[1]))
        else:
            names = tuple(feature_names)
            if len(names) != features.shape[1]:
                raise ValueError("feature_names 길이는 matrix 피처 수와 같아야 합니다.")
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError("feature_names는 비어 있지 않은 문자열이어야 합니다.")
            if len(names) != len(set(names)):
                raise ValueError("feature_names는 중복될 수 없습니다.")

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean = np.mean(features, axis=0)
            scale = np.std(features, axis=0)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("학습 표준화 통계를 유한하게 계산할 수 없습니다.")
        scale = np.where(scale > 0.0, scale, 1.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (features - mean) / scale
        if not np.all(np.isfinite(standardized)):
            raise ValueError("학습 피처를 유한하게 표준화할 수 없습니다.")
        design = np.column_stack((np.ones(features.shape[0]), standardized))
        penalty = np.concatenate(([0.0], np.full(features.shape[1], self.l2)))
        parameters = _fit_logistic_parameters(
            design,
            targets,
            penalty=penalty,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
        )

        self.feature_names = names
        self.mean = tuple(float(value) for value in mean)
        self.scale = tuple(float(value) for value in scale)
        self.intercept = float(parameters[0])
        self.coefficients = tuple(float(value) for value in parameters[1:])
        return self

    def _require_fitted(self) -> int:
        if self.intercept is None or not self.coefficients:
            raise ValueError("로지스틱 회귀 모델이 학습되지 않았습니다.")
        feature_count = len(self.coefficients)
        if not (
            len(self.feature_names) == len(self.mean) == len(self.scale) == feature_count
        ):
            raise ValueError("로지스틱 회귀 모델 상태가 일관되지 않습니다.")
        return feature_count

    def predict_proba(self, matrix: Iterable[Iterable[float]]) -> np.ndarray:
        feature_count = self._require_fitted()
        features = _as_matrix(
            matrix,
            name="matrix",
            expected_features=feature_count,
            require_rows=False,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (
                features - np.asarray(self.mean, dtype=float)
            ) / np.asarray(self.scale, dtype=float)
        if not np.all(np.isfinite(standardized)):
            raise ValueError("예측 피처를 유한하게 표준화할 수 없습니다.")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = self.intercept + standardized @ np.asarray(
                self.coefficients, dtype=float
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("유한한 로지스틱 회귀 점수를 계산할 수 없습니다.")
        probabilities = _sigmoid(scores)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("유한한 로지스틱 회귀 확률을 계산할 수 없습니다.")
        return probabilities

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "feature_names": list(self.feature_names),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "coefficients": [float(value) for value in self.coefficients],
            "intercept": float(self.intercept),
            "l2": float(self.l2),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("로지스틱 회귀 payload는 매핑이어야 합니다.")
        try:
            raw_names = payload["feature_names"]
            if isinstance(raw_names, (str, bytes)):
                raise ValueError("feature_names는 문자열 목록이어야 합니다.")
            names = tuple(raw_names)
            mean = tuple(
                _finite_float(value, name="mean") for value in payload["mean"]
            )
            scale = tuple(
                _positive_float(value, name="scale") for value in payload["scale"]
            )
            coefficients = tuple(
                _finite_float(value, name="coefficients")
                for value in payload["coefficients"]
            )
            intercept = _finite_float(payload["intercept"], name="intercept")
            model = cls(l2=_nonnegative_float(payload["l2"], name="l2"))
        except (KeyError, TypeError) as exc:
            raise ValueError("로지스틱 회귀 payload가 올바르지 않습니다.") from exc
        if not coefficients or not (
            len(names) == len(mean) == len(scale) == len(coefficients)
        ):
            raise ValueError("로지스틱 회귀 payload의 피처 길이가 일치하지 않습니다.")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names는 비어 있지 않은 문자열이어야 합니다.")
        if len(names) != len(set(names)):
            raise ValueError("feature_names는 중복될 수 없습니다.")
        model.feature_names = names
        model.mean = mean
        model.scale = scale
        model.coefficients = coefficients
        model.intercept = intercept
        return model


def _as_probabilities(
    raw_probabilities: Iterable[float], *, require_values: bool
) -> np.ndarray:
    try:
        materialized: Any = (
            raw_probabilities
            if isinstance(raw_probabilities, np.ndarray)
            else list(raw_probabilities)
        )
        probabilities = np.asarray(materialized, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("raw_probabilities는 숫자 1차원 입력이어야 합니다.") from exc
    if probabilities.ndim != 1:
        raise ValueError("raw_probabilities는 1차원 입력이어야 합니다.")
    if require_values and probabilities.size == 0:
        raise ValueError("raw_probabilities가 비어 있습니다.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("raw_probabilities는 유한해야 합니다.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("raw_probabilities는 0부터 1 사이여야 합니다.")
    return probabilities


def _probability_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
    )
    return np.log(clipped) - np.log1p(-clipped)


class PlattCalibrator:
    """원시 확률의 logit에 1차 로지스틱 보정을 학습합니다."""

    def __init__(self, *, l2: float = 1e-6) -> None:
        self.l2 = _nonnegative_float(l2, name="l2")
        self.coefficient: float | None = None
        self.intercept: float | None = None

    def fit(
        self,
        raw_probabilities: Iterable[float],
        target: Iterable[float],
        l2: float | None = None,
    ) -> Self:
        probabilities = _as_probabilities(raw_probabilities, require_values=True)
        targets = _as_target(target, expected_rows=probabilities.size)
        if l2 is not None:
            self.l2 = _nonnegative_float(l2, name="l2")
        logits = _probability_logit(probabilities)
        design = np.column_stack((np.ones(probabilities.size), logits))
        parameters = _fit_logistic_parameters(
            design,
            targets,
            penalty=np.asarray([0.0, self.l2], dtype=float),
            max_iter=100,
            tolerance=1e-9,
        )
        self.intercept = float(parameters[0])
        self.coefficient = float(parameters[1])
        return self

    def _require_fitted(self) -> None:
        if self.intercept is None or self.coefficient is None:
            raise ValueError("Platt 보정기가 학습되지 않았습니다.")

    def predict(self, raw_probabilities: Iterable[float]) -> np.ndarray:
        self._require_fitted()
        probabilities = _as_probabilities(raw_probabilities, require_values=False)
        scores = self.intercept + self.coefficient * _probability_logit(probabilities)
        return _sigmoid(scores)

    def to_dict(self) -> dict[str, float]:
        self._require_fitted()
        return {
            "coefficient": float(self.coefficient),
            "intercept": float(self.intercept),
            "l2": float(self.l2),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("Platt payload는 매핑이어야 합니다.")
        try:
            calibrator = cls(l2=_nonnegative_float(payload["l2"], name="l2"))
            calibrator.coefficient = _finite_float(
                payload["coefficient"], name="coefficient"
            )
            calibrator.intercept = _finite_float(
                payload["intercept"], name="intercept"
            )
        except KeyError as exc:
            raise ValueError("Platt payload가 올바르지 않습니다.") from exc
        return calibrator


@dataclass(slots=True)
class ModelBundle:
    """피처 명세, 원시 모델, 확률 보정기를 하나의 계산 가능한 묶음으로 보존합니다."""

    feature_specs: Sequence[FeatureSpec]
    logistic: StandardizedLogisticRegression
    calibrator: PlattCalibrator

    def __post_init__(self) -> None:
        self.feature_specs = _materialize_specs(self.feature_specs)
        if not isinstance(self.logistic, StandardizedLogisticRegression):
            raise ValueError("logistic은 StandardizedLogisticRegression이어야 합니다.")
        if not isinstance(self.calibrator, PlattCalibrator):
            raise ValueError("calibrator는 PlattCalibrator이어야 합니다.")
        feature_count = self.logistic._require_fitted()
        self.calibrator._require_fitted()
        if feature_count != len(self.feature_specs):
            raise ValueError("피처 명세 수와 로지스틱 모델 피처 수가 다릅니다.")
        spec_names = tuple(spec.name for spec in self.feature_specs)
        if self.logistic.feature_names != spec_names:
            raise ValueError("피처 명세 이름/순서와 로지스틱 모델이 다릅니다.")

    def predict_matrix(
        self,
        matrix: Iterable[Iterable[float]],
        *,
        calibrated: bool = False,
    ) -> np.ndarray:
        raw_probability = self.logistic.predict_proba(matrix)
        if not isinstance(calibrated, bool):
            raise ValueError("calibrated는 bool이어야 합니다.")
        return self.calibrator.predict(raw_probability) if calibrated else raw_probability

    def predict_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        calibrated: bool = False,
    ) -> np.ndarray:
        return self.predict_matrix(
            extract_feature_matrix(rows, self.feature_specs),
            calibrated=calibrated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_specs": [spec.to_dict() for spec in self.feature_specs],
            "logistic": self.logistic.to_dict(),
            "calibrator": self.calibrator.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("모델 번들 payload는 매핑이어야 합니다.")
        try:
            feature_specs = tuple(
                FeatureSpec.from_dict(spec) for spec in payload["feature_specs"]
            )
            logistic = StandardizedLogisticRegression.from_dict(payload["logistic"])
            calibrator = PlattCalibrator.from_dict(payload["calibrator"])
        except (KeyError, TypeError) as exc:
            raise ValueError("모델 번들 payload가 올바르지 않습니다.") from exc
        return cls(feature_specs, logistic, calibrator)
