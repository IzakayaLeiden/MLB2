from __future__ import annotations

import math
import platform
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evaluation import evaluate_prediction_set
from .io import sha256_file, write_json, write_rows_csv
from .modeling import (
    DEFAULT_FEATURE_SPECS,
    PlattCalibrator,
    StandardizedLogisticRegression,
    extract_feature_matrix,
)


AUDIT_SEASONS = (2022, 2023, 2024, 2025)
PROBABILITY_COLUMNS = (
    "p_constant",
    "p_elo",
    "p_logistic_raw",
    "p_platt_base_raw",
    "p_logistic_platt",
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _target(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["home_win"]) for row in rows], dtype=float)


def _fit_logistic(rows: Sequence[Mapping[str, Any]], l2: float) -> StandardizedLogisticRegression:
    return StandardizedLogisticRegression(l2=l2).fit(
        extract_feature_matrix(rows, DEFAULT_FEATURE_SPECS),
        _target(rows),
        feature_names=[spec.name for spec in DEFAULT_FEATURE_SPECS],
    )


def generate_historical_audit_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    l2: float,
    seasons: Sequence[int] = AUDIT_SEASONS,
) -> list[dict[str, Any]]:
    """공식 선택 규칙으로 시즌별 out-of-fold 경기 확률을 재생합니다."""

    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2는 0 이상의 유한한 값이어야 합니다.")
    materialized = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["official_date"]), int(row["game_id"])),
    )
    audit_rows: list[dict[str, Any]] = []
    for season in seasons:
        validation = [row for row in materialized if int(row["season"]) == int(season)]
        prior = [row for row in materialized if int(row["season"]) < int(season)]
        calibration = [row for row in materialized if int(row["season"]) == int(season) - 1]
        pre_calibration = [row for row in materialized if int(row["season"]) < int(season) - 1]
        if not validation or not prior or not calibration or not pre_calibration:
            raise ValueError(f"감사 예측에 필요한 시즌 데이터가 부족합니다: {season}")

        constant = np.full(len(validation), float(np.mean(_target(prior))))
        elo = np.asarray([float(row["elo_expected_home_win_probability"]) for row in validation])

        raw_candidate = _fit_logistic(prior, l2)
        raw_probability = raw_candidate.predict_proba(
            extract_feature_matrix(validation, DEFAULT_FEATURE_SPECS)
        )

        platt_base = _fit_logistic(pre_calibration, l2)
        calibration_raw = platt_base.predict_proba(
            extract_feature_matrix(calibration, DEFAULT_FEATURE_SPECS)
        )
        calibrator = PlattCalibrator().fit(calibration_raw, _target(calibration))
        platt_base_probability = platt_base.predict_proba(
            extract_feature_matrix(validation, DEFAULT_FEATURE_SPECS)
        )
        platt_probability = calibrator.predict(platt_base_probability)

        split = "sealed_holdout" if int(season) == 2025 else "model_selection"
        for index, row in enumerate(validation):
            audit_rows.append(
                {
                    "game_id": int(row["game_id"]),
                    "official_date": str(row["official_date"]),
                    "season": int(season),
                    "split": split,
                    "home_win": int(row["home_win"]),
                    "p_constant": float(constant[index]),
                    "p_elo": float(elo[index]),
                    "p_logistic_raw": float(raw_probability[index]),
                    "p_platt_base_raw": float(platt_base_probability[index]),
                    "p_logistic_platt": float(platt_probability[index]),
                    "eligible": True,
                    "exclusion_reason": None,
                }
            )
    return audit_rows


def _metric_loss(target: np.ndarray, probability: np.ndarray, metric: str) -> np.ndarray:
    if metric == "log_loss":
        epsilon = np.finfo(float).eps
        clipped = np.clip(probability, epsilon, 1.0 - epsilon)
        return -(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
    if metric == "brier_score":
        return np.square(probability - target)
    raise ValueError(f"지원하지 않는 paired metric입니다: {metric}")


def paired_date_block_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    challenger: str,
    baseline: str,
    metric: str,
    iterations: int = 10_000,
    seed: int = 20260721,
) -> dict[str, Any]:
    """같은 공식 날짜를 한 블록으로 재표집해 paired 손실 차이를 추정합니다."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("paired bootstrap 입력이 비어 있습니다.")
    if iterations < 100:
        raise ValueError("bootstrap iterations는 100 이상이어야 합니다.")
    by_date: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_date[str(row["official_date"])].append(row)
    dates = sorted(by_date)
    block_delta_sums: list[float] = []
    block_counts: list[int] = []
    for official_date in dates:
        block = by_date[official_date]
        target = np.asarray([int(row["home_win"]) for row in block], dtype=float)
        challenger_probability = np.asarray([float(row[challenger]) for row in block], dtype=float)
        baseline_probability = np.asarray([float(row[baseline]) for row in block], dtype=float)
        delta = _metric_loss(target, challenger_probability, metric) - _metric_loss(
            target, baseline_probability, metric
        )
        block_delta_sums.append(float(np.sum(delta)))
        block_counts.append(len(block))

    delta_sums = np.asarray(block_delta_sums, dtype=float)
    counts = np.asarray(block_counts, dtype=float)
    point_estimate = float(np.sum(delta_sums) / np.sum(counts))
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = rng.integers(0, len(dates), size=len(dates))
        samples[index] = float(np.sum(delta_sums[selected]) / np.sum(counts[selected]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    probability_better = float(np.mean(samples < 0.0))
    two_sided = float(min(1.0, 2.0 * min(probability_better, 1.0 - probability_better)))
    return {
        "metric": metric,
        "challenger": challenger,
        "baseline": baseline,
        "delta_definition": "challenger_minus_baseline_lower_is_better",
        "point_estimate": point_estimate,
        "confidence_interval_95": [float(lower), float(upper)],
        "bootstrap_probability_challenger_better": probability_better,
        "two_sided_bootstrap_p_value": two_sided,
        "iterations": int(iterations),
        "seed": int(seed),
        "date_blocks": len(dates),
        "rows": len(materialized),
    }


def _evaluation(rows: Sequence[Mapping[str, Any]], probability_column: str) -> dict[str, Any]:
    return evaluate_prediction_set(
        [int(row["home_win"]) for row in rows],
        [float(row[probability_column]) for row in rows],
    )


def _section(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    evaluations = {column: _evaluation(rows, column) for column in PROBABILITY_COLUMNS}
    comparisons: dict[str, Any] = {}
    for challenger, baseline in (
        ("p_logistic_platt", "p_elo"),
        ("p_logistic_platt", "p_constant"),
        ("p_logistic_raw", "p_elo"),
        ("p_logistic_platt", "p_platt_base_raw"),
    ):
        key = f"{challenger}_vs_{baseline}"
        comparisons[key] = {
            metric: paired_date_block_bootstrap(
                rows,
                challenger=challenger,
                baseline=baseline,
                metric=metric,
                iterations=iterations,
                seed=seed,
            )
            for metric in ("log_loss", "brier_score")
        }
    return {
        "rows": len(rows),
        "date_blocks": len({str(row["official_date"]) for row in rows}),
        "evaluations": evaluations,
        "paired_date_block_bootstrap": comparisons,
    }


def build_audit_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    iterations: int = 10_000,
    seed: int = 20260721,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    seasons = sorted({int(row["season"]) for row in materialized})
    season_sections = {
        str(season): _section(
            [row for row in materialized if int(row["season"]) == season],
            iterations=iterations,
            seed=seed + season,
        )
        for season in seasons
    }
    selection_rows = [row for row in materialized if int(row["season"]) in {2022, 2023, 2024}]
    holdout_rows = [row for row in materialized if int(row["season"]) == 2025]
    if not selection_rows or not holdout_rows:
        raise ValueError("감사 보고서에는 2022~2024 선택 구간과 2025 홀드아웃이 모두 필요합니다.")
    return {
        "schema_version": "model-audit-v1",
        "created_at_utc": created_at_utc or _now(),
        "probability_columns": list(PROBABILITY_COLUMNS),
        "raw_probability_definitions": {
            "p_logistic_raw": "공식 raw LR 후보: 검증 시즌 이전 전체 시즌으로 학습",
            "p_platt_base_raw": "Platt 후보의 보정 전 확률: 직전 보정 시즌보다 앞선 시즌으로 학습",
            "p_logistic_platt": "p_platt_base_raw를 직전 시즌으로 Platt 보정",
        },
        "season_results": season_sections,
        "combined_model_selection_2022_2024": _section(
            selection_rows, iterations=iterations, seed=seed + 2024
        ),
        "sealed_holdout_2025": _section(
            holdout_rows, iterations=iterations, seed=seed + 2025
        ),
    }


def write_audit_bundle(
    *,
    feature_rows: Iterable[Mapping[str, Any]],
    features_path: str | Path,
    games_path: str | Path,
    exclusions_path: str | Path,
    model: Mapping[str, Any],
    model_path: str | Path,
    output_dir: str | Path,
    code_revision: str,
    iterations: int = 10_000,
    seed: int = 20260721,
) -> dict[str, Any]:
    if model.get("model_type") != "logistic_platt":
        raise ValueError("model-v1 감사 번들은 logistic_platt 모델만 지원합니다.")
    l2 = float(model["training"]["l2"])
    materialized_features = [dict(row) for row in feature_rows]
    predictions = generate_historical_audit_rows(materialized_features, l2=l2)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selection_path = write_rows_csv(
        destination / "selection-predictions.csv",
        [row for row in predictions if row["split"] == "model_selection"],
    )
    holdout_path = write_rows_csv(
        destination / "holdout-predictions.csv",
        [row for row in predictions if row["split"] == "sealed_holdout"],
    )
    report = build_audit_report(predictions, iterations=iterations, seed=seed)
    report_path = write_json(destination / "audit-report.json", report)
    features_copy = destination / "pregame-features.parquet"
    games_copy = destination / "normalized-games.parquet"
    exclusions_copy = destination / "exclusions.csv"
    model_copy = destination / "model-v1.json"
    shutil.copy2(features_path, features_copy)
    shutil.copy2(games_path, games_copy)
    shutil.copy2(exclusions_path, exclusions_copy)
    shutil.copy2(model_path, model_copy)
    environment_path = destination / "environment-lock.txt"
    environment_path.write_text(
        "\n".join(
            (
                f"python=={platform.python_version()}",
                f"numpy=={version('numpy')}",
                f"pandas=={version('pandas')}",
                f"pyarrow=={version('pyarrow')}",
                "",
            )
        ),
        encoding="utf-8",
    )
    artifact_paths = (
        selection_path,
        holdout_path,
        report_path,
        features_copy,
        games_copy,
        exclusions_copy,
        model_copy,
        environment_path,
    )
    manifest = {
        "schema_version": "model-audit-manifest-v1",
        "created_at_utc": report["created_at_utc"],
        "code_revision": code_revision,
        "model": {
            "version": model["model_version"],
            "canonical_sha256": model["model_sha256"],
            "file_sha256": sha256_file(model_path),
            "selection_fingerprint": model["selection_fingerprint"],
        },
        "input": {
            "features_file_sha256": sha256_file(features_path),
            "games_file_sha256": sha256_file(games_path),
            "exclusions_file_sha256": sha256_file(exclusions_path),
            "rows": len(materialized_features),
        },
        "config": {"l2": l2, "bootstrap_iterations": iterations, "seed": seed},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
    }
    manifest_path = write_json(destination / "manifest.json", manifest)
    sums_path = destination / "sha256sums.txt"
    sums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (*artifact_paths, manifest_path)
        ),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "AUDIT_SEASONS",
    "PROBABILITY_COLUMNS",
    "build_audit_report",
    "generate_historical_audit_rows",
    "paired_date_block_bootstrap",
    "write_audit_bundle",
]
