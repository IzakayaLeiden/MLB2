from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Mapping

import numpy as np

from .evaluation import evaluate_prediction_set


def evaluate_public_gate(
    *,
    feeds: Iterable[Mapping[str, Any]],
    grades: Iterable[Mapping[str, Any]],
    model: Mapping[str, Any],
    as_of_date: str | date,
    critical_errors: int = 0,
    high_errors: int = 0,
) -> dict[str, Any]:
    """역사 검증과 미래 섀도 운영의 공개 조건을 fail-closed로 판정합니다."""

    as_of = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date
    feed_rows = list(feeds)
    grade_rows = list(grades)
    eligible_games = sum(int(feed.get("quality", {}).get("eligible_games", 0)) for feed in feed_rows)
    sealed_predictions = sum(int(feed.get("quality", {}).get("sealed_predictions", 0)) for feed in feed_rows)
    late_predictions = sum(len(feed.get("quality", {}).get("late_game_ids", [])) for feed in feed_rows)
    target_dates = sorted({str(feed.get("target_date_et")) for feed in feed_rows if feed.get("target_date_et")})
    elapsed_days = (as_of - date.fromisoformat(target_dates[0])).days if target_dates else 0
    observations = [
        row
        for grade in grade_rows
        for row in grade.get("grades", [])
        if bool(row.get("evaluation_eligible"))
    ]
    targets = [int(row["home_win"]) for row in observations]
    selected_probability = [float(row["home_win_probability"]) for row in observations]
    elo_probability = [float(row["elo_home_win_probability"]) for row in observations]
    constant_rate = float(model.get("training", {}).get("constant_home_win_rate", 0.5))
    constant_probability = [constant_rate] * len(observations)
    evaluations: dict[str, Any] = {}
    if observations:
        evaluations = {
            "constant": evaluate_prediction_set(targets, constant_probability),
            "elo": evaluate_prediction_set(targets, elo_probability),
            "selected": evaluate_prediction_set(targets, selected_probability),
        }
    coverage = sealed_predictions / eligible_games if eligible_games else 0.0
    def pair(name: str) -> tuple[float, float]:
        metrics = evaluations[name]["metrics"]
        return float(metrics["log_loss"]), float(metrics["brier_score"])
    beats_constant = bool(evaluations) and all(left < right for left, right in zip(pair("selected"), pair("constant")))
    lr_selected = model.get("model_type") in {"logistic", "logistic_platt"}
    beats_elo = bool(evaluations) and all(left < right for left, right in zip(pair("selected"), pair("elo")))
    checks = {
        "minimum_30_days": elapsed_days >= 30,
        "minimum_300_graded_games": len(observations) >= 300,
        "coverage_at_least_95_percent": coverage >= 0.95,
        "critical_high_data_errors_zero": critical_errors == 0 and high_errors == 0,
        "post_start_or_late_seals_zero": late_predictions == 0,
        "beats_constant_on_both": beats_constant,
        "beats_elo_on_both_if_lr": beats_elo if lr_selected else True,
        "historical_holdout_passed": bool(model.get("holdout_evaluation", {}).get("passed")),
        "model_is_frozen": bool(model.get("frozen")),
    }
    passed = all(checks.values())
    safety_checks = (
        "coverage_at_least_95_percent",
        "critical_high_data_errors_zero",
        "post_start_or_late_seals_zero",
        "historical_holdout_passed",
        "model_is_frozen",
    )
    safety_passed = all(checks[name] for name in safety_checks)
    evidence_complete = checks["minimum_30_days"] and checks["minimum_300_graded_games"]
    if passed:
        failure_action = None
    elif not safety_passed:
        failure_action = "disable_prediction_publication_and_investigate"
    elif not evidence_complete:
        failure_action = "continue_future_validation"
    else:
        failure_action = "return_to_model_and_feature_improvement"
    return {
        "schema_version": "public-gate-v1",
        "as_of_date": as_of.isoformat(),
        "passed": passed,
        "public_release_allowed": passed,
        "prediction_publication_safe": safety_passed,
        "checks": checks,
        "metrics": {
            "elapsed_days": elapsed_days,
            "graded_games": len(observations),
            "eligible_games": eligible_games,
            "sealed_predictions": sealed_predictions,
            "coverage": round(coverage, 9),
            "late_predictions": late_predictions,
            "critical_errors": critical_errors,
            "high_errors": high_errors,
        },
        "evaluations": evaluations,
        "failure_action": failure_action,
    }


__all__ = ["evaluate_public_gate"]
