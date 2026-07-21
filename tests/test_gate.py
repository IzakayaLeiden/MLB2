from __future__ import annotations

from mlb_predictor.gate import evaluate_public_gate


def model(model_type: str = "logistic") -> dict[str, object]:
    return {
        "frozen": True,
        "model_type": model_type,
        "holdout_evaluation": {"passed": True},
        "training": {"constant_home_win_rate": 0.5},
    }


def test_public_gate_requires_every_historical_future_and_quality_check() -> None:
    feed = {
        "target_date_et": "2025-01-01",
        "quality": {"eligible_games": 300, "sealed_predictions": 300, "late_game_ids": []},
    }
    grades = [
        {
            "grades": [
                {
                    "home_win": index % 2,
                    "home_win_probability": 0.9 if index % 2 else 0.1,
                    "elo_home_win_probability": 0.7 if index % 2 else 0.3,
                    "evaluation_eligible": True,
                }
                for index in range(300)
            ]
        }
    ]

    result = evaluate_public_gate(feeds=[feed], grades=grades, model=model(), as_of_date="2025-02-01")

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_public_gate_fails_closed_without_prospective_evidence() -> None:
    result = evaluate_public_gate(feeds=[], grades=[], model=model("elo"), as_of_date="2025-02-01")

    assert result["passed"] is False
    assert result["checks"]["minimum_30_days"] is False
    assert result["checks"]["minimum_300_graded_games"] is False
    assert result["checks"]["beats_constant_on_both"] is False
    assert result["failure_action"] == "disable_prediction_publication_and_investigate"


def test_public_gate_allows_safe_publication_while_future_evidence_accumulates() -> None:
    feed = {
        "target_date_et": "2025-02-01",
        "quality": {"eligible_games": 15, "sealed_predictions": 15, "late_game_ids": []},
    }

    result = evaluate_public_gate(feeds=[feed], grades=[], model=model("elo"), as_of_date="2025-02-01")

    assert result["passed"] is False
    assert result["prediction_publication_safe"] is True
    assert result["failure_action"] == "continue_future_validation"
