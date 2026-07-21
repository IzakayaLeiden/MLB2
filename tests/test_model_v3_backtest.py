from __future__ import annotations

import pytest

from mlb_predictor.model_v3_backtest import (
    _predict_season,
    _candidate_ranking_key,
    _holdout_diagnostics,
    add_interaction_features,
    add_neutral_interaction_features,
    add_starter_readiness_features,
)


def test_starter_readiness_uses_only_starts_before_official_date() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "official_date": "2025-04-10",
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }
    prior = {
        (2024, 11): {"games_started": 2, "innings_pitched_outs": 36},
        (2024, 22): {"games_started": 2, "innings_pitched_outs": 30},
    }
    logs = {
        (2025, 11): [
            {"date": "2025-04-04", "stats": {"games_started": 1, "innings_pitched_outs": 12}},
            {"date": "2025-04-10", "stats": {"games_started": 1, "innings_pitched_outs": 27}},
        ],
        (2025, 22): [
            {"date": "2025-04-03", "stats": {"games_started": 1, "innings_pitched_outs": 15}},
        ],
    }

    result = add_starter_readiness_features([row], prior, logs)[0]

    assert result["starter_readiness_through_policy"] == "strictly_before_official_date"
    assert result["home_starter_rest_days"] == 5
    assert result["away_starter_rest_days"] == 6
    assert result["starter_rest_days_difference"] == -1
    assert result["home_starter_expected_innings"] == pytest.approx(5.5)
    assert result["away_starter_expected_innings"] == pytest.approx(5.0)
    assert result["starter_expected_innings_advantage"] == pytest.approx(0.5)


def test_starter_readiness_ignores_relief_appearances_for_rest() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "official_date": "2025-04-10",
        "away_probable_pitcher_id": None,
        "home_probable_pitcher_id": 11,
    }
    logs = {
        (2025, 11): [
            {"date": "2025-04-02", "stats": {"games_started": 1, "innings_pitched_outs": 18}},
            {"date": "2025-04-09", "stats": {"games_started": 0, "innings_pitched_outs": 3}},
        ]
    }

    result = add_starter_readiness_features([row], {}, logs)[0]

    assert result["home_starter_rest_days"] == 7
    assert result["home_starter_recent_start_count"] == 1
    assert result["away_starter_readiness_missing"] == 1
    assert result["home_starter_readiness_missing"] == 0


def test_starter_readiness_missing_uses_neutral_defaults() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "official_date": "2025-04-10",
        "away_probable_pitcher_id": None,
        "home_probable_pitcher_id": None,
    }

    result = add_starter_readiness_features([row], {}, {})[0]

    assert result["starter_rest_days_difference"] == 0
    assert result["starter_expected_innings_advantage"] == 0
    assert result["away_starter_readiness_missing"] == 1
    assert result["home_starter_readiness_missing"] == 1


def test_interactions_are_derived_only_from_existing_pregame_features() -> None:
    row = {
        "home_elo_minus_away": -20.0,
        "away_starter_expected_innings": 5.0,
        "home_starter_expected_innings": 6.0,
        "starter_k_minus_bb_rate_difference": 0.02,
        "starter_earned_run_rate_advantage": 0.3,
        "lineup_ops_advantage": 0.04,
        "starter_expected_innings_advantage": 1.0,
        "bullpen_core_unavailable_count_advantage": -2.0,
    }

    result = add_interaction_features([row])[0]

    assert result["elo_signed_square"] == -400.0
    assert result["starter_kbb_expected_innings_interaction"] == pytest.approx(0.11)
    assert result["starter_era_expected_innings_interaction"] == pytest.approx(1.65)
    assert result["lineup_ops_expected_innings_interaction"] == pytest.approx(0.22)
    assert result["starter_bullpen_availability_interaction"] == -2.0


def test_neutral_interactions_do_not_change_existing_values() -> None:
    result = add_neutral_interaction_features([{"game_id": 10}])[0]

    assert result["game_id"] == 10
    assert result["elo_signed_square"] == 0.0


def test_recent_training_window_excludes_older_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[int]] = []

    class StubModel:
        def predict_proba(self, matrix: object) -> list[float]:
            return [0.5]

    def stub_fit(rows: list[dict[str, int]], *, l2: float) -> StubModel:
        captured.append([int(row["season"]) for row in rows])
        return StubModel()

    monkeypatch.setattr("mlb_predictor.model_v3_backtest._fit", stub_fit)
    monkeypatch.setattr(
        "mlb_predictor.model_v3_backtest.extract_feature_matrix",
        lambda rows, specs: [[0.0] for _ in rows],
    )
    rows = [
        {"season": season, "home_win": season % 2}
        for season in range(2018, 2026)
    ]

    validation, _ = _predict_season(rows, season=2025, l2=0.1, training_years=3)

    assert captured == [[2022, 2023, 2024]]
    assert [row["season"] for row in validation] == [2025]


def test_candidate_ranking_prefers_cross_season_floor_over_aggregate_peak() -> None:
    candidates = {
        "unstable": {
            "evaluation": {"metrics": {"accuracy": 0.60, "log_loss": 0.67, "brier_score": 0.24}},
            "season_evaluations": {
                "2022": {"metrics": {"accuracy": 0.65}},
                "2023": {"metrics": {"accuracy": 0.59}},
                "2024": {"metrics": {"accuracy": 0.56}},
            },
        },
        "stable": {
            "evaluation": {"metrics": {"accuracy": 0.59, "log_loss": 0.68, "brier_score": 0.25}},
            "season_evaluations": {
                "2022": {"metrics": {"accuracy": 0.60}},
                "2023": {"metrics": {"accuracy": 0.59}},
                "2024": {"metrics": {"accuracy": 0.58}},
            },
        },
    }

    selected = min(candidates, key=lambda key: _candidate_ranking_key(key, candidates))

    assert selected == "stable"


def test_holdout_diagnostics_distinguish_accuracy_from_coverage() -> None:
    rows = [
        {"home_win": 1, "p_model_v3": 0.70, "p_elo": 0.60},
        {"home_win": 0, "p_model_v3": 0.30, "p_elo": 0.40},
        {"home_win": 0, "p_model_v3": 0.51, "p_elo": 0.49},
        {"home_win": 1, "p_model_v3": 0.49, "p_elo": 0.51},
    ]

    result = _holdout_diagnostics(rows)

    assert result["correct_predictions"] == 2
    assert result["required_correct_for_sixty_percent"] == 3
    assert result["additional_correct_needed_for_sixty_percent"] == 1
    assert result["elo_agreement"]["row_count"] == 2
    assert result["elo_disagreement"]["row_count"] == 2
    high_confidence = result["accuracy_at_confidence"][3]
    assert high_confidence["minimum_distance_from_fifty"] == 0.10
    assert high_confidence["coverage"] == 0.5
    assert high_confidence["accuracy"] == 1.0
