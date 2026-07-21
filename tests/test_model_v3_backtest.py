from __future__ import annotations

import pytest

from mlb_predictor.model_v3_backtest import (
    MODEL_V3_FEATURE_SPECS,
    _predict_season,
    _candidate_ranking_key,
    _holdout_diagnostics,
    _predict_lda_season,
    add_context_features,
    add_neutral_context_features,
    add_platoon_features,
    add_platoon_performance_features,
    add_recent_starter_form_features,
    add_interaction_features,
    add_neutral_interaction_features,
    add_starter_readiness_features,
    blend_with_elo,
)


def test_blend_with_elo_uses_fixed_model_weight() -> None:
    rows = [
        {"elo_expected_home_win_probability": 0.60},
        {"elo_expected_home_win_probability": 0.40},
    ]

    result = blend_with_elo(rows, [0.80, 0.20], model_weight=0.25)

    assert result.tolist() == pytest.approx([0.65, 0.35])


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


def test_context_features_keep_recent_offense_and_defense_separate() -> None:
    result = add_context_features(
        [
            {
                "home_recent_runs_scored": 5.2,
                "away_recent_runs_scored": 4.1,
                "home_recent_runs_allowed": 3.8,
                "away_recent_runs_allowed": 4.5,
                "season_win_pct_difference": -0.2,
                "recent_win_pct_difference": 0.1,
            }
        ]
    )[0]

    assert result["recent_offense_difference"] == pytest.approx(1.1)
    assert result["recent_defense_advantage"] == pytest.approx(0.7)
    assert result["season_win_pct_signed_square"] == pytest.approx(-0.04)
    assert result["recent_win_pct_signed_square"] == pytest.approx(0.01)


def test_neutral_context_features_preserve_original_candidate_contract() -> None:
    result = add_neutral_context_features([{"game_id": 10}])[0]

    assert result["game_id"] == 10
    assert result["recent_offense_difference"] == 0.0


def test_recent_starter_form_uses_only_prior_starts() -> None:
    row = {
        "season": 2025,
        "official_date": "2025-04-10",
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }
    logs = {
        (2025, 11): [
            {"date": "2025-04-04", "stats": {"games_started": 1, "batters_faced": 20, "strikeouts": 6, "walks": 1, "earned_runs": 1, "innings_pitched_outs": 18, "pitches_thrown": 90}},
            {"date": "2025-04-10", "stats": {"games_started": 1, "batters_faced": 20, "strikeouts": 0, "walks": 10, "earned_runs": 10, "innings_pitched_outs": 3, "pitches_thrown": 90}},
        ],
        (2025, 22): [
            {"date": "2025-04-03", "stats": {"games_started": 1, "batters_faced": 20, "strikeouts": 2, "walks": 3, "earned_runs": 4, "innings_pitched_outs": 12, "pitches_thrown": 85}},
        ],
    }

    result = add_recent_starter_form_features([row], logs)[0]

    assert result["starter_recent_k_minus_bb_difference"] > 0
    assert result["starter_recent_earned_run_advantage"] > 0
    assert result["starter_recent_outs_advantage"] == pytest.approx(2.0)
    assert result["away_starter_recent_form_missing"] == 0
    assert result["home_starter_recent_form_missing"] == 0


def test_platoon_features_compare_each_lineup_to_opposing_starter_hand() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }
    away_ids = list(range(100, 109))
    home_ids = list(range(200, 209))
    batter_stats = {
        **{(2024, player_id): {"bat_side": "R"} for player_id in away_ids},
        **{(2024, player_id): {"bat_side": "R"} for player_id in home_ids},
    }
    pitcher_stats = {
        (2024, 11): {"pitch_hand": "L"},
        (2024, 22): {"pitch_hand": "R"},
    }

    result = add_platoon_features(
        [row],
        {10: {"away_player_ids": away_ids, "home_player_ids": home_ids}},
        batter_stats,
        pitcher_stats,
    )[0]

    assert result["lineup_platoon_advantage"] == -1.0
    assert result["lineup_same_side_exposure_advantage"] == -1.0
    assert result["away_lineup_handedness_missing_rate"] == 0.0
    assert result["home_lineup_handedness_missing_rate"] == 0.0


def test_prior_season_platoon_performance_matches_opposing_starter() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }
    away_ids = list(range(100, 109))
    home_ids = list(range(200, 209))
    overall = {
        (2024, player_id): {
            "has_history": True,
            "plate_appearances": 100,
            "at_bats": 90,
            "hits": 20,
            "doubles": 4,
            "triples": 0,
            "home_runs": 2,
            "walks": 8,
            "hit_by_pitch": 1,
            "sac_flies": 1,
        }
        for player_id in [*away_ids, *home_ids]
    }
    strong = {**overall[(2024, home_ids[0])], "hits": 40, "home_runs": 8, "walks": 15}
    platoon = {
        (2024, player_id): {"vr": strong if player_id in home_ids else overall[(2024, player_id)]}
        for player_id in [*away_ids, *home_ids]
    }

    result = add_platoon_performance_features(
        [row],
        {10: {"away_player_ids": away_ids, "home_player_ids": home_ids}},
        overall,
        platoon,
        {(2024, 11): {"pitch_hand": "R"}, (2024, 22): {"pitch_hand": "R"}},
    )[0]

    assert result["lineup_platoon_ops_advantage"] > 0
    assert result["away_lineup_platoon_stats_missing_rate"] == 0.0
    assert result["home_lineup_platoon_stats_missing_rate"] == 0.0


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


def test_lda_candidate_returns_finite_probabilities() -> None:
    rows = []
    for season in (2020, 2021, 2022):
        for index in range(8):
            row = {
                spec.source: (
                    float(index + season % 3) if spec.transform == "numeric" else "night"
                )
                for spec in MODEL_V3_FEATURE_SPECS
            }
            row.update({"season": season, "home_win": index % 2})
            rows.append(row)

    validation, probability = _predict_lda_season(rows, season=2022, shrinkage=0.5)

    assert len(validation) == 8
    assert len(probability) == 8
    assert all(0.0 < value < 1.0 for value in probability)
