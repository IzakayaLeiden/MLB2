from __future__ import annotations

import pytest

from mlb_predictor.pitching_backtest import add_bullpen_workload_features, add_prior_season_starter_features, add_rolling_starter_features


def test_prior_season_features_use_only_previous_year_stats() -> None:
    row = {
        "game_id": 1,
        "season": 2025,
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }
    stats = {
        (2024, 22): {"has_history": True, "batters_faced": 400, "strikeouts": 80, "walks": 40, "home_runs": 20, "earned_runs": 60},
        (2024, 11): {"has_history": True, "batters_faced": 400, "strikeouts": 140, "walks": 30, "home_runs": 10, "earned_runs": 35},
        (2025, 22): {"has_history": True, "batters_faced": 400, "strikeouts": 300, "walks": 0, "home_runs": 0, "earned_runs": 0},
    }

    result = add_prior_season_starter_features([row], stats)[0]

    assert result["pitching_stats_season"] == 2024
    assert result["starter_identity_point_in_time_verified"] is False
    assert result["starter_k_minus_bb_rate_difference"] > 0
    assert result["starter_earned_run_rate_advantage"] > 0
    assert result["starter_home_run_rate_advantage"] > 0


def test_missing_prior_season_pitcher_is_explicit_and_shrunk_to_prior() -> None:
    row = {
        "game_id": 1,
        "season": 2025,
        "away_probable_pitcher_id": None,
        "home_probable_pitcher_id": 11,
    }
    result = add_prior_season_starter_features([row], {})[0]

    assert result["away_starter_history_missing"] == 1
    assert result["home_starter_history_missing"] == 1
    assert result["starter_k_minus_bb_rate_difference"] == pytest.approx(0.0)


def test_rolling_features_exclude_target_date_game_log() -> None:
    row = {
        "game_id": 1,
        "season": 2025,
        "official_date": "2025-04-10",
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }
    prior = {
        (2024, 22): {"has_history": True, "batters_faced": 200, "strikeouts": 40, "walks": 20, "home_runs": 10, "earned_runs": 30},
        (2024, 11): {"has_history": True, "batters_faced": 200, "strikeouts": 40, "walks": 20, "home_runs": 10, "earned_runs": 30},
    }
    logs = {
        (2025, 11): [
            {"date": "2025-04-09", "stats": {"batters_faced": 20, "strikeouts": 10, "walks": 0, "home_runs": 0, "earned_runs": 0}},
            {"date": "2025-04-10", "stats": {"batters_faced": 20, "strikeouts": 0, "walks": 10, "home_runs": 5, "earned_runs": 10}},
        ]
    }
    result = add_rolling_starter_features([row], prior, logs)[0]

    assert result["pitching_stats_through_policy"] == "strictly_before_official_date"
    assert result["starter_k_minus_bb_rate_difference"] > 0


def test_bullpen_workload_excludes_same_date_and_subtracts_starter() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "official_date": "2025-04-10",
        "away_team_id": 2,
        "home_team_id": 1,
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }
    player_logs = {
        (2025, 11): [{"date": "2025-04-09", "game_id": 9, "stats": {"pitches_thrown": 70}}],
        (2025, 22): [{"date": "2025-04-09", "game_id": 8, "stats": {"pitches_thrown": 80}}],
    }
    team_logs = {
        (2025, 1): [
            {"date": "2025-04-09", "game_id": 9, "pitches_thrown": 120},
            {"date": "2025-04-10", "game_id": 10, "pitches_thrown": 999},
        ],
        (2025, 2): [{"date": "2025-04-09", "game_id": 8, "pitches_thrown": 110}],
    }
    prior_home = {**row, "game_id": 9, "official_date": "2025-04-09", "home_probable_pitcher_id": 11}
    prior_away = {**row, "game_id": 8, "official_date": "2025-04-09", "away_probable_pitcher_id": 22}
    results = add_bullpen_workload_features([prior_home, prior_away, row], player_logs, team_logs)
    result = next(item for item in results if item["game_id"] == 10)

    assert result["bullpen_pitches_1d_advantage"] == -20
    assert result["bullpen_pitches_3d_advantage"] == -20
