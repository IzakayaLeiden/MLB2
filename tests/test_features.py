from __future__ import annotations

from copy import deepcopy

import pytest

from mlb_predictor.features import FeatureConfig, build_forecast_features, build_pregame_features


def by_id(rows):
    return {row["game_id"]: row for row in rows}


def without_target(row):
    return {key: value for key, value in row.items() if key != "home_win"}


def test_current_and_future_scores_do_not_leak_into_existing_features(game_factory) -> None:
    games = [
        game_factory(100, "2025-04-01", 1, 2, 5, 3),
        game_factory(101, "2025-04-02", 3, 1, 4, 2),
        game_factory(102, "2025-04-03", 1, 4, 10, 0),
    ]
    config = FeatureConfig(home_field_elo_advantage=0)
    baseline = by_id(build_pregame_features(games, config))

    changed_future = deepcopy(games)
    changed_future[2]["home_score"] = 0
    changed_future[2]["away_score"] = 99
    changed_future[2]["home_win"] = 0
    mutated = by_id(build_pregame_features(changed_future, config))

    assert baseline[100] == mutated[100]
    assert baseline[101] == mutated[101]
    assert without_target(baseline[102]) == without_target(mutated[102])
    assert baseline[100]["home_games_before"] == 0
    assert baseline[101]["away_recent_games_count"] == 1
    assert baseline[101]["away_recent_win_pct"] == 1.0
    assert baseline[101]["away_recent_runs_scored"] == 5.0
    assert baseline[101]["away_recent_runs_allowed"] == 3.0
    assert baseline[102]["home_recent_games_count"] == 2
    assert baseline[102]["home_recent_win_pct"] == 0.5
    assert baseline[102]["home_recent_runs_scored"] == 3.5
    assert baseline[102]["home_recent_runs_allowed"] == 3.5


def test_same_date_doubleheader_uses_atomic_snapshot_and_is_order_independent(game_factory) -> None:
    first = game_factory(200, "2025-04-10", 1, 2, 1, 0, start_hour=18)
    second = game_factory(201, "2025-04-10", 1, 2, 0, 1, start_hour=22)
    second["double_header"] = "Y"
    second["game_number"] = 2
    next_day = game_factory(202, "2025-04-11", 1, 2, 2, 1)
    config = FeatureConfig(home_field_elo_advantage=0)

    forward = by_id(build_pregame_features([first, second, next_day], config))
    reversed_input = by_id(build_pregame_features([second, first, next_day], config))

    assert forward == reversed_input
    for game_id in (200, 201):
        assert forward[game_id]["home_elo_pregame"] == 1500.0
        assert forward[game_id]["away_elo_pregame"] == 1500.0
        assert forward[game_id]["home_games_before"] == 0
        assert forward[game_id]["elo_expected_home_win_probability"] == 0.5
    assert forward[202]["home_elo_pregame"] == pytest.approx(1500.0)
    assert forward[202]["home_recent_games_count"] == 2
    assert forward[202]["home_recent_win_pct"] == 0.5
    assert forward[202]["home_recent_runs_scored"] == 0.5
    assert forward[202]["home_recent_runs_allowed"] == 0.5


def test_rolling_window_is_team_relative_and_truncated(game_factory) -> None:
    games = [
        game_factory(600, "2025-06-01", 1, 2, 5, 3),
        game_factory(601, "2025-06-02", 3, 1, 4, 2),
        game_factory(602, "2025-06-03", 1, 4, 7, 1),
        game_factory(603, "2025-06-04", 5, 1, 5, 6),
        game_factory(604, "2025-06-05", 1, 6, 3, 2),
    ]
    rows = by_id(build_pregame_features(games, FeatureConfig(recent_window=3, home_field_elo_advantage=0)))

    assert rows[603]["away_recent_games_count"] == 3
    assert rows[603]["away_recent_win_pct"] == pytest.approx(2 / 3)
    assert rows[603]["away_recent_runs_scored"] == pytest.approx(14 / 3, abs=1e-6)
    assert rows[603]["away_recent_runs_allowed"] == pytest.approx(8 / 3, abs=1e-6)
    assert rows[604]["home_recent_games_count"] == 3
    assert rows[604]["home_recent_win_pct"] == pytest.approx(2 / 3)
    assert rows[604]["home_recent_runs_scored"] == pytest.approx(5.0)
    assert rows[604]["home_recent_runs_allowed"] == pytest.approx(10 / 3, abs=1e-6)


def test_new_season_resets_form_and_regresses_elo(game_factory) -> None:
    games = [
        game_factory(700, "2024-09-29", 1, 2, 5, 3, season=2024),
        game_factory(701, "2025-03-27", 1, 2, 3, 2, season=2025),
    ]
    rows = by_id(build_pregame_features(games, FeatureConfig(home_field_elo_advantage=0, offseason_elo_carry=0.75)))

    assert rows[701]["home_games_before"] == 0
    assert rows[701]["home_recent_games_count"] == 0
    assert rows[701]["home_history_through_date"] is None
    assert rows[701]["home_elo_pregame"] == pytest.approx(1507.5)
    assert rows[701]["away_elo_pregame"] == pytest.approx(1492.5)


def test_forecast_features_use_only_completed_prior_dates_and_have_no_target(game_factory) -> None:
    history = [game_factory(800, "2025-04-01", 1, 2, 5, 3)]
    scheduled = game_factory(801, "2025-04-02", 2, 1, 1, 0)
    for key in ("home_score", "away_score", "home_win", "home_is_winner", "away_is_winner"):
        scheduled.pop(key)
    scheduled["status"] = "Preview"

    row = build_forecast_features(history, [scheduled])[0]

    assert "home_win" not in row
    assert row["away_history_through_date"] == "2025-04-01"
    assert row["home_history_through_date"] == "2025-04-01"
    assert row["away_recent_win_pct"] == 1.0
    assert row["home_recent_win_pct"] == 0.0


def test_forecast_rejects_same_day_completed_history(game_factory) -> None:
    history = [game_factory(810, "2025-04-02", 1, 2, 5, 3)]
    scheduled = game_factory(811, "2025-04-02", 3, 4, 1, 0)
    for key in ("home_score", "away_score", "home_win"):
        scheduled.pop(key)

    with pytest.raises(ValueError, match="늦어야"):
        build_forecast_features(history, [scheduled])
