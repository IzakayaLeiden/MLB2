from __future__ import annotations

import pytest

from mlb_predictor.bullpen_backtest import (
    add_probabilistic_reliever_features,
    add_reliever_availability_features,
)


def _prior(*, strikeouts: int = 30, walks: int = 10, saves: int = 2, holds: int = 5) -> dict:
    return {
        "has_history": True,
        "batters_faced": 100,
        "strikeouts": strikeouts,
        "walks": walks,
        "home_runs": 3,
        "earned_runs": 10,
        "games_finished": 10,
        "saves": saves,
        "holds": holds,
    }


def _appearance(date: str, team_id: int, *, pitches: int, started: int = 0) -> dict:
    return {
        "date": date,
        "team_id": team_id,
        "stats": {
            "has_history": True,
            "games_started": started,
            "pitches_thrown": pitches,
            "batters_faced": 4,
            "strikeouts": 1,
            "walks": 0,
            "home_runs": 0,
            "earned_runs": 0,
            "games_finished": 1,
            "saves": 0,
            "holds": 0,
        },
    }


def test_reliever_availability_uses_individual_past_workload() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "official_date": "2025-04-10",
        "away_team_id": 2,
        "home_team_id": 1,
    }
    home_ids = [101, 102, 103, 104]
    away_ids = [201, 202, 203, 204]
    rosters = {(2025, 1): home_ids, (2025, 2): away_ids}
    prior = {(2024, player_id): _prior() for player_id in [*home_ids, *away_ids]}
    logs = {
        **{(2025, player_id): [_appearance("2025-04-09", 1, pitches=30)] for player_id in home_ids},
        **{(2025, player_id): [_appearance("2025-04-08", 2, pitches=10)] for player_id in away_ids},
    }

    result = add_reliever_availability_features([row], rosters, prior, logs)[0]

    assert result["reliever_stats_through_policy"] == "strictly_before_official_date"
    assert result["home_known_core_reliever_count"] == 4
    assert result["away_known_core_reliever_count"] == 4
    assert result["bullpen_core_fatigue_advantage"] < 0
    assert result["bullpen_core_unavailable_count_advantage"] == -4


def test_reliever_availability_excludes_same_date_and_starter_appearances() -> None:
    row = {
        "game_id": 10,
        "season": 2025,
        "official_date": "2025-04-10",
        "away_team_id": 2,
        "home_team_id": 1,
    }
    rosters = {(2025, 1): [101], (2025, 2): []}
    prior = {(2024, 101): _prior()}
    baseline_logs = {(2025, 101): [_appearance("2025-04-08", 1, pitches=10)]}
    noisy_logs = {
        (2025, 101): [
            *baseline_logs[(2025, 101)],
            _appearance("2025-04-09", 1, pitches=100, started=1),
            _appearance("2025-04-10", 1, pitches=100),
        ]
    }

    baseline = add_reliever_availability_features([row], rosters, prior, baseline_logs)[0]
    noisy = add_reliever_availability_features([row], rosters, prior, noisy_logs)[0]

    assert noisy["bullpen_core_fatigue_advantage"] == pytest.approx(baseline["bullpen_core_fatigue_advantage"])
    assert noisy["bullpen_core_unavailable_count_advantage"] == baseline["bullpen_core_unavailable_count_advantage"]


def test_probabilistic_availability_penalizes_heavy_recent_workload() -> None:
    row = {
        "season": 2025,
        "official_date": "2025-04-10",
        "home_team_id": 1,
        "away_team_id": 2,
        "home_starter_expected_innings": 5.0,
        "away_starter_expected_innings": 5.0,
    }
    home_ids = [101, 102, 103, 104]
    away_ids = [201, 202, 203, 204]
    rosters = {(2025, 1): home_ids, (2025, 2): away_ids}
    prior = {
        (2024, player_id): _prior(saves=10, holds=5)
        for player_id in home_ids + away_ids
    }
    logs = {
        **{
            (2025, player_id): [
                _appearance("2025-04-08", 1, pitches=20),
                _appearance("2025-04-09", 1, pitches=30),
            ]
            for player_id in home_ids
        },
        **{
            (2025, player_id): [_appearance("2025-04-07", 2, pitches=10)]
            for player_id in away_ids
        },
    }

    result = add_probabilistic_reliever_features([row], rosters, prior, logs)[0]

    assert result["bullpen_expected_available_core_advantage"] < 0.0
    assert result["bullpen_high_leverage_availability_advantage"] < 0.0
    assert result["bullpen_back_to_back_risk_advantage"] < 0.0
