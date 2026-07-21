from __future__ import annotations

import pytest

from mlb_predictor.schedule_context import (
    VENUE_CONTEXT_FEATURE_NAMES,
    add_schedule_context_features,
    ensure_schedule_context_feature_values,
    select_schedule_context_feature_block,
)


def _game(
    game_id: int,
    official_date: str,
    home: int,
    away: int,
    *,
    venue: int,
    home_score: int = 5,
    away_score: int = 3,
    day_night: str = "night",
) -> dict:
    return {
        "game_id": game_id,
        "official_date": official_date,
        "home_team_id": home,
        "away_team_id": away,
        "venue_id": venue,
        "home_score": home_score,
        "away_score": away_score,
        "day_night": day_night,
    }


def test_schedule_context_uses_only_prior_dates() -> None:
    games = [
        _game(1, "2025-04-01", 1, 2, venue=10, home_score=10, away_score=5),
        _game(2, "2025-04-01", 3, 4, venue=20, home_score=1, away_score=0),
        _game(3, "2025-04-02", 1, 2, venue=10, home_score=1, away_score=0),
    ]

    result = add_schedule_context_features(games, games, venue_prior_games=1.0)

    assert result[0]["venue_run_environment"] == 0.0
    assert result[1]["venue_run_environment"] == 0.0
    assert result[2]["venue_run_environment"] > 0.0


def test_single_future_date_is_seeded_from_completed_schedule() -> None:
    history = [
        _game(1, "2025-04-01", 1, 2, venue=10, day_night="night"),
        _game(2, "2025-04-02", 3, 1, venue=20, day_night="night"),
    ]
    target = [_game(3, "2025-04-03", 2, 1, venue=10, day_night="day")]

    result = add_schedule_context_features(target, history)

    assert result[0]["schedule_no_rest_travel_advantage"] == 1.0
    assert result[0]["schedule_night_to_day_advantage"] == 1.0
    assert result[0]["away_road_trip_game_number"] == 2.0


def test_same_date_doubleheader_shares_atomic_schedule_state() -> None:
    history = [_game(1, "2025-04-01", 1, 2, venue=10)]
    targets = [
        _game(2, "2025-04-02", 1, 2, venue=10),
        _game(3, "2025-04-02", 1, 2, venue=10),
    ]

    first, second = add_schedule_context_features(targets, history)

    for name in (
        "schedule_consecutive_days_advantage",
        "schedule_no_rest_travel_advantage",
        "away_road_trip_game_number",
    ):
        assert first[name] == pytest.approx(second[name])


def test_schedule_defaults_do_not_overwrite_real_candidate_values() -> None:
    [result] = ensure_schedule_context_feature_values(
        [{"venue_home_win_advantage": 0.125}]
    )

    assert result["venue_home_win_advantage"] == 0.125
    assert result["schedule_no_rest_travel_advantage"] == 0.0


def test_schedule_block_selection_neutralizes_excluded_fields() -> None:
    [result] = select_schedule_context_feature_block(
        [{"venue_home_win_advantage": 0.125, "schedule_no_rest_travel_advantage": 1.0}],
        VENUE_CONTEXT_FEATURE_NAMES,
    )

    assert result["venue_home_win_advantage"] == 0.125
    assert result["schedule_no_rest_travel_advantage"] == 0.0
