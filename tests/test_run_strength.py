from __future__ import annotations

import pytest

from mlb_predictor.run_strength import add_dynamic_run_strength_features


def _game(game_id: int, official_date: str, home: int, away: int, home_score: int, away_score: int) -> dict:
    return {
        "game_id": game_id,
        "season": 2025,
        "official_date": official_date,
        "home_team_id": home,
        "away_team_id": away,
        "home_score": home_score,
        "away_score": away_score,
    }


def test_run_strength_uses_only_scores_before_official_date() -> None:
    games = [
        _game(1, "2025-04-01", 1, 2, 10, 1),
        _game(2, "2025-04-02", 1, 2, 1, 10),
    ]

    result = add_dynamic_run_strength_features(games, games, half_life_games=1.0)

    first, second = result
    assert first["run_strength_expected_margin"] == 0.0
    assert second["run_strength_expected_margin"] > 0


def test_same_date_doubleheader_features_share_atomic_snapshot() -> None:
    games = [
        _game(1, "2025-04-01", 1, 2, 9, 0),
        _game(2, "2025-04-01", 1, 2, 0, 9),
        _game(3, "2025-04-02", 1, 2, 5, 4),
    ]

    result = add_dynamic_run_strength_features(games, games, half_life_games=1.0)

    assert result[0]["run_strength_expected_margin"] == result[1]["run_strength_expected_margin"] == 0.0
    assert result[2]["run_strength_expected_margin"] == pytest.approx(0.0)


def test_single_future_date_is_seeded_from_completed_history() -> None:
    rows = [_game(10, "2025-04-03", 1, 2, 0, 0)]
    games = [
        _game(1, "2025-04-01", 1, 2, 9, 1),
        _game(2, "2025-04-02", 1, 2, 8, 2),
    ]

    result = add_dynamic_run_strength_features(rows, games, half_life_games=1.0)

    assert len(result) == 1
    assert result[0]["run_strength_offense_difference"] > 0.0
    assert result[0]["run_strength_defense_advantage"] > 0.0
    assert result[0]["run_strength_history_difference"] == 0.0
