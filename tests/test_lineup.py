from __future__ import annotations

import pytest

from mlb_predictor.lineup import (
    add_lineup_features,
    normalize_batter_stats_payload,
    smoothed_batter_ops,
    smoothed_platoon_ops,
)


def _stats(*, hits: int = 20, doubles: int = 4, triples: int = 0, home_runs: int = 2, walks: int = 8) -> dict:
    return {
        "has_history": True,
        "plate_appearances": 100,
        "at_bats": 90,
        "hits": hits,
        "doubles": doubles,
        "triples": triples,
        "home_runs": home_runs,
        "walks": walks,
        "hit_by_pitch": 1,
        "sac_flies": 1,
    }


def test_normalize_batter_stats_extracts_counting_stats() -> None:
    result = normalize_batter_stats_payload(
        {"stats": [{"splits": [{"stat": {"plateAppearances": 12, "atBats": 10, "hits": 4, "doubles": 1, "homeRuns": 1, "baseOnBalls": 2}}]}]}
    )
    assert result["has_history"] is True
    assert result["plate_appearances"] == 12
    assert result["hits"] == 4
    assert result["home_runs"] == 1
    assert smoothed_batter_ops(result) > 0.73


def test_platoon_ops_is_shrunk_toward_overall_history() -> None:
    overall = _stats(hits=20, doubles=4, home_runs=2, walks=8)
    strong_split = _stats(hits=40, doubles=10, home_runs=8, walks=15)

    result = smoothed_platoon_ops(strong_split, overall)

    assert result > smoothed_batter_ops(overall)
    assert result < 1.5


def test_lineup_features_exclude_target_date_batting_log() -> None:
    row = {"game_id": 10, "season": 2025, "official_date": "2025-04-10"}
    home_ids = list(range(1, 10))
    away_ids = list(range(11, 20))
    lineups = {10: {"home_player_ids": home_ids, "away_player_ids": away_ids}}
    prior = {(2024, player_id): _stats() for player_id in [*home_ids, *away_ids]}
    logs = {
        (2025, 1): [
            {"date": "2025-04-09", "stats": _stats(hits=4, doubles=1, home_runs=3, walks=1)},
            {"date": "2025-04-10", "stats": _stats(hits=0, doubles=0, home_runs=0, walks=0)},
        ]
    }

    with_same_date = add_lineup_features([row], lineups, prior, logs)[0]
    without_same_date = add_lineup_features(
        [row],
        lineups,
        prior,
        {(2025, 1): logs[(2025, 1)][:1]},
    )[0]

    assert with_same_date["lineup_stats_through_policy"] == "strictly_before_official_date"
    assert with_same_date["lineup_ops_advantage"] == pytest.approx(without_same_date["lineup_ops_advantage"])
    assert with_same_date["lineup_identity_point_in_time_verified"] is False


def test_incomplete_lineup_is_neutralized_and_reported() -> None:
    row = {"game_id": 10, "season": 2025, "official_date": "2025-04-10"}
    result = add_lineup_features(
        [row],
        {10: {"home_player_ids": [1], "away_player_ids": []}},
        {(2024, 1): _stats()},
        {},
    )[0]

    assert result["home_lineup_player_count"] == 1
    assert result["away_lineup_player_count"] == 0
    assert result["home_lineup_history_missing_rate"] == pytest.approx(8 / 9)
    assert result["away_lineup_history_missing_rate"] == 1.0
