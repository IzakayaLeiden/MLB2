from __future__ import annotations

import pytest

from mlb_predictor.pitching import (
    aggregate_bullpen_usage,
    create_pregame_pitching_snapshot,
    extract_team_bullpen_usage,
    innings_pitched_to_outs,
    make_pitcher_evidence,
    normalize_pitcher_stats_payload,
    pitching_snapshot_to_features,
    write_pregame_pitching_snapshots,
    validate_pregame_pitching_snapshot,
)


def _scheduled_game() -> dict:
    return {
        "game_id": 900,
        "official_date": "2026-07-21",
        "game_start_utc": "2026-07-21T23:00:00Z",
        "away_team_id": 2,
        "home_team_id": 1,
        "away_probable_pitcher_id": 22,
        "away_probable_pitcher_name": "Away Starter",
        "home_probable_pitcher_id": 11,
        "home_probable_pitcher_name": "Home Starter",
    }


def _pitcher_evidence(player_id: int) -> dict:
    return make_pitcher_evidence(
        player_id=player_id,
        payload={
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "gamesPlayed": 20,
                                "gamesStarted": 20,
                                "inningsPitched": "120.1",
                                "battersFaced": 490,
                                "numberOfPitches": 1800,
                                "strikeOuts": 130,
                                "baseOnBalls": 31,
                                "homeRuns": 12,
                                "hits": 100,
                                "earnedRuns": 40,
                            }
                        }
                    ]
                }
            ]
        },
        stats_start_date="2026-03-01",
        stats_through_date="2026-07-20",
        fetched_at_utc="2026-07-21T12:00:00Z",
        source_url=f"https://example.test/people/{player_id}/stats",
        response_sha256=f"sha-{player_id}",
    )


def _bullpen(team_id: int) -> dict:
    return {
        "team_id": team_id,
        "data_through_date": "2026-07-20",
        "source_game_ids": [800 + team_id],
        "relievers": [],
        "team_pitches_1d": 0,
        "team_pitches_2d": 0,
        "team_pitches_3d": 0,
        "team_batters_faced_3d": 0,
    }


def _sources() -> list[dict]:
    return [
        {
            "fetched_at_utc": "2026-07-21T12:00:00Z",
            "source_url": "https://example.test/schedule",
            "response_sha256": "schedule-sha",
        }
    ]


def test_normalize_pitcher_stats_preserves_missing_history() -> None:
    assert normalize_pitcher_stats_payload({"stats": [{"splits": []}]}) == {
        "has_history": False,
        "games_played": 0,
        "games_started": 0,
        "innings_pitched": "0.0",
        "innings_pitched_outs": 0,
        "batters_faced": 0,
        "pitches_thrown": 0,
        "strikeouts": 0,
        "walks": 0,
        "home_runs": 0,
        "hits": 0,
        "earned_runs": 0,
    }


def test_innings_pitched_uses_baseball_outs_not_decimal_fraction() -> None:
    assert innings_pitched_to_outs("120.2") == 362


def test_extract_team_bullpen_usage_excludes_starter() -> None:
    boxscore = {
        "teams": {
            "home": {
                "pitchers": [11, 12],
                "players": {
                    "ID11": {"person": {"fullName": "Starter"}, "stats": {"pitching": {"gamesStarted": 1, "pitchesThrown": 90}}},
                    "ID12": {"person": {"fullName": "Reliever"}, "stats": {"pitching": {"gamesStarted": 0, "pitchesThrown": 21, "battersFaced": 6}}},
                },
            }
        }
    }
    usage = extract_team_bullpen_usage(boxscore, side="home", game_id=700, official_date="2026-07-20", team_id=1)
    assert usage["relievers"] == [{"player_id": 12, "name": "Reliever", "pitches_thrown": 21, "batters_faced": 6}]


def test_aggregate_bullpen_usage_uses_only_prior_three_dates() -> None:
    rows = [
        {"game_id": 1, "official_date": "2026-07-20", "team_id": 1, "relievers": [{"player_id": 12, "name": "R", "pitches_thrown": 20, "batters_faced": 5}]},
        {"game_id": 2, "official_date": "2026-07-19", "team_id": 1, "relievers": [{"player_id": 12, "name": "R", "pitches_thrown": 10, "batters_faced": 3}]},
        {"game_id": 3, "official_date": "2026-07-17", "team_id": 1, "relievers": [{"player_id": 12, "name": "R", "pitches_thrown": 99, "batters_faced": 20}]},
        {"game_id": 4, "official_date": "2026-07-21", "team_id": 1, "relievers": [{"player_id": 12, "name": "R", "pitches_thrown": 99, "batters_faced": 20}]},
    ]
    result = aggregate_bullpen_usage(rows, team_id=1, target_date="2026-07-21")
    assert result["source_game_ids"] == [1, 2]
    assert result["team_pitches_1d"] == 20
    assert result["team_pitches_2d"] == 30
    assert result["team_pitches_3d"] == 30


def test_verified_snapshot_passes_strict_time_boundaries() -> None:
    snapshot = create_pregame_pitching_snapshot(
        scheduled_game=_scheduled_game(),
        starter_evidence={11: _pitcher_evidence(11), 22: _pitcher_evidence(22)},
        bullpen_by_team={1: _bullpen(1), 2: _bullpen(2)},
        sources=_sources(),
        created_at_utc="2026-07-21T12:05:00Z",
    )
    assert snapshot["quality"] == {"status": "passed", "errors": []}
    assert validate_pregame_pitching_snapshot(snapshot) == []


def test_snapshot_rejects_late_or_unproven_sources_and_date_leakage() -> None:
    snapshot = create_pregame_pitching_snapshot(
        scheduled_game=_scheduled_game(),
        starter_evidence={11: _pitcher_evidence(11), 22: {**_pitcher_evidence(22), "stats_through_date": "2026-07-21"}},
        bullpen_by_team={1: _bullpen(1), 2: {**_bullpen(2), "source_game_ids": [900]}},
        sources=[{"fetched_at_utc": "2026-07-21T22:30:00Z", "source_url": "https://example.test/schedule", "response_sha256": None}],
        created_at_utc="2026-07-21T22:30:00Z",
    )
    errors = snapshot["quality"]["errors"]
    assert "snapshot_created_after_eligibility_cutoff" in errors
    assert "source_fetched_after_eligibility_cutoff" in errors
    assert "source_provenance_incomplete" in errors
    assert "away_starter_stats_not_past_only" in errors
    assert "away_bullpen_includes_target_game" in errors


def test_snapshot_rejects_source_observed_after_declared_creation() -> None:
    snapshot = create_pregame_pitching_snapshot(
        scheduled_game=_scheduled_game(),
        starter_evidence={11: _pitcher_evidence(11), 22: _pitcher_evidence(22)},
        bullpen_by_team={1: _bullpen(1), 2: _bullpen(2)},
        sources=[{"fetched_at_utc": "2026-07-21T12:10:00Z", "source_url": "https://example.test/schedule", "response_sha256": "sha"}],
        created_at_utc="2026-07-21T12:05:00Z",
    )
    assert "source_fetched_after_snapshot_created" in snapshot["quality"]["errors"]


def test_snapshot_allows_unannounced_pitcher_only_with_explicit_reason() -> None:
    game = _scheduled_game()
    game["home_probable_pitcher_id"] = None
    snapshot = create_pregame_pitching_snapshot(
        scheduled_game=game,
        starter_evidence={22: _pitcher_evidence(22)},
        bullpen_by_team={1: _bullpen(1), 2: _bullpen(2)},
        sources=_sources(),
        created_at_utc="2026-07-21T12:05:00Z",
    )
    assert snapshot["starters"]["home"]["missing_reason"] == "probable_pitcher_not_announced"
    assert snapshot["quality"]["status"] == "passed"


def test_snapshot_features_have_declared_home_advantage_directions() -> None:
    away = _pitcher_evidence(22)
    home = _pitcher_evidence(11)
    away["stats"]["strikeouts"] = 80
    home["stats"]["strikeouts"] = 160
    away_bullpen = {**_bullpen(2), "team_pitches_1d": 70, "team_pitches_3d": 200, "team_batters_faced_3d": 50}
    home_bullpen = {**_bullpen(1), "team_pitches_1d": 20, "team_pitches_3d": 100, "team_batters_faced_3d": 25}
    snapshot = create_pregame_pitching_snapshot(
        scheduled_game=_scheduled_game(),
        starter_evidence={11: home, 22: away},
        bullpen_by_team={1: home_bullpen, 2: away_bullpen},
        sources=_sources(),
        created_at_utc="2026-07-21T12:05:00Z",
    )

    features = pitching_snapshot_to_features(snapshot)

    assert features["starter_k_minus_bb_rate_difference"] > 0
    assert features["bullpen_pitches_1d_advantage"] == 50
    assert features["bullpen_pitches_3d_advantage"] == 100
    assert features["bullpen_batters_faced_3d_advantage"] == 25


def test_sealed_snapshot_cannot_be_overwritten(tmp_path) -> None:
    snapshot = create_pregame_pitching_snapshot(
        scheduled_game=_scheduled_game(),
        starter_evidence={11: _pitcher_evidence(11), 22: _pitcher_evidence(22)},
        bullpen_by_team={1: _bullpen(1), 2: _bullpen(2)},
        sources=_sources(),
        created_at_utc="2026-07-21T12:05:00Z",
    )
    paths = write_pregame_pitching_snapshots([snapshot], output_dir=tmp_path)
    assert paths[0].name == "pitching-v2-900.json"

    with pytest.raises(FileExistsError, match="already exists"):
        write_pregame_pitching_snapshots([snapshot], output_dir=tmp_path)
