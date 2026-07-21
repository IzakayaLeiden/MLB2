from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from mlb_predictor.collector import CachedPayload
from mlb_predictor.model_v3_backtest import MODEL_V3_FEATURE_SPECS
from mlb_predictor.model_v3_snapshot import (
    _expected_lineups_from_last_completed_games,
    create_pregame_model_v3_snapshot,
    validate_pregame_model_v3_snapshot,
    write_pregame_model_v3_snapshots,
)


def _game() -> dict:
    return {
        "game_id": 900,
        "official_date": "2026-07-21",
        "game_start_utc": "2026-07-21T18:00:00Z",
        "away_team_id": 2,
        "home_team_id": 1,
        "away_probable_pitcher_id": 22,
        "home_probable_pitcher_id": 11,
    }


def _feature_row() -> dict:
    features = {
        spec.source: (0.0 if spec.transform == "numeric" else "night")
        for spec in MODEL_V3_FEATURE_SPECS
    }
    features.update(
        {
            "away_known_core_reliever_count": 4,
            "home_known_core_reliever_count": 4,
        }
    )
    return features


def _snapshot(*, created_at: str = "2026-07-21T15:00:00Z", fetched_at: str = "2026-07-21T14:00:00Z") -> dict:
    return create_pregame_model_v3_snapshot(
        scheduled_game=_game(),
        feature_row=_feature_row(),
        lineup={
            "away_player_ids": list(range(101, 110)),
            "home_player_ids": list(range(201, 210)),
        },
        active_rosters={
            "away": {"pitcher_ids": [301, 302, 303, 304]},
            "home": {"pitcher_ids": [401, 402, 403, 404]},
        },
        sources=[
            {
                "kind": "target_schedule_lineups",
                "fetched_at_utc": fetched_at,
                "source_url": "https://example.test/schedule",
                "response_sha256": "abc123",
            }
        ],
        created_at_utc=created_at,
    )


def test_verified_model_v3_snapshot_passes_all_point_in_time_checks() -> None:
    snapshot = _snapshot()

    assert snapshot["quality"] == {"status": "passed", "errors": []}
    assert validate_pregame_model_v3_snapshot(snapshot) == []
    assert snapshot["data_through_date"] == "2026-07-20"


def test_model_v3_snapshot_fails_closed_for_late_source_or_incomplete_lineup() -> None:
    late = _snapshot(fetched_at="2026-07-21T17:30:00Z")
    incomplete = create_pregame_model_v3_snapshot(
        scheduled_game=_game(),
        feature_row=_feature_row(),
        lineup={"away_player_ids": [101], "home_player_ids": list(range(201, 210))},
        active_rosters={
            "away": {"pitcher_ids": [301, 302, 303, 304]},
            "home": {"pitcher_ids": [401, 402, 403, 404]},
        },
        sources=[
            {
                "fetched_at_utc": "2026-07-21T14:00:00Z",
                "source_url": "https://example.test/schedule",
                "response_sha256": "abc123",
            }
        ],
        created_at_utc="2026-07-21T15:00:00Z",
    )

    assert "source_fetched_after_eligibility_cutoff" in late["quality"]["errors"]
    assert "source_fetched_after_snapshot_created" in late["quality"]["errors"]
    assert "away_lineup_incomplete" in incomplete["quality"]["errors"]


def test_model_v3_snapshot_is_sealed_against_overwrite(tmp_path) -> None:
    snapshot = _snapshot()

    paths = write_pregame_model_v3_snapshots([snapshot], output_dir=tmp_path)

    assert paths[0].name == "model-v3-900.json"
    with pytest.raises(FileExistsError, match="sealed model-v3 snapshot"):
        write_pregame_model_v3_snapshots([snapshot], output_dir=tmp_path)


def test_expected_lineup_uses_only_last_completed_game_batting_order() -> None:
    prior_game = {
        "gamePk": 800,
        "gameType": "R",
        "season": "2026",
        "gameDate": "2026-07-20T18:00:00Z",
        "officialDate": "2026-07-20",
        "status": {"abstractGameState": "Final", "detailedState": "Final", "statusCode": "F"},
        "teams": {
            "away": {"team": {"id": 2, "name": "Away"}, "score": 3},
            "home": {"team": {"id": 1, "name": "Home"}, "score": 4},
        },
        "venue": {"id": 10, "name": "Park"},
        "isTie": False,
    }

    def payload(name: str, body: dict) -> CachedPayload:
        return CachedPayload(
            start_date="2026-07-20",
            end_date="2026-07-20",
            cache_path=Path(name),
            payload=body,
            from_cache=False,
            source_url=f"https://example.test/{name}",
            fetched_at_utc="2026-07-21T12:00:00Z",
            response_sha256=name,
        )

    class StubClient:
        def fetch_schedule(self, *args: object, **kwargs: object) -> list[CachedPayload]:
            return [payload("schedule", {"dates": [{"date": "2026-07-20", "games": [prior_game]}]})]

        def fetch_game_boxscore(self, game_id: int, *, refresh: bool) -> CachedPayload:
            assert game_id == 800
            return payload(
                "boxscore",
                {
                    "teams": {
                        "away": {"battingOrder": list(range(101, 110))},
                        "home": {"battingOrder": list(range(201, 210))},
                    }
                },
            )

    expected, sources = _expected_lineups_from_last_completed_games(
        StubClient(),  # type: ignore[arg-type]
        [_game()],
        target=date(2026, 7, 21),
        refresh=False,
    )

    assert expected == {1: list(range(201, 210)), 2: list(range(101, 110))}
    assert [source["kind"] for source in sources] == [
        "expected_lineup_prior_schedule",
        "expected_lineup_prior_boxscore",
    ]
