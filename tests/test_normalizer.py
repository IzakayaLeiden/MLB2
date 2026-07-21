from __future__ import annotations

from mlb_predictor.normalizer import normalize_future_schedule_payloads, normalize_schedule_payloads


def test_normalizer_keeps_only_final_non_tie_scored_games(schedule_payload) -> None:
    rows, skipped = normalize_schedule_payloads([schedule_payload])

    assert [row["game_id"] for row in rows] == [100]
    assert rows[0]["home_win"] == 1
    assert rows[0]["home_score"] == 5
    assert rows[0]["away_score"] == 3
    assert rows[0]["home_probable_pitcher_name"] == "Home Starter"
    assert {(item.game_id, item.reason) for item in skipped} == {
        (101, "status_not_final"),
        (102, "tie_game"),
        (103, "missing_score"),
    }


def test_normalizer_does_not_mutate_source(schedule_payload, cloned_payload) -> None:
    normalize_schedule_payloads([schedule_payload])
    assert schedule_payload == cloned_payload


def test_normalizer_excludes_rescheduled_game_outside_requested_official_date(schedule_payload) -> None:
    rescheduled = schedule_payload["dates"][0]["games"][0]
    rescheduled["officialDate"] = "2025-04-06"

    rows, skipped = normalize_schedule_payloads(
        [schedule_payload],
        start_date="2025-04-01",
        end_date="2025-04-05",
    )

    assert rows == []
    assert {(item.game_id, item.reason) for item in skipped} >= {(100, "official_date_out_of_range")}


def test_normalizer_fail_closes_resumed_game_to_prevent_temporal_leakage(schedule_payload) -> None:
    resumed = schedule_payload["dates"][0]["games"][0]
    resumed["gameDate"] = "2025-08-26T18:05:00Z"
    resumed["officialDate"] = "2025-04-01"
    resumed["resumedFrom"] = "2025-04-01T22:40:00Z"
    resumed["resumedFromDate"] = "2025-04-01"

    rows, skipped = normalize_schedule_payloads([schedule_payload])

    assert 100 not in {row["game_id"] for row in rows}
    assert (100, "resumed_game_temporal_ambiguity") in {(item.game_id, item.reason) for item in skipped}


def test_future_schedule_has_separate_result_free_schema(schedule_payload) -> None:
    rows, skipped = normalize_future_schedule_payloads([schedule_payload], target_date="2025-04-01")

    assert [row["game_id"] for row in rows] == [101]
    assert rows[0]["schema_version"] == "scheduled-game-v1"
    assert rows[0]["schedule_state"] == "scheduled"
    assert rows[0]["forecast_eligible"] is True
    assert "home_score" not in rows[0]
    assert "home_win" not in rows[0]
    assert {item.reason for item in skipped} == {"status_final"}


def test_future_schedule_preserves_postponed_and_cancelled_states(schedule_payload) -> None:
    games = schedule_payload["dates"][0]["games"]
    postponed = dict(games[1])
    postponed["gamePk"] = 201
    postponed["status"] = {"abstractGameState": "Preview", "detailedState": "Postponed", "statusCode": "D"}
    cancelled = dict(games[1])
    cancelled["gamePk"] = 202
    cancelled["status"] = {"abstractGameState": "Preview", "detailedState": "Cancelled", "statusCode": "C"}

    rows, _ = normalize_future_schedule_payloads([{"dates": [{"games": [postponed, cancelled]}]}])

    assert [(row["schedule_state"], row["forecast_eligible"]) for row in rows] == [
        ("postponed", False),
        ("cancelled", False),
    ]
