from __future__ import annotations

from mlb_predictor.normalizer import normalize_schedule_payloads


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
