from __future__ import annotations

import json
from datetime import date
from urllib.parse import parse_qs, urlparse

import pytest

from mlb_predictor.collector import MlbStatsApiClient, iter_date_chunks


def test_date_chunks_cover_range_without_overlap() -> None:
    chunks = list(iter_date_chunks("2025-04-01", "2025-04-10", 4))
    assert chunks == [
        (date(2025, 4, 1), date(2025, 4, 4)),
        (date(2025, 4, 5), date(2025, 4, 8)),
        (date(2025, 4, 9), date(2025, 4, 10)),
    ]


@pytest.mark.parametrize("chunk_days", [0, 32])
def test_date_chunks_reject_invalid_size(chunk_days: int) -> None:
    with pytest.raises(ValueError):
        list(iter_date_chunks("2025-04-01", "2025-04-02", chunk_days))


def test_schedule_url_has_explicit_regular_season_contract() -> None:
    url = MlbStatsApiClient.build_schedule_url(date(2025, 4, 1), date(2025, 4, 2))
    query = parse_qs(urlparse(url).query)
    assert query["sportId"] == ["1"]
    assert query["gameType"] == ["R"]
    assert query["startDate"] == ["2025-04-01"]
    assert query["endDate"] == ["2025-04-02"]
    assert query["hydrate"] == ["probablePitcher,team,venue"]


def test_lineup_schedule_url_requests_lineups_without_changing_base_contract() -> None:
    url = MlbStatsApiClient.build_schedule_lineups_url(date(2025, 4, 1), date(2025, 4, 2))
    query = parse_qs(urlparse(url).query)
    assert query["sportId"] == ["1"]
    assert query["gameType"] == ["R"]
    assert query["hydrate"] == ["probablePitcher,team,venue,lineups"]


def test_active_roster_url_is_distinct_from_retrospective_full_season_roster() -> None:
    active = parse_qs(urlparse(MlbStatsApiClient.build_team_roster_url(119, 2026, roster_type="active")).query)
    retrospective = parse_qs(urlparse(MlbStatsApiClient.build_team_roster_url(119, 2026)).query)

    assert active["rosterType"] == ["active"]
    assert retrospective["rosterType"] == ["fullSeason"]


def test_client_uses_cache_without_network(tmp_path, schedule_payload, monkeypatch) -> None:
    client = MlbStatsApiClient(tmp_path)
    calls: list[str] = []

    def fake_request(url: str):
        calls.append(url)
        return schedule_payload

    monkeypatch.setattr(client, "_request_json", fake_request)
    first = client.fetch_schedule("2025-04-01", "2025-04-01")
    second = client.fetch_schedule("2025-04-01", "2025-04-01")

    assert len(calls) == 1
    assert first[0].from_cache is False
    assert first[0].fetched_at_utc is not None
    assert first[0].response_sha256 is not None
    assert second[0].from_cache is True
    assert second[0].fetched_at_utc == first[0].fetched_at_utc
    assert second[0].payload == schedule_payload
    assert (tmp_path / "schedule_2025-04-01_2025-04-01.json.meta.json").exists()


def test_legacy_cache_has_no_point_in_time_timestamp(tmp_path, schedule_payload, monkeypatch) -> None:
    cache_path = tmp_path / "schedule_2025-04-01_2025-04-01.json"
    cache_path.write_text(json.dumps(schedule_payload), encoding="utf-8")
    client = MlbStatsApiClient(tmp_path)
    monkeypatch.setattr(client, "_request_json", lambda url: pytest.fail("network must not be used"))

    result = client.fetch_schedule("2025-04-01", "2025-04-01")

    assert result[0].from_cache is True
    assert result[0].fetched_at_utc is None
    assert result[0].response_sha256 is not None


def test_player_stats_and_boxscore_urls_are_explicit() -> None:
    stats_url = MlbStatsApiClient.build_player_pitching_stats_url(123, date(2025, 3, 1), date(2025, 4, 1))
    stats_query = parse_qs(urlparse(stats_url).query)
    assert urlparse(stats_url).path == "/api/v1/people/123/stats"
    assert stats_query == {
        "stats": ["byDateRange"],
        "group": ["pitching"],
        "gameType": ["R"],
        "startDate": ["2025-03-01"],
        "endDate": ["2025-04-01"],
    }
    assert MlbStatsApiClient.build_boxscore_url(456).endswith("/api/v1/game/456/boxscore")


def test_generic_stats_cache_records_provenance(tmp_path, monkeypatch) -> None:
    client = MlbStatsApiClient(tmp_path)
    monkeypatch.setattr(client, "_request_json", lambda url: {"stats": []})

    result = client.fetch_player_pitching_stats(123, "2025-03-01", "2025-04-01")

    assert result.fetched_at_utc is not None
    assert result.response_sha256 is not None
    assert result.cache_path.name == "pitcher_123_2025-03-01_2025-04-01.json"


def test_people_season_stats_are_batched_and_cached(tmp_path, monkeypatch) -> None:
    client = MlbStatsApiClient(tmp_path)
    calls: list[str] = []

    def fake_request(url: str):
        calls.append(url)
        return {"people": []}

    monkeypatch.setattr(client, "_request_json", fake_request)
    results = client.fetch_people_pitching_season_stats([3, 2, 1], 2024, batch_size=2)

    assert len(results) == 2
    assert len(calls) == 2
    first_query = parse_qs(urlparse(calls[0]).query)
    assert first_query["personIds"] == ["1,2"]
    assert first_query["hydrate"] == ["stats(group=[pitching],type=[season],season=2024)"]
    assert all(result.fetched_at_utc for result in results)


def test_people_game_log_url_uses_game_log_hydration() -> None:
    url = MlbStatsApiClient.build_people_pitching_game_log_url([2, 1], 2025)
    query = parse_qs(urlparse(url).query)
    assert query["personIds"] == ["1,2"]
    assert query["hydrate"] == ["stats(group=[pitching],type=[gameLog],season=2025)"]


def test_people_batting_urls_use_hitting_group() -> None:
    season_url = MlbStatsApiClient.build_people_batting_season_url([2, 1], 2024)
    game_log_url = MlbStatsApiClient.build_people_batting_game_log_url([2, 1], 2025)
    assert parse_qs(urlparse(season_url).query)["hydrate"] == ["stats(group=[hitting],type=[season],season=2024)"]
    assert parse_qs(urlparse(game_log_url).query)["hydrate"] == ["stats(group=[hitting],type=[gameLog],season=2025)"]


def test_people_batting_platoon_url_requests_left_and_right_splits() -> None:
    url = MlbStatsApiClient.build_people_batting_platoon_url([2, 1], 2024)
    query = parse_qs(urlparse(url).query)

    assert query["personIds"] == ["1,2"]
    assert query["hydrate"] == [
        "stats(group=[hitting],type=[statSplits],season=2024,sitCodes=[vl,vr])"
    ]


def test_team_game_log_url_is_regular_season_only() -> None:
    url = MlbStatsApiClient.build_team_pitching_game_log_url(119, 2025)
    query = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/api/v1/teams/119/stats"
    assert query == {"stats": ["gameLog"], "group": ["pitching"], "season": ["2025"], "gameType": ["R"]}


def test_team_roster_url_uses_full_season_roster() -> None:
    url = MlbStatsApiClient.build_team_roster_url(119, 2025)
    assert urlparse(url).path == "/api/v1/teams/119/roster"
    assert parse_qs(urlparse(url).query) == {"rosterType": ["fullSeason"], "season": ["2025"]}


def test_client_does_not_cache_invalid_network_payload(tmp_path, monkeypatch) -> None:
    client = MlbStatsApiClient(tmp_path)
    monkeypatch.setattr(client, "_request_json", lambda url: {})

    with pytest.raises(RuntimeError, match="dates 누락"):
        client.fetch_schedule("2025-04-01", "2025-04-01")

    assert list(tmp_path.glob("*.json")) == []


def test_client_recovers_from_invalid_existing_cache(tmp_path, schedule_payload, monkeypatch) -> None:
    cache_path = tmp_path / "schedule_2025-04-01_2025-04-01.json"
    cache_path.write_text("{}", encoding="utf-8")
    client = MlbStatsApiClient(tmp_path)
    calls: list[str] = []

    def fake_request(url: str):
        calls.append(url)
        return schedule_payload

    monkeypatch.setattr(client, "_request_json", fake_request)
    result = client.fetch_schedule("2025-04-01", "2025-04-01")

    assert len(calls) == 1
    assert result[0].from_cache is False
    assert result[0].payload == schedule_payload
