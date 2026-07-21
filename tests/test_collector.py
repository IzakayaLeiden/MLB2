from __future__ import annotations

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
    assert second[0].from_cache is True
    assert second[0].payload == schedule_payload


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
