from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def iter_date_chunks(start_date: str | date, end_date: str | date, chunk_days: int) -> Iterator[tuple[date, date]]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ValueError("start_date는 end_date보다 늦을 수 없습니다.")
    if not 1 <= chunk_days <= 31:
        raise ValueError("chunk_days는 1~31 범위여야 합니다.")

    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


@dataclass(frozen=True)
class CachedPayload:
    start_date: str
    end_date: str
    cache_path: Path
    payload: dict[str, Any]
    from_cache: bool
    source_url: str


class MlbStatsApiClient:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        user_agent: str = "mlb-pregame-dataset/0.1",
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts는 1 이상이어야 합니다.")
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.user_agent = user_agent

    @staticmethod
    def build_schedule_url(start_date: date, end_date: date) -> str:
        params = {
            "sportId": 1,
            "gameType": "R",
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "hydrate": "probablePitcher,team,venue",
        }
        return f"{API_BASE_URL}?{urlencode(params)}"

    def fetch_schedule(
        self,
        start_date: str | date,
        end_date: str | date,
        *,
        chunk_days: int = 7,
        refresh: bool = False,
    ) -> list[CachedPayload]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        results: list[CachedPayload] = []
        for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, chunk_days):
            cache_path = self.cache_dir / f"schedule_{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
            source_url = self.build_schedule_url(chunk_start, chunk_end)
            if cache_path.exists() and not refresh:
                try:
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    self._validate_payload_shape(payload, source_url)
                    from_cache = True
                except (json.JSONDecodeError, RuntimeError):
                    payload = self._request_json(source_url)
                    self._validate_payload_shape(payload, source_url)
                    self._write_json_atomic(cache_path, payload)
                    from_cache = False
            else:
                payload = self._request_json(source_url)
                self._validate_payload_shape(payload, source_url)
                self._write_json_atomic(cache_path, payload)
                from_cache = False
            results.append(
                CachedPayload(
                    start_date=chunk_start.isoformat(),
                    end_date=chunk_end.isoformat(),
                    cache_path=cache_path,
                    payload=payload,
                    from_cache=from_cache,
                    source_url=source_url,
                )
            )
        return results

    def _request_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                request = Request(url, headers={"Accept": "application/json", "User-Agent": self.user_agent})
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    if response.status != 200:
                        raise RuntimeError(f"MLB Stats API가 HTTP {response.status}를 반환했습니다.")
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RuntimeError("MLB Stats API 응답이 JSON 객체가 아닙니다.")
                return payload
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"MLB Stats API 요청이 {self.max_attempts}회 모두 실패했습니다: {last_error}") from last_error

    @staticmethod
    def _validate_payload_shape(payload: dict[str, Any], source_url: str) -> None:
        dates = payload.get("dates")
        if not isinstance(dates, list):
            raise RuntimeError(f"예상하지 못한 일정 응답 구조입니다(dates 누락): {source_url}")

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(path)
