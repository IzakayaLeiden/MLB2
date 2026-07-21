from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_URL = "https://statsapi.mlb.com/api/v1/schedule"
PEOPLE_BASE_URL = "https://statsapi.mlb.com/api/v1/people"
GAME_BASE_URL = "https://statsapi.mlb.com/api/v1/game"
TEAM_BASE_URL = "https://statsapi.mlb.com/api/v1/teams"


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
    fetched_at_utc: str | None = None
    response_sha256: str | None = None


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

    @staticmethod
    def build_schedule_lineups_url(start_date: date, end_date: date) -> str:
        params = {
            "sportId": 1,
            "gameType": "R",
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "hydrate": "probablePitcher,team,venue,lineups",
        }
        return f"{API_BASE_URL}?{urlencode(params)}"

    @staticmethod
    def build_player_pitching_stats_url(player_id: int, start_date: date, end_date: date) -> str:
        if player_id <= 0:
            raise ValueError("player_id must be positive.")
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        params = {
            "stats": "byDateRange",
            "group": "pitching",
            "gameType": "R",
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
        }
        return f"{PEOPLE_BASE_URL}/{player_id}/stats?{urlencode(params)}"

    @staticmethod
    def build_boxscore_url(game_id: int) -> str:
        if game_id <= 0:
            raise ValueError("game_id must be positive.")
        return f"{GAME_BASE_URL}/{game_id}/boxscore"

    @staticmethod
    def build_people_pitching_season_url(person_ids: Sequence[int], season: int) -> str:
        return MlbStatsApiClient._build_people_pitching_hydrate_url(person_ids, season, stats_type="season")

    @staticmethod
    def build_people_pitching_game_log_url(person_ids: Sequence[int], season: int) -> str:
        return MlbStatsApiClient._build_people_pitching_hydrate_url(person_ids, season, stats_type="gameLog")

    @staticmethod
    def build_people_batting_season_url(person_ids: Sequence[int], season: int) -> str:
        return MlbStatsApiClient._build_people_stats_hydrate_url(person_ids, season, group="hitting", stats_type="season")

    @staticmethod
    def build_people_batting_game_log_url(person_ids: Sequence[int], season: int) -> str:
        return MlbStatsApiClient._build_people_stats_hydrate_url(person_ids, season, group="hitting", stats_type="gameLog")

    @staticmethod
    def build_team_pitching_game_log_url(team_id: int, season: int) -> str:
        if team_id <= 0 or season < 1876:
            raise ValueError("team_id or season is invalid.")
        params = {"stats": "gameLog", "group": "pitching", "season": season, "gameType": "R"}
        return f"{TEAM_BASE_URL}/{team_id}/stats?{urlencode(params)}"

    @staticmethod
    def build_team_roster_url(team_id: int, season: int, *, roster_type: str = "fullSeason") -> str:
        if team_id <= 0 or season < 1876:
            raise ValueError("team_id or season is invalid.")
        if roster_type not in {"fullSeason", "active"}:
            raise ValueError("roster_type is invalid.")
        params = {"rosterType": roster_type, "season": season}
        return f"{TEAM_BASE_URL}/{team_id}/roster?{urlencode(params)}"

    @staticmethod
    def _build_people_pitching_hydrate_url(person_ids: Sequence[int], season: int, *, stats_type: str) -> str:
        return MlbStatsApiClient._build_people_stats_hydrate_url(
            person_ids,
            season,
            group="pitching",
            stats_type=stats_type,
        )

    @staticmethod
    def _build_people_stats_hydrate_url(
        person_ids: Sequence[int],
        season: int,
        *,
        group: str,
        stats_type: str,
    ) -> str:
        unique_ids = sorted(set(int(person_id) for person_id in person_ids))
        if not unique_ids or any(person_id <= 0 for person_id in unique_ids):
            raise ValueError("person_ids must contain positive IDs.")
        if season < 1876:
            raise ValueError("season is invalid.")
        if group not in {"pitching", "hitting"} or stats_type not in {"season", "gameLog"}:
            raise ValueError("group or stats_type is invalid.")
        params = {
            "personIds": ",".join(str(person_id) for person_id in unique_ids),
            "hydrate": f"stats(group=[{group}],type=[{stats_type}],season={season})",
        }
        return f"{PEOPLE_BASE_URL}?{urlencode(params)}"

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
                    self._write_cache_metadata(cache_path, payload, source_url)
                    from_cache = False
            else:
                payload = self._request_json(source_url)
                self._validate_payload_shape(payload, source_url)
                self._write_json_atomic(cache_path, payload)
                self._write_cache_metadata(cache_path, payload, source_url)
                from_cache = False
            metadata = self._read_cache_metadata(cache_path, payload, source_url)
            results.append(
                CachedPayload(
                    start_date=chunk_start.isoformat(),
                    end_date=chunk_end.isoformat(),
                    cache_path=cache_path,
                    payload=payload,
                    from_cache=from_cache,
                    source_url=source_url,
                    fetched_at_utc=metadata.get("fetched_at_utc") if metadata else None,
                    response_sha256=metadata.get("response_sha256") if metadata else self._payload_sha256(payload),
                )
            )
        return results

    def fetch_schedule_lineups(
        self,
        start_date: str | date,
        end_date: str | date,
        *,
        chunk_days: int = 31,
        refresh: bool = False,
    ) -> list[CachedPayload]:
        results: list[CachedPayload] = []
        for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, chunk_days):
            cache_path = self.cache_dir / "lineup-schedules" / f"lineups_{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
            source_url = self.build_schedule_lineups_url(chunk_start, chunk_end)
            results.append(
                self._fetch_cached_payload(
                    cache_path=cache_path,
                    source_url=source_url,
                    start_date=chunk_start.isoformat(),
                    end_date=chunk_end.isoformat(),
                    refresh=refresh,
                    validator=self._validate_payload_shape,
                )
            )
        return results

    def fetch_player_pitching_stats(
        self,
        player_id: int,
        start_date: str | date,
        end_date: str | date,
        *,
        refresh: bool = False,
    ) -> CachedPayload:
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        source_url = self.build_player_pitching_stats_url(player_id, start, end)
        cache_path = self.cache_dir / "pitchers" / f"pitcher_{player_id}_{start.isoformat()}_{end.isoformat()}.json"
        return self._fetch_cached_payload(
            cache_path=cache_path,
            source_url=source_url,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            refresh=refresh,
            validator=self._validate_stats_payload_shape,
        )

    def fetch_game_boxscore(self, game_id: int, *, refresh: bool = False) -> CachedPayload:
        source_url = self.build_boxscore_url(game_id)
        cache_path = self.cache_dir / "boxscores" / f"boxscore_{game_id}.json"
        return self._fetch_cached_payload(
            cache_path=cache_path,
            source_url=source_url,
            start_date="",
            end_date="",
            refresh=refresh,
            validator=self._validate_boxscore_payload_shape,
        )

    def fetch_people_pitching_season_stats(
        self,
        person_ids: Sequence[int],
        season: int,
        *,
        batch_size: int = 50,
        refresh: bool = False,
    ) -> list[CachedPayload]:
        unique_ids = sorted(set(int(person_id) for person_id in person_ids))
        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100.")
        results: list[CachedPayload] = []
        for offset in range(0, len(unique_ids), batch_size):
            batch = unique_ids[offset : offset + batch_size]
            source_url = self.build_people_pitching_season_url(batch, season)
            identifier = hashlib.sha256(",".join(str(value) for value in batch).encode("ascii")).hexdigest()[:16]
            cache_path = self.cache_dir / "people-seasons" / f"pitching_{season}_{identifier}.json"
            results.append(
                self._fetch_cached_payload(
                    cache_path=cache_path,
                    source_url=source_url,
                    start_date=f"{season}-01-01",
                    end_date=f"{season}-12-31",
                    refresh=refresh,
                    validator=self._validate_people_payload_shape,
                )
            )
        return results

    def fetch_people_pitching_game_logs(
        self,
        person_ids: Sequence[int],
        season: int,
        *,
        batch_size: int = 50,
        refresh: bool = False,
    ) -> list[CachedPayload]:
        unique_ids = sorted(set(int(person_id) for person_id in person_ids))
        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100.")
        results: list[CachedPayload] = []
        for offset in range(0, len(unique_ids), batch_size):
            batch = unique_ids[offset : offset + batch_size]
            source_url = self.build_people_pitching_game_log_url(batch, season)
            identifier = hashlib.sha256(",".join(str(value) for value in batch).encode("ascii")).hexdigest()[:16]
            cache_path = self.cache_dir / "people-game-logs" / f"pitching_{season}_{identifier}.json"
            results.append(
                self._fetch_cached_payload(
                    cache_path=cache_path,
                    source_url=source_url,
                    start_date=f"{season}-01-01",
                    end_date=f"{season}-12-31",
                    refresh=refresh,
                    validator=self._validate_people_payload_shape,
                )
            )
        return results

    def fetch_people_batting_season_stats(
        self,
        person_ids: Sequence[int],
        season: int,
        *,
        batch_size: int = 100,
        refresh: bool = False,
    ) -> list[CachedPayload]:
        return self._fetch_people_batting_batches(
            person_ids,
            season,
            stats_type="season",
            batch_size=batch_size,
            refresh=refresh,
        )

    def fetch_people_batting_game_logs(
        self,
        person_ids: Sequence[int],
        season: int,
        *,
        batch_size: int = 100,
        refresh: bool = False,
    ) -> list[CachedPayload]:
        return self._fetch_people_batting_batches(
            person_ids,
            season,
            stats_type="gameLog",
            batch_size=batch_size,
            refresh=refresh,
        )

    def _fetch_people_batting_batches(
        self,
        person_ids: Sequence[int],
        season: int,
        *,
        stats_type: str,
        batch_size: int,
        refresh: bool,
    ) -> list[CachedPayload]:
        unique_ids = sorted(set(int(person_id) for person_id in person_ids))
        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100.")
        results: list[CachedPayload] = []
        for offset in range(0, len(unique_ids), batch_size):
            batch = unique_ids[offset : offset + batch_size]
            source_url = (
                self.build_people_batting_season_url(batch, season)
                if stats_type == "season"
                else self.build_people_batting_game_log_url(batch, season)
            )
            identifier = hashlib.sha256(",".join(str(value) for value in batch).encode("ascii")).hexdigest()[:16]
            cache_dir = "people-batting-seasons" if stats_type == "season" else "people-batting-game-logs"
            cache_path = self.cache_dir / cache_dir / f"hitting_{season}_{identifier}.json"
            results.append(
                self._fetch_cached_payload(
                    cache_path=cache_path,
                    source_url=source_url,
                    start_date=f"{season}-01-01",
                    end_date=f"{season}-12-31",
                    refresh=refresh,
                    validator=self._validate_people_payload_shape,
                )
            )
        return results

    def fetch_team_pitching_game_log(self, team_id: int, season: int, *, refresh: bool = False) -> CachedPayload:
        source_url = self.build_team_pitching_game_log_url(team_id, season)
        cache_path = self.cache_dir / "team-game-logs" / f"pitching_{season}_{team_id}.json"
        return self._fetch_cached_payload(
            cache_path=cache_path,
            source_url=source_url,
            start_date=f"{season}-01-01",
            end_date=f"{season}-12-31",
            refresh=refresh,
            validator=self._validate_stats_payload_shape,
        )

    def fetch_team_full_season_roster(self, team_id: int, season: int, *, refresh: bool = False) -> CachedPayload:
        source_url = self.build_team_roster_url(team_id, season)
        cache_path = self.cache_dir / "team-rosters" / f"roster_{season}_{team_id}.json"
        return self._fetch_cached_payload(
            cache_path=cache_path,
            source_url=source_url,
            start_date=f"{season}-01-01",
            end_date=f"{season}-12-31",
            refresh=refresh,
            validator=self._validate_roster_payload_shape,
        )

    def fetch_team_active_roster(self, team_id: int, season: int, *, refresh: bool = False) -> CachedPayload:
        source_url = self.build_team_roster_url(team_id, season, roster_type="active")
        cache_path = self.cache_dir / "active-team-rosters" / f"roster_{season}_{team_id}.json"
        return self._fetch_cached_payload(
            cache_path=cache_path,
            source_url=source_url,
            start_date=f"{season}-01-01",
            end_date=f"{season}-12-31",
            refresh=refresh,
            validator=self._validate_roster_payload_shape,
        )

    def _fetch_cached_payload(
        self,
        *,
        cache_path: Path,
        source_url: str,
        start_date: str,
        end_date: str,
        refresh: bool,
        validator: Any,
    ) -> CachedPayload:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        from_cache = False
        if cache_path.exists() and not refresh:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                validator(payload, source_url)
                from_cache = True
            except (json.JSONDecodeError, RuntimeError):
                payload = self._request_json(source_url)
                validator(payload, source_url)
                self._write_json_atomic(cache_path, payload)
                self._write_cache_metadata(cache_path, payload, source_url)
        else:
            payload = self._request_json(source_url)
            validator(payload, source_url)
            self._write_json_atomic(cache_path, payload)
            self._write_cache_metadata(cache_path, payload, source_url)
        metadata = self._read_cache_metadata(cache_path, payload, source_url)
        return CachedPayload(
            start_date=start_date,
            end_date=end_date,
            cache_path=cache_path,
            payload=payload,
            from_cache=from_cache,
            source_url=source_url,
            fetched_at_utc=metadata.get("fetched_at_utc") if metadata else None,
            response_sha256=metadata.get("response_sha256") if metadata else self._payload_sha256(payload),
        )

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
    def _validate_stats_payload_shape(payload: dict[str, Any], source_url: str) -> None:
        if not isinstance(payload.get("stats"), list):
            raise RuntimeError(f"Unexpected player stats response (stats missing): {source_url}")

    @staticmethod
    def _validate_boxscore_payload_shape(payload: dict[str, Any], source_url: str) -> None:
        if not isinstance(payload.get("teams"), dict):
            raise RuntimeError(f"Unexpected boxscore response (teams missing): {source_url}")

    @staticmethod
    def _validate_people_payload_shape(payload: dict[str, Any], source_url: str) -> None:
        if not isinstance(payload.get("people"), list):
            raise RuntimeError(f"Unexpected people response (people missing): {source_url}")

    @staticmethod
    def _validate_roster_payload_shape(payload: dict[str, Any], source_url: str) -> None:
        if not isinstance(payload.get("roster"), list):
            raise RuntimeError(f"Unexpected roster response (roster missing): {source_url}")

    @staticmethod
    def _payload_sha256(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def _write_cache_metadata(cls, cache_path: Path, payload: dict[str, Any], source_url: str) -> None:
        metadata = {
            "schema_version": "mlb-api-cache-metadata-v1",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_url": source_url,
            "response_sha256": cls._payload_sha256(payload),
        }
        cls._write_json_atomic(cache_path.with_suffix(cache_path.suffix + ".meta.json"), metadata)

    @classmethod
    def _read_cache_metadata(
        cls,
        cache_path: Path,
        payload: dict[str, Any],
        source_url: str,
    ) -> dict[str, Any] | None:
        metadata_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if (
            metadata.get("schema_version") != "mlb-api-cache-metadata-v1"
            or metadata.get("source_url") != source_url
            or metadata.get("response_sha256") != cls._payload_sha256(payload)
            or not isinstance(metadata.get("fetched_at_utc"), str)
        ):
            return None
        return metadata

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(path)
