from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from .collector import CachedPayload, MlbStatsApiClient
from .pitching import normalize_pitcher_stats_payload, smoothed_pitcher_rate


CORE_RELIEVER_COUNT = 4
RELIEVER_FEATURE_NAMES = (
    "bullpen_core_quality_advantage",
    "bullpen_core_fatigue_advantage",
    "bullpen_core_unavailable_count_advantage",
)


def _source_record(payload: CachedPayload, *, kind: str, season: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "season": season,
        "cache_path": str(payload.cache_path),
        "fetched_at_utc": payload.fetched_at_utc,
        "source_url": payload.source_url,
        "response_sha256": payload.response_sha256,
    }


def collect_full_season_pitcher_rosters(
    *,
    client: MlbStatsApiClient,
    rows: Sequence[Mapping[str, Any]],
    target_seasons: Sequence[int],
    refresh: bool = False,
) -> tuple[dict[tuple[int, int], list[int]], list[dict[str, Any]]]:
    rosters: dict[tuple[int, int], list[int]] = {}
    sources: list[dict[str, Any]] = []
    for season in target_seasons:
        team_ids = sorted(
            {
                int(team_id)
                for row in rows
                if int(row["season"]) == int(season)
                for team_id in (row["away_team_id"], row["home_team_id"])
            }
        )
        for team_id in team_ids:
            response = client.fetch_team_full_season_roster(team_id, int(season), refresh=refresh)
            sources.append(_source_record(response, kind="full_season_roster", season=int(season)))
            pitcher_ids = sorted(
                {
                    int(entry["person"]["id"])
                    for entry in response.payload.get("roster", [])
                    if isinstance(entry, Mapping)
                    and isinstance(entry.get("person"), Mapping)
                    and entry["person"].get("id")
                    and isinstance(entry.get("position"), Mapping)
                    and entry["position"].get("type") == "Pitcher"
                }
            )
            rosters[(int(season), team_id)] = pitcher_ids
    return rosters, sources


def collect_reliever_pitching_data(
    *,
    client: MlbStatsApiClient,
    rosters: Mapping[tuple[int, int], Sequence[int]],
    target_seasons: Sequence[int],
    refresh: bool = False,
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[tuple[int, int], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    prior_stats: dict[tuple[int, int], dict[str, Any]] = {}
    game_logs: dict[tuple[int, int], list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for season in target_seasons:
        player_ids = sorted(
            {
                int(player_id)
                for (roster_season, _), ids in rosters.items()
                if int(roster_season) == int(season)
                for player_id in ids
            }
        )
        if not player_ids:
            continue
        stats_season = int(season) - 1
        for response in client.fetch_people_pitching_season_stats(
            player_ids,
            stats_season,
            batch_size=50,
            refresh=refresh,
        ):
            sources.append(_source_record(response, kind="reliever_prior_season_stats", season=stats_season))
            for person in response.payload.get("people", []):
                if isinstance(person, Mapping) and person.get("id"):
                    prior_stats[(stats_season, int(person["id"]))] = normalize_pitcher_stats_payload(
                        {"stats": person.get("stats", [])}
                    )
        for response in client.fetch_people_pitching_game_logs(
            player_ids,
            int(season),
            batch_size=50,
            refresh=refresh,
        ):
            sources.append(_source_record(response, kind="reliever_current_game_logs", season=int(season)))
            for person in response.payload.get("people", []):
                if not isinstance(person, Mapping) or not person.get("id"):
                    continue
                appearances: list[dict[str, Any]] = []
                for group in person.get("stats", []):
                    if not isinstance(group, Mapping):
                        continue
                    for split in group.get("splits", []):
                        if not isinstance(split, Mapping) or not split.get("date"):
                            continue
                        team = split.get("team", {}) if isinstance(split.get("team"), Mapping) else {}
                        game = split.get("game", {}) if isinstance(split.get("game"), Mapping) else {}
                        appearances.append(
                            {
                                "date": str(split["date"]),
                                "game_id": game.get("gamePk"),
                                "team_id": team.get("id"),
                                "stats": normalize_pitcher_stats_payload({"stats": [{"splits": [split]}]}),
                            }
                        )
                game_logs[(int(season), int(person["id"]))] = sorted(
                    appearances,
                    key=lambda item: (item["date"], int(item.get("game_id") or 0)),
                )
    return prior_stats, game_logs, sources


def _combine_pitching_stats(
    previous: Mapping[str, Any] | None,
    appearances: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = (
        "batters_faced",
        "strikeouts",
        "walks",
        "home_runs",
        "earned_runs",
        "games_finished",
        "saves",
        "holds",
    )
    totals = {field: int((previous or {}).get(field, 0) or 0) for field in fields}
    totals["has_history"] = bool((previous or {}).get("has_history"))
    for appearance in appearances:
        stats = appearance.get("stats", {}) if isinstance(appearance.get("stats"), Mapping) else {}
        totals["has_history"] = totals["has_history"] or bool(stats.get("has_history"))
        for field in fields:
            totals[field] += int(stats.get(field, 0) or 0)
    return totals


def _reliever_state(
    *,
    season: int,
    team_id: int,
    target_date: str,
    pitcher_ids: Sequence[int],
    prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
    game_logs: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    target = date.fromisoformat(target_date)
    candidates: list[dict[str, float]] = []
    for player_id in pitcher_ids:
        relief_appearances = [
            appearance
            for appearance in game_logs.get((season, int(player_id)), [])
            if str(appearance["date"]) < target_date
            and int(appearance.get("team_id") or -1) == team_id
            and int(appearance.get("stats", {}).get("games_started", 0) or 0) == 0
        ]
        if not relief_appearances:
            continue
        combined = _combine_pitching_stats(prior_stats.get((season - 1, int(player_id))), relief_appearances)
        k_minus_bb = smoothed_pitcher_rate(combined, "strikeouts") - smoothed_pitcher_rate(combined, "walks")
        role_score = (
            float(combined.get("saves", 0)) * 3.0
            + float(combined.get("holds", 0)) * 2.0
            + float(combined.get("games_finished", 0))
            + len(relief_appearances) * 0.25
        )
        pitches = {1: 0.0, 2: 0.0, 3: 0.0}
        appearance_days: set[int] = set()
        for appearance in relief_appearances:
            days_ago = (target - date.fromisoformat(str(appearance["date"]))).days
            if 1 <= days_ago <= 3:
                appearance_days.add(days_ago)
                thrown = float(appearance.get("stats", {}).get("pitches_thrown", 0) or 0)
                for window in (1, 2, 3):
                    if days_ago <= window:
                        pitches[window] += thrown
        fatigue = min(3.0, pitches[1] / 25.0 + pitches[2] / 50.0 + pitches[3] / 90.0)
        unavailable = float(pitches[1] >= 25 or pitches[2] >= 40 or {1, 2}.issubset(appearance_days))
        candidates.append(
            {
                "role_score": role_score,
                "quality": k_minus_bb,
                "fatigue": fatigue,
                "unavailable": unavailable,
            }
        )
    core = sorted(candidates, key=lambda item: (-item["role_score"], -item["quality"]))[:CORE_RELIEVER_COUNT]
    missing = CORE_RELIEVER_COUNT - len(core)
    return {
        "quality": sum(item["quality"] for item in core) / CORE_RELIEVER_COUNT,
        "fatigue": sum(item["fatigue"] for item in core) / CORE_RELIEVER_COUNT,
        "unavailable_count": sum(item["unavailable"] for item in core),
        "known_count": float(len(core)),
        "missing_count": float(missing),
    }


def add_reliever_availability_features(
    rows: Iterable[Mapping[str, Any]],
    rosters: Mapping[tuple[int, int], Sequence[int]],
    prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
    game_logs: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        season = int(row["season"])
        official_date = str(row["official_date"])
        states: dict[str, dict[str, float]] = {}
        for side in ("away", "home"):
            team_id = int(row[f"{side}_team_id"])
            states[side] = _reliever_state(
                season=season,
                team_id=team_id,
                target_date=official_date,
                pitcher_ids=rosters.get((season, team_id), []),
                prior_stats=prior_stats,
                game_logs=game_logs,
            )
        row.update(
            {
                "reliever_feature_provenance": "retrospective_past_team_appearances_v1",
                "reliever_stats_through_policy": "strictly_before_official_date",
                "bullpen_core_quality_advantage": states["home"]["quality"] - states["away"]["quality"],
                "bullpen_core_fatigue_advantage": states["away"]["fatigue"] - states["home"]["fatigue"],
                "bullpen_core_unavailable_count_advantage": states["away"]["unavailable_count"] - states["home"]["unavailable_count"],
                "away_known_core_reliever_count": int(states["away"]["known_count"]),
                "home_known_core_reliever_count": int(states["home"]["known_count"]),
            }
        )
        augmented.append(row)
    return augmented


def add_neutral_reliever_features(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "bullpen_core_quality_advantage": 0.0,
            "bullpen_core_fatigue_advantage": 0.0,
            "bullpen_core_unavailable_count_advantage": 0.0,
        }
        for row in rows
    ]
