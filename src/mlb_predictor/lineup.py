from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .collector import CachedPayload, MlbStatsApiClient


LINEUP_SIZE = 9
LINEUP_WEIGHTS = (1.08, 1.07, 1.05, 1.03, 1.00, 0.98, 0.95, 0.93, 0.91)
BATTING_PRIOR_PLATE_APPEARANCES = 200.0
LEAGUE_OBP_PRIOR = 0.320
LEAGUE_SLG_PRIOR = 0.410

LINEUP_FEATURE_NAMES = (
    "lineup_ops_advantage",
    "lineup_top4_ops_advantage",
    "away_lineup_history_missing_rate",
    "home_lineup_history_missing_rate",
)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_batter_stats_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    splits: list[Mapping[str, Any]] = []
    for group in payload.get("stats", []):
        if isinstance(group, Mapping) and isinstance(group.get("splits"), list):
            splits.extend(split for split in group["splits"] if isinstance(split, Mapping))
    stat = splits[0].get("stat", {}) if splits and isinstance(splits[0].get("stat"), Mapping) else {}
    return {
        "has_history": bool(splits),
        "plate_appearances": int(_number(stat.get("plateAppearances"))),
        "at_bats": int(_number(stat.get("atBats"))),
        "hits": int(_number(stat.get("hits"))),
        "doubles": int(_number(stat.get("doubles"))),
        "triples": int(_number(stat.get("triples"))),
        "home_runs": int(_number(stat.get("homeRuns"))),
        "walks": int(_number(stat.get("baseOnBalls"))),
        "hit_by_pitch": int(_number(stat.get("hitByPitch"))),
        "sac_flies": int(_number(stat.get("sacFlies"))),
    }


def _source_record(payload: CachedPayload, *, kind: str, season: int | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "season": season,
        "cache_path": str(payload.cache_path),
        "fetched_at_utc": payload.fetched_at_utc,
        "source_url": payload.source_url,
        "response_sha256": payload.response_sha256,
    }


def collect_historical_lineups(
    *,
    client: MlbStatsApiClient,
    rows: Sequence[Mapping[str, Any]],
    refresh: bool = False,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    relevant_game_ids = {int(row["game_id"]) for row in rows}
    dates_by_season: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        dates_by_season[int(row["season"])].append(str(row["official_date"]))
    lineups: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for season, dates in sorted(dates_by_season.items()):
        payloads = client.fetch_schedule_lineups(min(dates), max(dates), chunk_days=31, refresh=refresh)
        for payload in payloads:
            sources.append(_source_record(payload, kind="historical_lineup_schedule", season=season))
            for date_entry in payload.payload.get("dates", []):
                if not isinstance(date_entry, Mapping):
                    continue
                for game in date_entry.get("games", []):
                    if not isinstance(game, Mapping) or int(game.get("gamePk", -1)) not in relevant_game_ids:
                        continue
                    game_id = int(game["gamePk"])
                    raw_lineups = game.get("lineups", {}) if isinstance(game.get("lineups"), Mapping) else {}
                    lineups[game_id] = {
                        "game_id": game_id,
                        "official_date": str(game.get("officialDate") or date_entry.get("date")),
                        "home_player_ids": [
                            int(player["id"])
                            for player in raw_lineups.get("homePlayers", [])
                            if isinstance(player, Mapping) and player.get("id")
                        ],
                        "away_player_ids": [
                            int(player["id"])
                            for player in raw_lineups.get("awayPlayers", [])
                            if isinstance(player, Mapping) and player.get("id")
                        ],
                        "point_in_time_verified": False,
                    }
    return lineups, sources


def collect_prior_season_batting_stats(
    *,
    client: MlbStatsApiClient,
    rows: Sequence[Mapping[str, Any]],
    lineups: Mapping[int, Mapping[str, Any]],
    target_seasons: Sequence[int],
    refresh: bool = False,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    stats: dict[tuple[int, int], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for season in target_seasons:
        player_ids = sorted(
            {
                int(player_id)
                for row in rows
                if int(row["season"]) == int(season)
                for side in ("away", "home")
                for player_id in lineups.get(int(row["game_id"]), {}).get(f"{side}_player_ids", [])
            }
        )
        if not player_ids:
            continue
        stats_season = int(season) - 1
        for response in client.fetch_people_batting_season_stats(player_ids, stats_season, refresh=refresh):
            sources.append(_source_record(response, kind="prior_season_batting", season=stats_season))
            for person in response.payload.get("people", []):
                if isinstance(person, Mapping) and person.get("id"):
                    normalized = normalize_batter_stats_payload(
                        {"stats": person.get("stats", [])}
                    )
                    bat_side = person.get("batSide", {}) if isinstance(person.get("batSide"), Mapping) else {}
                    normalized["bat_side"] = bat_side.get("code")
                    stats[(stats_season, int(person["id"]))] = normalized
    return stats, sources


def collect_current_season_batting_game_logs(
    *,
    client: MlbStatsApiClient,
    rows: Sequence[Mapping[str, Any]],
    lineups: Mapping[int, Mapping[str, Any]],
    target_seasons: Sequence[int],
    refresh: bool = False,
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], list[dict[str, Any]]]:
    logs: dict[tuple[int, int], list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for season in target_seasons:
        player_ids = sorted(
            {
                int(player_id)
                for row in rows
                if int(row["season"]) == int(season)
                for side in ("away", "home")
                for player_id in lineups.get(int(row["game_id"]), {}).get(f"{side}_player_ids", [])
            }
        )
        if not player_ids:
            continue
        for response in client.fetch_people_batting_game_logs(player_ids, int(season), batch_size=50, refresh=refresh):
            sources.append(_source_record(response, kind="current_season_batting_game_log", season=int(season)))
            for person in response.payload.get("people", []):
                if not isinstance(person, Mapping) or not person.get("id"):
                    continue
                splits: list[dict[str, Any]] = []
                for group in person.get("stats", []):
                    if not isinstance(group, Mapping):
                        continue
                    for split in group.get("splits", []):
                        if not isinstance(split, Mapping) or not split.get("date"):
                            continue
                        game = split.get("game", {}) if isinstance(split.get("game"), Mapping) else {}
                        splits.append(
                            {
                                "date": str(split["date"]),
                                "game_id": game.get("gamePk"),
                                "stats": normalize_batter_stats_payload({"stats": [{"splits": [split]}]}),
                            }
                        )
                logs[(int(season), int(person["id"]))] = sorted(
                    splits,
                    key=lambda item: (item["date"], int(item.get("game_id") or 0)),
                )
    return logs, sources


def collect_prior_season_batting_platoon_stats(
    *,
    client: MlbStatsApiClient,
    rows: Sequence[Mapping[str, Any]],
    lineups: Mapping[int, Mapping[str, Any]],
    target_seasons: Sequence[int],
    refresh: bool = False,
) -> tuple[dict[tuple[int, int], dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    stats: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for season in target_seasons:
        player_ids = sorted(
            {
                int(player_id)
                for row in rows
                if int(row["season"]) == int(season)
                for side in ("away", "home")
                for player_id in lineups.get(int(row["game_id"]), {}).get(f"{side}_player_ids", [])
            }
        )
        if not player_ids:
            continue
        stats_season = int(season) - 1
        for response in client.fetch_people_batting_platoon_stats(player_ids, stats_season, refresh=refresh):
            sources.append(_source_record(response, kind="prior_season_batting_platoon", season=stats_season))
            for person in response.payload.get("people", []):
                if not isinstance(person, Mapping) or not person.get("id"):
                    continue
                splits_by_code: dict[str, dict[str, Any]] = {}
                for group in person.get("stats", []):
                    if not isinstance(group, Mapping):
                        continue
                    for split in group.get("splits", []):
                        if not isinstance(split, Mapping):
                            continue
                        split_metadata = split.get("split", {}) if isinstance(split.get("split"), Mapping) else {}
                        code = str(split_metadata.get("code") or "")
                        if code in {"vl", "vr"}:
                            splits_by_code[code] = normalize_batter_stats_payload(
                                {"stats": [{"splits": [split]}]}
                            )
                stats[(stats_season, int(person["id"]))] = splits_by_code
    return stats, sources


def _combine_batting_stats(
    previous: Mapping[str, Any] | None,
    appearances: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = (
        "plate_appearances",
        "at_bats",
        "hits",
        "doubles",
        "triples",
        "home_runs",
        "walks",
        "hit_by_pitch",
        "sac_flies",
    )
    totals = {field: int((previous or {}).get(field, 0) or 0) for field in fields}
    has_history = bool((previous or {}).get("has_history"))
    for appearance in appearances:
        stats = appearance.get("stats", {}) if isinstance(appearance.get("stats"), Mapping) else {}
        has_history = has_history or bool(stats.get("has_history"))
        for field in fields:
            totals[field] += int(stats.get(field, 0) or 0)
    totals["has_history"] = has_history
    return totals


def smoothed_batter_ops(stats: Mapping[str, Any]) -> float:
    hits = float(stats.get("hits", 0) or 0)
    walks = float(stats.get("walks", 0) or 0)
    hit_by_pitch = float(stats.get("hit_by_pitch", 0) or 0)
    at_bats = float(stats.get("at_bats", 0) or 0)
    sac_flies = float(stats.get("sac_flies", 0) or 0)
    doubles = float(stats.get("doubles", 0) or 0)
    triples = float(stats.get("triples", 0) or 0)
    home_runs = float(stats.get("home_runs", 0) or 0)
    obp_denominator = at_bats + walks + hit_by_pitch + sac_flies
    total_bases = hits + doubles + 2.0 * triples + 3.0 * home_runs
    obp = (
        hits + walks + hit_by_pitch + LEAGUE_OBP_PRIOR * BATTING_PRIOR_PLATE_APPEARANCES
    ) / (obp_denominator + BATTING_PRIOR_PLATE_APPEARANCES)
    slg = (
        total_bases + LEAGUE_SLG_PRIOR * BATTING_PRIOR_PLATE_APPEARANCES
    ) / (at_bats + BATTING_PRIOR_PLATE_APPEARANCES)
    return obp + slg


def smoothed_platoon_ops(
    split_stats: Mapping[str, Any] | None,
    overall_stats: Mapping[str, Any] | None,
    *,
    prior_plate_appearances: float = 100.0,
) -> float:
    overall = smoothed_batter_ops(overall_stats or {})
    split = split_stats or {}
    plate_appearances = float(split.get("plate_appearances", 0) or 0)
    if plate_appearances <= 0:
        return overall
    at_bats = float(split.get("at_bats", 0) or 0)
    hits = float(split.get("hits", 0) or 0)
    walks = float(split.get("walks", 0) or 0)
    hit_by_pitch = float(split.get("hit_by_pitch", 0) or 0)
    sac_flies = float(split.get("sac_flies", 0) or 0)
    doubles = float(split.get("doubles", 0) or 0)
    triples = float(split.get("triples", 0) or 0)
    home_runs = float(split.get("home_runs", 0) or 0)
    obp_denominator = at_bats + walks + hit_by_pitch + sac_flies
    total_bases = hits + doubles + 2.0 * triples + 3.0 * home_runs
    observed_obp = (hits + walks + hit_by_pitch) / obp_denominator if obp_denominator > 0 else overall / 2.0
    observed_slg = total_bases / at_bats if at_bats > 0 else overall / 2.0
    observed_ops = observed_obp + observed_slg
    return (
        observed_ops * plate_appearances + overall * prior_plate_appearances
    ) / (plate_appearances + prior_plate_appearances)


def add_lineup_features(
    rows: Iterable[Mapping[str, Any]],
    lineups: Mapping[int, Mapping[str, Any]],
    prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
    game_logs: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        season = int(row["season"])
        official_date = str(row["official_date"])
        lineup = lineups.get(int(row["game_id"]), {})
        side_values: dict[str, dict[str, float]] = {}
        for side in ("away", "home"):
            player_ids = [int(value) for value in lineup.get(f"{side}_player_ids", [])][:LINEUP_SIZE]
            ops_values: list[float] = []
            missing = 0
            for player_id in player_ids:
                current_appearances = [
                    appearance
                    for appearance in game_logs.get((season, player_id), [])
                    if str(appearance["date"]) < official_date
                ]
                combined = _combine_batting_stats(prior_stats.get((season - 1, player_id)), current_appearances)
                missing += int(not combined["has_history"])
                ops_values.append(smoothed_batter_ops(combined))
            missing += max(0, LINEUP_SIZE - len(player_ids))
            if len(ops_values) < LINEUP_SIZE:
                ops_values.extend([LEAGUE_OBP_PRIOR + LEAGUE_SLG_PRIOR] * (LINEUP_SIZE - len(ops_values)))
            weighted_ops = sum(value * weight for value, weight in zip(ops_values, LINEUP_WEIGHTS)) / sum(LINEUP_WEIGHTS)
            top4_ops = sum(ops_values[:4]) / 4.0
            side_values[side] = {
                "weighted_ops": weighted_ops,
                "top4_ops": top4_ops,
                "missing_rate": float(missing) / LINEUP_SIZE,
                "player_count": len(player_ids),
            }
        row.update(
            {
                "lineup_feature_provenance": "retrospective_schedule_lineups_prior_plus_past_game_logs_v1",
                "lineup_stats_through_policy": "strictly_before_official_date",
                "lineup_identity_point_in_time_verified": False,
                "away_lineup_player_count": int(side_values["away"]["player_count"]),
                "home_lineup_player_count": int(side_values["home"]["player_count"]),
                "away_lineup_ops": side_values["away"]["weighted_ops"],
                "home_lineup_ops": side_values["home"]["weighted_ops"],
                "lineup_ops_advantage": side_values["home"]["weighted_ops"] - side_values["away"]["weighted_ops"],
                "lineup_top4_ops_advantage": side_values["home"]["top4_ops"] - side_values["away"]["top4_ops"],
                "away_lineup_history_missing_rate": side_values["away"]["missing_rate"],
                "home_lineup_history_missing_rate": side_values["home"]["missing_rate"],
            }
        )
        augmented.append(row)
    return augmented


def add_neutral_lineup_features(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "lineup_ops_advantage": 0.0,
            "lineup_top4_ops_advantage": 0.0,
            "away_lineup_history_missing_rate": 1.0,
            "home_lineup_history_missing_rate": 1.0,
        }
        for row in rows
    ]
