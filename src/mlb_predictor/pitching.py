from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .collector import CachedPayload, MlbStatsApiClient
from .normalizer import normalize_schedule_payloads


SNAPSHOT_SCHEMA_VERSION = "pregame-pitching-snapshot-v2"
PROVENANCE_MODE = "verified_point_in_time"
PITCHER_RATE_PRIOR_BATTERS = 200.0
PITCHER_RATE_PRIORS = {
    "strikeouts": 0.225,
    "walks": 0.085,
    "home_runs": 0.030,
    "earned_runs": 0.105,
}


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC timestamp must include an offset: {value}")
    return parsed.astimezone(timezone.utc)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def innings_pitched_to_outs(value: Any) -> int:
    text = str(value or "0")
    whole_text, dot, fractional_text = text.partition(".")
    try:
        whole = int(whole_text)
        fractional = int(fractional_text or "0") if dot else 0
    except ValueError as exc:
        raise ValueError(f"Invalid inningsPitched value: {value}") from exc
    if whole < 0 or fractional not in {0, 1, 2}:
        raise ValueError(f"Invalid inningsPitched value: {value}")
    return whole * 3 + fractional


def normalize_pitcher_stats_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an MLB byDateRange pitching response without inventing missing history."""

    splits: list[Mapping[str, Any]] = []
    for group in payload.get("stats", []):
        if isinstance(group, Mapping) and isinstance(group.get("splits"), list):
            splits.extend(split for split in group["splits"] if isinstance(split, Mapping))
    stat = splits[0].get("stat", {}) if splits and isinstance(splits[0].get("stat"), Mapping) else {}
    has_history = bool(splits)
    return {
        "has_history": has_history,
        "games_played": int(_number(stat.get("gamesPlayed"))),
        "games_started": int(_number(stat.get("gamesStarted"))),
        "innings_pitched": str(stat.get("inningsPitched") or "0.0"),
        "innings_pitched_outs": innings_pitched_to_outs(stat.get("inningsPitched")),
        "batters_faced": int(_number(stat.get("battersFaced"))),
        "pitches_thrown": int(_number(stat.get("numberOfPitches") or stat.get("pitchesThrown"))),
        "strikeouts": int(_number(stat.get("strikeOuts"))),
        "walks": int(_number(stat.get("baseOnBalls"))),
        "home_runs": int(_number(stat.get("homeRuns"))),
        "hits": int(_number(stat.get("hits"))),
        "earned_runs": int(_number(stat.get("earnedRuns"))),
    }


def make_pitcher_evidence(
    *,
    player_id: int,
    payload: Mapping[str, Any],
    stats_start_date: str,
    stats_through_date: str,
    fetched_at_utc: str | None,
    source_url: str,
    response_sha256: str | None,
) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "stats_start_date": stats_start_date,
        "stats_through_date": stats_through_date,
        "fetched_at_utc": fetched_at_utc,
        "source_url": source_url,
        "response_sha256": response_sha256,
        "stats": normalize_pitcher_stats_payload(payload),
    }


def extract_team_bullpen_usage(
    boxscore: Mapping[str, Any],
    *,
    side: str,
    game_id: int,
    official_date: str,
    team_id: int,
) -> dict[str, Any]:
    """Extract reliever workload from a completed prior-game boxscore."""

    if side not in {"home", "away"}:
        raise ValueError("side must be home or away.")
    team = boxscore.get("teams", {}).get(side, {}) if isinstance(boxscore.get("teams"), Mapping) else {}
    players = team.get("players", {}) if isinstance(team, Mapping) and isinstance(team.get("players"), Mapping) else {}
    pitcher_ids = team.get("pitchers", []) if isinstance(team, Mapping) and isinstance(team.get("pitchers"), list) else []
    relievers: list[dict[str, Any]] = []
    for raw_pitcher_id in pitcher_ids:
        pitcher_id = int(raw_pitcher_id)
        player = players.get(f"ID{pitcher_id}", {})
        pitching = (
            player.get("stats", {}).get("pitching", {})
            if isinstance(player, Mapping) and isinstance(player.get("stats"), Mapping)
            else {}
        )
        if not isinstance(pitching, Mapping) or int(_number(pitching.get("gamesStarted"))) > 0:
            continue
        relievers.append(
            {
                "player_id": pitcher_id,
                "name": player.get("person", {}).get("fullName") if isinstance(player.get("person"), Mapping) else None,
                "pitches_thrown": int(_number(pitching.get("pitchesThrown") or pitching.get("numberOfPitches"))),
                "batters_faced": int(_number(pitching.get("battersFaced"))),
            }
        )
    return {
        "game_id": game_id,
        "official_date": official_date,
        "team_id": team_id,
        "relievers": relievers,
    }


def aggregate_bullpen_usage(
    usage_rows: Iterable[Mapping[str, Any]],
    *,
    team_id: int,
    target_date: str,
) -> dict[str, Any]:
    target = date.fromisoformat(target_date)
    totals: dict[int, dict[str, Any]] = {}
    game_ids: list[int] = []
    for row in usage_rows:
        if int(row.get("team_id", -1)) != team_id:
            continue
        game_date = date.fromisoformat(str(row["official_date"]))
        days_ago = (target - game_date).days
        if not 1 <= days_ago <= 3:
            continue
        game_ids.append(int(row["game_id"]))
        for reliever in row.get("relievers", []):
            if not isinstance(reliever, Mapping):
                continue
            player_id = int(reliever["player_id"])
            entry = totals.setdefault(
                player_id,
                {"player_id": player_id, "name": reliever.get("name"), "pitches_1d": 0, "pitches_2d": 0, "pitches_3d": 0, "batters_faced_3d": 0},
            )
            pitches = int(_number(reliever.get("pitches_thrown")))
            batters = int(_number(reliever.get("batters_faced")))
            if days_ago <= 1:
                entry["pitches_1d"] += pitches
            if days_ago <= 2:
                entry["pitches_2d"] += pitches
            entry["pitches_3d"] += pitches
            entry["batters_faced_3d"] += batters
    relievers = sorted(totals.values(), key=lambda item: (-item["pitches_3d"], item["player_id"]))
    return {
        "team_id": team_id,
        "data_through_date": (target - timedelta(days=1)).isoformat(),
        "source_game_ids": sorted(set(game_ids)),
        "relievers": relievers,
        "team_pitches_1d": sum(item["pitches_1d"] for item in relievers),
        "team_pitches_2d": sum(item["pitches_2d"] for item in relievers),
        "team_pitches_3d": sum(item["pitches_3d"] for item in relievers),
        "team_batters_faced_3d": sum(item["batters_faced_3d"] for item in relievers),
    }


def validate_pregame_pitching_snapshot(snapshot: Mapping[str, Any], *, lead_minutes: int = 60) -> list[str]:
    errors: list[str] = []
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    try:
        game_start = _parse_utc(str(snapshot["game_start_utc"]))
        created_at = _parse_utc(str(snapshot["created_at_utc"]))
        cutoff = game_start - timedelta(minutes=lead_minutes)
        if created_at > cutoff:
            errors.append("snapshot_created_after_eligibility_cutoff")
    except (KeyError, TypeError, ValueError):
        errors.append("timestamp_invalid")
        cutoff = None

    try:
        official_date = date.fromisoformat(str(snapshot["official_date"]))
    except (KeyError, ValueError):
        errors.append("official_date_invalid")
        official_date = None

    for source in snapshot.get("sources", []):
        if not isinstance(source, Mapping):
            errors.append("source_invalid")
            continue
        observed = source.get("fetched_at_utc")
        if not observed:
            errors.append("source_fetched_at_missing")
        elif cutoff is not None:
            try:
                observed_at = _parse_utc(str(observed))
                if observed_at > cutoff:
                    errors.append("source_fetched_after_eligibility_cutoff")
                if observed_at > created_at:
                    errors.append("source_fetched_after_snapshot_created")
            except ValueError:
                errors.append("source_fetched_at_invalid")
        if not source.get("source_url") or not source.get("response_sha256"):
            errors.append("source_provenance_incomplete")

    for side in ("away", "home"):
        starter = snapshot.get("starters", {}).get(side, {}) if isinstance(snapshot.get("starters"), Mapping) else {}
        if not starter.get("player_id"):
            if not starter.get("missing_reason"):
                errors.append(f"{side}_starter_missing_without_reason")
        else:
            stats_through = starter.get("stats_through_date")
            try:
                if official_date is not None and date.fromisoformat(str(stats_through)) >= official_date:
                    errors.append(f"{side}_starter_stats_not_past_only")
            except ValueError:
                errors.append(f"{side}_starter_stats_date_invalid")

        bullpen = snapshot.get("bullpen", {}).get(side, {}) if isinstance(snapshot.get("bullpen"), Mapping) else {}
        for game_id in bullpen.get("source_game_ids", []):
            if int(game_id) == int(snapshot.get("game_id", -1)):
                errors.append(f"{side}_bullpen_includes_target_game")
        try:
            if official_date is not None and date.fromisoformat(str(bullpen.get("data_through_date"))) >= official_date:
                errors.append(f"{side}_bullpen_not_past_only")
        except ValueError:
            errors.append(f"{side}_bullpen_date_invalid")
    return sorted(set(errors))


def create_pregame_pitching_snapshot(
    *,
    scheduled_game: Mapping[str, Any],
    starter_evidence: Mapping[int, Mapping[str, Any]],
    bullpen_by_team: Mapping[int, Mapping[str, Any]],
    sources: Iterable[Mapping[str, Any]],
    created_at_utc: str,
) -> dict[str, Any]:
    starters: dict[str, dict[str, Any]] = {}
    bullpen: dict[str, dict[str, Any]] = {}
    for side in ("away", "home"):
        raw_pitcher_id = scheduled_game.get(f"{side}_probable_pitcher_id")
        if raw_pitcher_id is None:
            starters[side] = {"player_id": None, "name": scheduled_game.get(f"{side}_probable_pitcher_name"), "missing_reason": "probable_pitcher_not_announced"}
        else:
            pitcher_id = int(raw_pitcher_id)
            evidence = starter_evidence.get(pitcher_id)
            if evidence is None:
                starters[side] = {"player_id": pitcher_id, "name": scheduled_game.get(f"{side}_probable_pitcher_name"), "missing_reason": "pitcher_stats_unavailable"}
            else:
                starters[side] = {"name": scheduled_game.get(f"{side}_probable_pitcher_name"), **dict(evidence)}
        team_id = int(scheduled_game[f"{side}_team_id"])
        bullpen[side] = dict(bullpen_by_team.get(team_id, {"team_id": team_id, "data_through_date": None, "source_game_ids": [], "relievers": []}))

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "provenance_mode": PROVENANCE_MODE,
        "game_id": int(scheduled_game["game_id"]),
        "official_date": str(scheduled_game["official_date"]),
        "game_start_utc": str(scheduled_game["game_start_utc"]),
        "created_at_utc": created_at_utc,
        "away_team_id": int(scheduled_game["away_team_id"]),
        "home_team_id": int(scheduled_game["home_team_id"]),
        "starters": starters,
        "bullpen": bullpen,
        "sources": [dict(source) for source in sources],
    }
    errors = validate_pregame_pitching_snapshot(snapshot)
    snapshot["quality"] = {"status": "passed" if not errors else "failed", "errors": errors}
    return snapshot


def _source_record(payload: CachedPayload, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "fetched_at_utc": payload.fetched_at_utc,
        "source_url": payload.source_url,
        "response_sha256": payload.response_sha256,
    }


def collect_pregame_pitching_snapshots(
    *,
    client: MlbStatsApiClient,
    scheduled_games: Iterable[Mapping[str, Any]],
    schedule_sources: Iterable[CachedPayload],
    target_date: str,
    created_at_utc: str | None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Collect a prospective, timestamped challenger snapshot for a single date."""

    target = date.fromisoformat(target_date)
    games = [dict(game) for game in scheduled_games]
    relevant_team_ids = {
        int(game[f"{side}_team_id"])
        for game in games
        for side in ("away", "home")
    }
    base_sources = [_source_record(payload, kind="target_schedule") for payload in schedule_sources]

    history_start = target - timedelta(days=3)
    history_end = target - timedelta(days=1)
    history_payloads = client.fetch_schedule(history_start, history_end, chunk_days=3, refresh=refresh)
    base_sources.extend(_source_record(payload, kind="prior_schedule") for payload in history_payloads)
    prior_games, _ = normalize_schedule_payloads(
        [payload.payload for payload in history_payloads],
        start_date=history_start,
        end_date=history_end,
    )

    usage_rows: list[dict[str, Any]] = []
    bullpen_sources_by_team: dict[int, list[dict[str, Any]]] = {team_id: [] for team_id in relevant_team_ids}
    for prior_game in prior_games:
        sides = [side for side in ("away", "home") if int(prior_game[f"{side}_team_id"]) in relevant_team_ids]
        if not sides:
            continue
        boxscore = client.fetch_game_boxscore(int(prior_game["game_id"]), refresh=refresh)
        boxscore_source = _source_record(boxscore, kind="prior_boxscore")
        for side in sides:
            team_id = int(prior_game[f"{side}_team_id"])
            bullpen_sources_by_team[team_id].append(boxscore_source)
            usage_rows.append(
                extract_team_bullpen_usage(
                    boxscore.payload,
                    side=side,
                    game_id=int(prior_game["game_id"]),
                    official_date=str(prior_game["official_date"]),
                    team_id=team_id,
                )
            )

    bullpen_by_team = {
        team_id: aggregate_bullpen_usage(usage_rows, team_id=team_id, target_date=target_date)
        for team_id in relevant_team_ids
    }
    pitcher_ids = {
        int(pitcher_id)
        for game in games
        for pitcher_id in (game.get("away_probable_pitcher_id"), game.get("home_probable_pitcher_id"))
        if pitcher_id is not None
    }
    stats_start = target - timedelta(days=365)
    stats_end = target - timedelta(days=1)
    starter_evidence: dict[int, dict[str, Any]] = {}
    pitcher_sources: dict[int, dict[str, Any]] = {}
    for pitcher_id in sorted(pitcher_ids):
        response = client.fetch_player_pitching_stats(pitcher_id, stats_start, stats_end, refresh=refresh)
        pitcher_sources[pitcher_id] = _source_record(response, kind="pitcher_stats")
        starter_evidence[pitcher_id] = make_pitcher_evidence(
            player_id=pitcher_id,
            payload=response.payload,
            stats_start_date=stats_start.isoformat(),
            stats_through_date=stats_end.isoformat(),
            fetched_at_utc=response.fetched_at_utc,
            source_url=response.source_url,
            response_sha256=response.response_sha256,
        )

    snapshot_created_at = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    snapshots: list[dict[str, Any]] = []
    for game in games:
        game_sources = list(base_sources)
        for side in ("away", "home"):
            team_id = int(game[f"{side}_team_id"])
            game_sources.extend(bullpen_sources_by_team[team_id])
            pitcher_id = game.get(f"{side}_probable_pitcher_id")
            if pitcher_id is not None and int(pitcher_id) in pitcher_sources:
                game_sources.append(pitcher_sources[int(pitcher_id)])
        unique_sources = {
            (source.get("source_url"), source.get("response_sha256")): source
            for source in game_sources
        }
        snapshots.append(create_pregame_pitching_snapshot(
            scheduled_game=game,
            starter_evidence=starter_evidence,
            bullpen_by_team=bullpen_by_team,
            sources=unique_sources.values(),
            created_at_utc=snapshot_created_at,
        ))
    return snapshots


def write_pregame_pitching_snapshots(
    snapshots: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
) -> list[Path]:
    written: list[Path] = []
    for snapshot in snapshots:
        target_dir = Path(output_dir) / str(snapshot["official_date"])
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"pitching-v2-{int(snapshot['game_id'])}.json"
        if path.exists():
            raise FileExistsError(f"A sealed pitching snapshot already exists: {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(snapshot), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        written.append(path)
    return written


def smoothed_pitcher_rate(stats: Mapping[str, Any], field: str) -> float:
    batters_faced = max(0.0, _number(stats.get("batters_faced")))
    observed = max(0.0, _number(stats.get(field)))
    prior = PITCHER_RATE_PRIORS[field]
    return (observed + prior * PITCHER_RATE_PRIOR_BATTERS) / (batters_faced + PITCHER_RATE_PRIOR_BATTERS)


def pitching_snapshot_to_features(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Create stable challenger features; this never changes the frozen model-v1 vector."""

    errors = validate_pregame_pitching_snapshot(snapshot)
    if errors or snapshot.get("quality", {}).get("status") != "passed":
        raise ValueError(f"Pitching snapshot failed validation: {errors or snapshot.get('quality')}")

    side_rates: dict[str, dict[str, float]] = {}
    missing: dict[str, int] = {}
    for side in ("away", "home"):
        starter = snapshot["starters"][side]
        stats = starter.get("stats", {}) if isinstance(starter, Mapping) else {}
        missing[side] = int(not starter.get("player_id") or not stats.get("has_history"))
        side_rates[side] = {
            field: smoothed_pitcher_rate(stats, field)
            for field in PITCHER_RATE_PRIORS
        }

    away_bullpen = snapshot["bullpen"]["away"]
    home_bullpen = snapshot["bullpen"]["home"]
    return {
        "game_id": int(snapshot["game_id"]),
        "official_date": str(snapshot["official_date"]),
        "pitching_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "starter_k_minus_bb_rate_difference": (
            side_rates["home"]["strikeouts"] - side_rates["home"]["walks"]
            - side_rates["away"]["strikeouts"] + side_rates["away"]["walks"]
        ),
        "starter_earned_run_rate_advantage": side_rates["away"]["earned_runs"] - side_rates["home"]["earned_runs"],
        "starter_home_run_rate_advantage": side_rates["away"]["home_runs"] - side_rates["home"]["home_runs"],
        "away_starter_history_missing": missing["away"],
        "home_starter_history_missing": missing["home"],
        "bullpen_pitches_1d_advantage": int(away_bullpen.get("team_pitches_1d", 0)) - int(home_bullpen.get("team_pitches_1d", 0)),
        "bullpen_pitches_3d_advantage": int(away_bullpen.get("team_pitches_3d", 0)) - int(home_bullpen.get("team_pitches_3d", 0)),
        "bullpen_batters_faced_3d_advantage": int(away_bullpen.get("team_batters_faced_3d", 0)) - int(home_bullpen.get("team_batters_faced_3d", 0)),
    }
