from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bullpen_backtest import (
    add_probabilistic_reliever_features,
    add_reliever_availability_features,
    collect_reliever_pitching_data,
)
from .collector import CachedPayload, MlbStatsApiClient
from .features import build_forecast_features
from .lineup import (
    add_lineup_features,
    collect_current_season_batting_game_logs,
    collect_historical_lineups,
    collect_prior_season_batting_platoon_stats,
    collect_prior_season_batting_stats,
)
from .model_v3_backtest import (
    MODEL_V3_FEATURE_SPECS,
    add_context_features,
    add_interaction_features,
    add_platoon_features,
    add_platoon_performance_features,
    add_recent_starter_form_features,
    add_starter_readiness_features,
)
from .normalizer import normalize_future_schedule_payloads, normalize_schedule_payloads
from .pitching_backtest import (
    add_rolling_starter_features,
    collect_current_season_pitching_game_logs,
    collect_prior_season_pitching_stats,
)
from .run_strength import add_dynamic_run_strength_features
from .schedule_context import add_schedule_context_features, add_schedule_context_interactions


SNAPSHOT_SCHEMA_VERSION = "pregame-model-v3-snapshot-v7"
PROVENANCE_MODE = "verified_point_in_time"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC timestamp must include an offset: {value}")
    return parsed.astimezone(timezone.utc)


def _source_record(payload: CachedPayload, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "fetched_at_utc": payload.fetched_at_utc,
        "source_url": payload.source_url,
        "response_sha256": payload.response_sha256,
    }


def validate_pregame_model_v3_snapshot(
    snapshot: Mapping[str, Any],
    *,
    lead_minutes: int = 60,
) -> list[str]:
    errors: list[str] = []
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if snapshot.get("provenance_mode") != PROVENANCE_MODE:
        errors.append("provenance_mode_invalid")

    cutoff: datetime | None = None
    created_at: datetime | None = None
    try:
        game_start = _parse_utc(str(snapshot["game_start_utc"]))
        created_at = _parse_utc(str(snapshot["created_at_utc"]))
        cutoff = game_start - timedelta(minutes=lead_minutes)
        if created_at > cutoff:
            errors.append("snapshot_created_after_eligibility_cutoff")
    except (KeyError, TypeError, ValueError):
        errors.append("timestamp_invalid")

    try:
        official_date = date.fromisoformat(str(snapshot["official_date"]))
        if date.fromisoformat(str(snapshot["data_through_date"])) >= official_date:
            errors.append("data_not_past_only")
    except (KeyError, TypeError, ValueError):
        errors.append("data_through_date_invalid")

    sources = snapshot.get("sources", [])
    if not isinstance(sources, list) or not sources:
        errors.append("sources_missing")
    else:
        for source in sources:
            if not isinstance(source, Mapping):
                errors.append("source_invalid")
                continue
            if not source.get("source_url") or not source.get("response_sha256"):
                errors.append("source_provenance_incomplete")
            observed = source.get("fetched_at_utc")
            if not observed:
                errors.append("source_fetched_at_missing")
                continue
            try:
                observed_at = _parse_utc(str(observed))
                if cutoff is not None and observed_at > cutoff:
                    errors.append("source_fetched_after_eligibility_cutoff")
                if created_at is not None and observed_at > created_at:
                    errors.append("source_fetched_after_snapshot_created")
            except ValueError:
                errors.append("source_fetched_at_invalid")

    for side in ("away", "home"):
        lineup = snapshot.get("lineups", {}).get(side, {})
        player_ids = lineup.get("player_ids", []) if isinstance(lineup, Mapping) else []
        if len(player_ids) != 9 or len(set(player_ids)) != 9:
            errors.append(f"{side}_lineup_incomplete")
        if lineup.get("status") not in {"confirmed", "expected_from_last_completed_game"}:
            errors.append(f"{side}_lineup_status_invalid")
        if not lineup.get("identity_point_in_time_verified"):
            errors.append(f"{side}_lineup_identity_unverified")
        roster = snapshot.get("active_rosters", {}).get(side, {})
        pitcher_ids = roster.get("pitcher_ids", []) if isinstance(roster, Mapping) else []
        if not pitcher_ids or not roster.get("identity_point_in_time_verified"):
            errors.append(f"{side}_active_roster_unverified")
        if int(snapshot.get("features", {}).get(f"{side}_known_core_reliever_count", 0)) < 4:
            errors.append(f"{side}_core_relievers_incomplete")
        if not snapshot.get(f"{side}_probable_pitcher_id"):
            errors.append(f"{side}_probable_pitcher_missing")

    features = snapshot.get("features", {})
    if not isinstance(features, Mapping):
        errors.append("features_invalid")
    else:
        for spec in MODEL_V3_FEATURE_SPECS:
            if spec.source not in features or features[spec.source] is None:
                errors.append(f"feature_missing:{spec.source}")
                continue
            if spec.transform == "numeric":
                try:
                    if not math.isfinite(float(features[spec.source])):
                        errors.append(f"feature_nonfinite:{spec.source}")
                except (TypeError, ValueError):
                    errors.append(f"feature_non_numeric:{spec.source}")
    return sorted(set(errors))


def create_pregame_model_v3_snapshot(
    *,
    scheduled_game: Mapping[str, Any],
    feature_row: Mapping[str, Any],
    lineup: Mapping[str, Any],
    active_rosters: Mapping[str, Mapping[str, Any]],
    sources: Iterable[Mapping[str, Any]],
    created_at_utc: str,
) -> dict[str, Any]:
    feature_sources = {spec.source for spec in MODEL_V3_FEATURE_SPECS}
    features = {name: feature_row.get(name) for name in sorted(feature_sources)}
    features.update(
        {
            "away_known_core_reliever_count": int(feature_row.get("away_known_core_reliever_count", 0)),
            "home_known_core_reliever_count": int(feature_row.get("home_known_core_reliever_count", 0)),
        }
    )
    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "provenance_mode": PROVENANCE_MODE,
        "game_id": int(scheduled_game["game_id"]),
        "official_date": str(scheduled_game["official_date"]),
        "game_start_utc": str(scheduled_game["game_start_utc"]),
        "created_at_utc": created_at_utc,
        "data_through_date": (date.fromisoformat(str(scheduled_game["official_date"])) - timedelta(days=1)).isoformat(),
        "away_team_id": int(scheduled_game["away_team_id"]),
        "home_team_id": int(scheduled_game["home_team_id"]),
        "away_probable_pitcher_id": scheduled_game.get("away_probable_pitcher_id"),
        "home_probable_pitcher_id": scheduled_game.get("home_probable_pitcher_id"),
        "lineups": {
            side: {
                "player_ids": [int(value) for value in lineup.get(f"{side}_player_ids", [])[:9]],
                "status": lineup.get(f"{side}_status", "confirmed"),
                "identity_point_in_time_verified": True,
            }
            for side in ("away", "home")
        },
        "active_rosters": {
            side: {
                "pitcher_ids": [int(value) for value in active_rosters.get(side, {}).get("pitcher_ids", [])],
                "identity_point_in_time_verified": True,
            }
            for side in ("away", "home")
        },
        "features": features,
        "sources": [dict(source) for source in sources],
    }
    errors = validate_pregame_model_v3_snapshot(snapshot)
    snapshot["quality"] = {"status": "passed" if not errors else "failed", "errors": errors}
    return snapshot


def _active_pitcher_rosters(
    client: MlbStatsApiClient,
    games: Sequence[Mapping[str, Any]],
    *,
    season: int,
    refresh: bool,
) -> tuple[dict[tuple[int, int], list[int]], list[dict[str, Any]]]:
    roster_map: dict[tuple[int, int], list[int]] = {}
    sources: list[dict[str, Any]] = []
    team_ids = sorted(
        {
            int(game[f"{side}_team_id"])
            for game in games
            for side in ("away", "home")
        }
    )
    for team_id in team_ids:
        response = client.fetch_team_active_roster(team_id, season, refresh=refresh)
        sources.append(_source_record(response, kind="active_roster"))
        roster_map[(season, team_id)] = sorted(
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
    return roster_map, sources


def _expected_lineups_from_last_completed_games(
    client: MlbStatsApiClient,
    games: Sequence[Mapping[str, Any]],
    *,
    target: date,
    refresh: bool,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    relevant_team_ids = {
        int(game[f"{side}_team_id"])
        for game in games
        for side in ("away", "home")
    }
    history_start = target - timedelta(days=7)
    history_end = target - timedelta(days=1)
    schedule_payloads = client.fetch_schedule(
        history_start,
        history_end,
        chunk_days=7,
        refresh=refresh,
    )
    sources = [_source_record(payload, kind="expected_lineup_prior_schedule") for payload in schedule_payloads]
    prior_games, _ = normalize_schedule_payloads(
        [payload.payload for payload in schedule_payloads],
        start_date=history_start,
        end_date=history_end,
    )
    expected: dict[int, list[int]] = {}
    boxscores: dict[int, CachedPayload] = {}
    for prior_game in sorted(
        prior_games,
        key=lambda row: (str(row["official_date"]), str(row.get("game_start_utc") or ""), int(row["game_id"])),
        reverse=True,
    ):
        for side in ("away", "home"):
            team_id = int(prior_game[f"{side}_team_id"])
            if team_id not in relevant_team_ids or team_id in expected:
                continue
            game_id = int(prior_game["game_id"])
            if game_id not in boxscores:
                boxscores[game_id] = client.fetch_game_boxscore(game_id, refresh=refresh)
                sources.append(_source_record(boxscores[game_id], kind="expected_lineup_prior_boxscore"))
            team = boxscores[game_id].payload.get("teams", {}).get(side, {})
            raw_order = team.get("battingOrder", []) if isinstance(team, Mapping) else []
            player_ids = [int(value) for value in raw_order if str(value).isdigit()]
            if len(player_ids) >= 9 and len(set(player_ids[:9])) == 9:
                expected[team_id] = player_ids[:9]
        if relevant_team_ids.issubset(expected):
            break
    return expected, sources


def collect_pregame_model_v3_snapshots(
    *,
    client: MlbStatsApiClient,
    completed_games: Iterable[Mapping[str, Any]],
    target_date: str,
    created_at_utc: str | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Collect fail-closed, prospective model-v3 feature snapshots."""

    target = date.fromisoformat(target_date)
    schedule_payloads = client.fetch_schedule_lineups(target, target, chunk_days=1, refresh=refresh)
    scheduled, _ = normalize_future_schedule_payloads(
        [payload.payload for payload in schedule_payloads],
        target_date=target,
    )
    games = [game for game in scheduled if bool(game.get("forecast_eligible"))]
    if not games:
        return []
    history = list(completed_games)
    base_rows = build_forecast_features(history, games)
    base_rows = add_dynamic_run_strength_features(base_rows, history)
    base_rows = add_schedule_context_features(base_rows, history)
    season = int(target.year)
    target_seasons = [season]

    lineups, lineup_sources = collect_historical_lineups(
        client=client,
        rows=base_rows,
        refresh=refresh,
    )
    expected_lineups, expected_lineup_sources = _expected_lineups_from_last_completed_games(
        client,
        games,
        target=target,
        refresh=refresh,
    )
    resolved_lineups: dict[int, dict[str, Any]] = {}
    for game in games:
        game_id = int(game["game_id"])
        resolved: dict[str, Any] = {}
        for side in ("away", "home"):
            confirmed = [int(value) for value in lineups.get(game_id, {}).get(f"{side}_player_ids", [])]
            if len(confirmed) >= 9 and len(set(confirmed[:9])) == 9:
                resolved[f"{side}_player_ids"] = confirmed[:9]
                resolved[f"{side}_status"] = "confirmed"
            else:
                team_id = int(game[f"{side}_team_id"])
                resolved[f"{side}_player_ids"] = expected_lineups.get(team_id, [])
                resolved[f"{side}_status"] = "expected_from_last_completed_game"
        resolved_lineups[game_id] = resolved
    starter_prior, starter_prior_sources = collect_prior_season_pitching_stats(
        client=client,
        rows=base_rows,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    starter_logs, starter_log_sources = collect_current_season_pitching_game_logs(
        client=client,
        rows=base_rows,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    batting_prior, batting_prior_sources = collect_prior_season_batting_stats(
        client=client,
        rows=base_rows,
        lineups=resolved_lineups,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    batting_logs, batting_log_sources = collect_current_season_batting_game_logs(
        client=client,
        rows=base_rows,
        lineups=resolved_lineups,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    batting_platoon, batting_platoon_sources = collect_prior_season_batting_platoon_stats(
        client=client,
        rows=base_rows,
        lineups=resolved_lineups,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    rosters, roster_sources = _active_pitcher_rosters(
        client,
        games,
        season=season,
        refresh=refresh,
    )
    reliever_prior, reliever_logs, reliever_sources = collect_reliever_pitching_data(
        client=client,
        rosters=rosters,
        target_seasons=target_seasons,
        refresh=refresh,
    )

    augmented = add_context_features(base_rows)
    augmented = add_rolling_starter_features(augmented, starter_prior, starter_logs)
    augmented = add_starter_readiness_features(augmented, starter_prior, starter_logs)
    augmented = add_recent_starter_form_features(augmented, starter_logs)
    augmented = add_lineup_features(augmented, resolved_lineups, batting_prior, batting_logs)
    augmented = add_platoon_features(
        augmented,
        resolved_lineups,
        batting_prior,
        starter_prior,
    )
    augmented = add_platoon_performance_features(
        augmented,
        resolved_lineups,
        batting_prior,
        batting_platoon,
        starter_prior,
    )
    augmented = add_reliever_availability_features(augmented, rosters, reliever_prior, reliever_logs)
    augmented = add_probabilistic_reliever_features(
        augmented,
        rosters,
        reliever_prior,
        reliever_logs,
    )
    augmented = add_interaction_features(augmented)
    augmented = add_schedule_context_interactions(augmented)
    augmented_by_game = {int(row["game_id"]): row for row in augmented}

    all_sources = [
        *[_source_record(payload, kind="target_schedule_lineups") for payload in schedule_payloads],
        *lineup_sources,
        *expected_lineup_sources,
        *starter_prior_sources,
        *starter_log_sources,
        *batting_prior_sources,
        *batting_log_sources,
        *batting_platoon_sources,
        *roster_sources,
        *reliever_sources,
    ]
    unique_sources = {
        (source.get("source_url"), source.get("response_sha256")): source
        for source in all_sources
    }
    created = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    snapshots: list[dict[str, Any]] = []
    for game in games:
        side_rosters = {
            side: {"pitcher_ids": rosters.get((season, int(game[f"{side}_team_id"])), [])}
            for side in ("away", "home")
        }
        snapshots.append(
            create_pregame_model_v3_snapshot(
                scheduled_game=game,
                feature_row=augmented_by_game[int(game["game_id"])],
                lineup=resolved_lineups.get(int(game["game_id"]), {}),
                active_rosters=side_rosters,
                sources=unique_sources.values(),
                created_at_utc=created,
            )
        )
    return snapshots


def write_pregame_model_v3_snapshots(
    snapshots: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
) -> list[Path]:
    written: list[Path] = []
    for snapshot in snapshots:
        target_dir = Path(output_dir) / str(snapshot["official_date"])
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"model-v3-{int(snapshot['game_id'])}.json"
        if path.exists():
            raise FileExistsError(f"A sealed model-v3 snapshot already exists: {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(dict(snapshot), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        written.append(path)
    return written
