from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Iterable, Mapping


LEAGUE_TOTAL_RUNS_PRIOR = 9.0
LEAGUE_HOME_WIN_PRIOR = 0.54
VENUE_PRIOR_GAMES = 100.0

SCHEDULE_CONTEXT_FEATURE_NAMES = (
    "venue_home_win_advantage",
    "venue_run_environment",
    "venue_history_log_games",
    "schedule_consecutive_days_advantage",
    "schedule_no_rest_travel_advantage",
    "schedule_night_to_day_advantage",
    "away_road_trip_game_number",
)

SCHEDULE_CONTEXT_INTERACTION_NAMES = (
    "venue_lineup_ops_interaction",
    "venue_starter_kbb_interaction",
    "schedule_fatigue_elo_interaction",
)

VENUE_CONTEXT_FEATURE_NAMES = SCHEDULE_CONTEXT_FEATURE_NAMES[:3]
FATIGUE_CONTEXT_FEATURE_NAMES = SCHEDULE_CONTEXT_FEATURE_NAMES[3:]


@dataclass
class _TeamScheduleState:
    last_date: date | None = None
    last_venue_id: int | None = None
    last_day_night: str | None = None
    last_was_away: bool = False
    consecutive_days: int = 0
    away_streak: int = 0


@dataclass
class _VenueState:
    games: int = 0
    total_runs: float = 0.0
    home_wins: int = 0


def _pregame_team_context(
    state: _TeamScheduleState,
    *,
    current_date: date,
    venue_id: int | None,
    day_night: str,
    is_away: bool,
) -> dict[str, float]:
    days_since = (current_date - state.last_date).days if state.last_date else None
    consecutive_days = state.consecutive_days + 1 if days_since == 1 else 1
    changed_venue = (
        state.last_venue_id is not None
        and venue_id is not None
        and state.last_venue_id != venue_id
    )
    no_rest_travel = float(days_since == 1 and changed_venue)
    night_to_day = float(
        days_since == 1 and state.last_day_night == "night" and day_night == "day"
    )
    road_trip_game_number = (
        state.away_streak + 1
        if is_away and state.last_was_away and days_since is not None and days_since <= 3
        else 1 if is_away else 0
    )
    return {
        "consecutive_days": float(min(consecutive_days, 20)),
        "no_rest_travel": no_rest_travel,
        "night_to_day": night_to_day,
        "road_trip_game_number": float(min(road_trip_game_number, 20)),
    }


def add_schedule_context_features(
    feature_rows: Iterable[Mapping[str, Any]],
    completed_games: Iterable[Mapping[str, Any]],
    *,
    venue_prior_games: float = VENUE_PRIOR_GAMES,
) -> list[dict[str, Any]]:
    """Add venue and schedule features using only prior official dates."""

    if venue_prior_games <= 0:
        raise ValueError("venue_prior_games must be positive.")
    rows = sorted(
        (dict(row) for row in feature_rows),
        key=lambda row: (str(row["official_date"]), int(row["game_id"])),
    )
    rows_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_date[str(row["official_date"])].append(row)
    completed_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in completed_games:
        game = dict(source)
        if game.get("home_score") is None or game.get("away_score") is None:
            continue
        completed_by_date[str(game["official_date"])].append(game)

    team_states: dict[int, _TeamScheduleState] = defaultdict(_TeamScheduleState)
    venue_states: dict[int, _VenueState] = defaultdict(_VenueState)
    league_games = 0
    league_total_runs = 0.0
    league_home_wins = 0
    output: list[dict[str, Any]] = []

    for official_date in sorted(set(rows_by_date) | set(completed_by_date)):
        current_date = date.fromisoformat(official_date)
        for row in rows_by_date.get(official_date, []):
            venue_id = int(row["venue_id"]) if row.get("venue_id") is not None else None
            day_night = str(row.get("day_night") or "unknown")
            home_context = _pregame_team_context(
                team_states[int(row["home_team_id"])],
                current_date=current_date,
                venue_id=venue_id,
                day_night=day_night,
                is_away=False,
            )
            away_context = _pregame_team_context(
                team_states[int(row["away_team_id"])],
                current_date=current_date,
                venue_id=venue_id,
                day_night=day_night,
                is_away=True,
            )
            venue = venue_states[venue_id] if venue_id is not None else _VenueState()
            league_run_rate = (
                league_total_runs / league_games if league_games else LEAGUE_TOTAL_RUNS_PRIOR
            )
            league_home_rate = (
                league_home_wins / league_games if league_games else LEAGUE_HOME_WIN_PRIOR
            )
            venue_run_rate = (
                venue.total_runs + venue_prior_games * league_run_rate
            ) / (venue.games + venue_prior_games)
            venue_home_rate = (
                venue.home_wins + venue_prior_games * league_home_rate
            ) / (venue.games + venue_prior_games)
            materialized = dict(row)
            materialized.update(
                {
                    "schedule_context_provenance": "completed_games_strictly_before_official_date_v1",
                    "venue_home_win_advantage": venue_home_rate - league_home_rate,
                    "venue_run_environment": venue_run_rate - league_run_rate,
                    "venue_history_log_games": math.log1p(venue.games),
                    "schedule_consecutive_days_advantage": (
                        away_context["consecutive_days"] - home_context["consecutive_days"]
                    ),
                    "schedule_no_rest_travel_advantage": (
                        away_context["no_rest_travel"] - home_context["no_rest_travel"]
                    ),
                    "schedule_night_to_day_advantage": (
                        away_context["night_to_day"] - home_context["night_to_day"]
                    ),
                    "away_road_trip_game_number": away_context["road_trip_game_number"],
                }
            )
            output.append(materialized)

        date_games = completed_by_date.get(official_date, [])
        team_observations: dict[int, dict[str, Any]] = {}
        for game in date_games:
            venue_id = int(game["venue_id"]) if game.get("venue_id") is not None else None
            home_score = float(game["home_score"])
            away_score = float(game["away_score"])
            home_win = int(home_score > away_score)
            league_games += 1
            league_total_runs += home_score + away_score
            league_home_wins += home_win
            if venue_id is not None:
                venue = venue_states[venue_id]
                venue.games += 1
                venue.total_runs += home_score + away_score
                venue.home_wins += home_win
            day_night = str(game.get("day_night") or "unknown")
            for side, is_away in (("home", False), ("away", True)):
                team_id = int(game[f"{side}_team_id"])
                observation = team_observations.setdefault(
                    team_id,
                    {"venue_id": venue_id, "day_night": day_night, "is_away": is_away},
                )
                if day_night == "night":
                    observation["day_night"] = "night"

        for team_id, observation in team_observations.items():
            state = team_states[team_id]
            days_since = (current_date - state.last_date).days if state.last_date else None
            state.consecutive_days = state.consecutive_days + 1 if days_since == 1 else 1
            is_away = bool(observation["is_away"])
            state.away_streak = (
                state.away_streak + 1
                if is_away and state.last_was_away and days_since is not None and days_since <= 3
                else 1 if is_away else 0
            )
            state.last_date = current_date
            state.last_venue_id = observation["venue_id"]
            state.last_day_night = str(observation["day_night"])
            state.last_was_away = is_away
    return output


def add_schedule_context_interactions(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        fatigue = (
            float(row["schedule_consecutive_days_advantage"])
            + float(row["schedule_no_rest_travel_advantage"])
            + float(row["schedule_night_to_day_advantage"])
        )
        row.update(
            {
                "venue_lineup_ops_interaction": (
                    float(row["venue_run_environment"]) * float(row["lineup_ops_advantage"])
                ),
                "venue_starter_kbb_interaction": (
                    float(row["venue_run_environment"])
                    * float(row["starter_k_minus_bb_rate_difference"])
                ),
                "schedule_fatigue_elo_interaction": (
                    fatigue * float(row["home_elo_minus_away"])
                ),
            }
        )
        augmented.append(row)
    return augmented


def add_neutral_schedule_context_features(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    names = (*SCHEDULE_CONTEXT_FEATURE_NAMES, *SCHEDULE_CONTEXT_INTERACTION_NAMES)
    return [dict(row, **{name: 0.0 for name in names}) for row in rows]


def ensure_schedule_context_feature_values(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fill absent schedule fields without overwriting a real candidate block."""

    names = (*SCHEDULE_CONTEXT_FEATURE_NAMES, *SCHEDULE_CONTEXT_INTERACTION_NAMES)
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for name in names:
            row.setdefault(name, 0.0)
        output.append(row)
    return output


def select_schedule_context_feature_block(
    rows: Iterable[Mapping[str, Any]],
    included_names: Iterable[str],
) -> list[dict[str, Any]]:
    included = set(included_names)
    allowed = set(SCHEDULE_CONTEXT_FEATURE_NAMES)
    if not included <= allowed:
        raise ValueError("included_names contains an unknown schedule feature.")
    output: list[dict[str, Any]] = []
    for source in ensure_schedule_context_feature_values(rows):
        row = dict(source)
        for name in allowed - included:
            row[name] = 0.0
        for name in SCHEDULE_CONTEXT_INTERACTION_NAMES:
            row[name] = 0.0
        output.append(row)
    return output
