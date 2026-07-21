from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping


LEAGUE_RUNS_PER_TEAM = 4.5
RUN_STRENGTH_FEATURE_NAMES = (
    "run_strength_offense_difference",
    "run_strength_defense_advantage",
    "run_strength_expected_margin",
    "run_strength_history_difference",
)


@dataclass
class _TeamRunState:
    offense: float = LEAGUE_RUNS_PER_TEAM
    defense_allowed: float = LEAGUE_RUNS_PER_TEAM
    games: int = 0
    season: int | None = None

    def enter_season(self, season: int, retention: float) -> None:
        if self.season == season:
            return
        if self.season is not None:
            self.offense = LEAGUE_RUNS_PER_TEAM + retention * (self.offense - LEAGUE_RUNS_PER_TEAM)
            self.defense_allowed = LEAGUE_RUNS_PER_TEAM + retention * (
                self.defense_allowed - LEAGUE_RUNS_PER_TEAM
            )
            self.games = int(round(self.games * retention))
        self.season = season


def add_dynamic_run_strength_features(
    feature_rows: Iterable[Mapping[str, Any]],
    completed_games: Iterable[Mapping[str, Any]],
    *,
    half_life_games: float = 20.0,
    offseason_retention: float = 0.75,
) -> list[dict[str, Any]]:
    """Build atomic-date offense and defense ratings from past final scores."""

    if half_life_games <= 0:
        raise ValueError("half_life_games must be positive.")
    if not 0.0 <= offseason_retention <= 1.0:
        raise ValueError("offseason_retention must be between zero and one.")
    rows = sorted(
        (dict(row) for row in feature_rows),
        key=lambda row: (str(row["official_date"]), int(row["game_id"])),
    )
    completed_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_game in completed_games:
        game = dict(source_game)
        if game.get("home_score") is None or game.get("away_score") is None:
            continue
        completed_by_date[str(game["official_date"])].append(game)

    rows_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_date[str(row["official_date"])].append(row)
    states: dict[int, _TeamRunState] = defaultdict(_TeamRunState)
    alpha = 1.0 - math.pow(0.5, 1.0 / half_life_games)
    output: list[dict[str, Any]] = []
    all_dates = sorted(set(rows_by_date) | set(completed_by_date))
    for official_date in all_dates:
        date_rows = rows_by_date.get(official_date, [])
        date_games = completed_by_date.get(official_date, [])
        if date_rows:
            season = int(date_rows[0].get("season") or official_date[:4])
        elif date_games:
            season = int(date_games[0].get("season") or official_date[:4])
        else:  # pragma: no cover - all_dates comes from the two non-empty mappings.
            continue
        for row in date_rows:
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            states[home_id].enter_season(season, offseason_retention)
            states[away_id].enter_season(season, offseason_retention)
            home = states[home_id]
            away = states[away_id]
            offense_difference = home.offense - away.offense
            defense_advantage = away.defense_allowed - home.defense_allowed
            materialized = dict(row)
            materialized.update(
                {
                    "run_strength_provenance": "completed_scores_strictly_before_official_date_v1",
                    "run_strength_half_life_games": half_life_games,
                    "run_strength_offseason_retention": offseason_retention,
                    "run_strength_offense_difference": offense_difference,
                    "run_strength_defense_advantage": defense_advantage,
                    "run_strength_expected_margin": offense_difference + defense_advantage,
                    "run_strength_history_difference": float(home.games - away.games),
                }
            )
            output.append(materialized)

        observations: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"offense": [], "defense": []}
        )
        for game in date_games:
            home_id = int(game["home_team_id"])
            away_id = int(game["away_team_id"])
            home_runs = float(game["home_score"])
            away_runs = float(game["away_score"])
            observations[home_id]["offense"].append(home_runs)
            observations[home_id]["defense"].append(away_runs)
            observations[away_id]["offense"].append(away_runs)
            observations[away_id]["defense"].append(home_runs)
        for team_id, values in observations.items():
            state = states[team_id]
            state.enter_season(season, offseason_retention)
            count = len(values["offense"])
            effective_alpha = 1.0 - math.pow(1.0 - alpha, count)
            state.offense = (1.0 - effective_alpha) * state.offense + effective_alpha * (
                sum(values["offense"]) / count
            )
            state.defense_allowed = (
                (1.0 - effective_alpha) * state.defense_allowed
                + effective_alpha * (sum(values["defense"]) / count)
            )
            state.games += count
    return output


def add_neutral_run_strength_features(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row, **{name: 0.0 for name in RUN_STRENGTH_FEATURE_NAMES}) for row in rows]
