from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from itertools import groupby
from typing import Any, Iterable

from .quality import raise_for_failed_reports, validate_raw_games


@dataclass(frozen=True)
class FeatureConfig:
    recent_window: int = 10
    initial_elo: float = 1500.0
    elo_k_factor: float = 20.0
    home_field_elo_advantage: float = 35.0
    offseason_elo_carry: float = 0.75
    neutral_runs_per_game: float = 4.5
    neutral_rest_days: int = 5
    feature_version: str = "pregame-v1"

    def __post_init__(self) -> None:
        if self.recent_window < 1:
            raise ValueError("recent_window은 1 이상이어야 합니다.")
        if self.elo_k_factor <= 0:
            raise ValueError("elo_k_factor는 양수여야 합니다.")
        if not 0 <= self.offseason_elo_carry <= 1:
            raise ValueError("offseason_elo_carry는 0~1 범위여야 합니다.")


@dataclass
class _TeamState:
    recent_window: int
    season_games: int = 0
    season_wins: int = 0
    last_game_date: date | None = None
    recent_wins: deque[int] = field(init=False)
    recent_runs_for: deque[int] = field(init=False)
    recent_runs_against: deque[int] = field(init=False)

    def __post_init__(self) -> None:
        self.recent_wins = deque(maxlen=self.recent_window)
        self.recent_runs_for = deque(maxlen=self.recent_window)
        self.recent_runs_against = deque(maxlen=self.recent_window)

    def snapshot(self, current_date: date, config: FeatureConfig) -> dict[str, Any]:
        recent_count = len(self.recent_wins)
        if self.last_game_date is None:
            rest_days = config.neutral_rest_days
        else:
            rest_days = max((current_date - self.last_game_date).days - 1, 0)
        return {
            "games_before": self.season_games,
            "season_win_pct": self.season_wins / self.season_games if self.season_games else 0.5,
            "recent_games_count": recent_count,
            "recent_win_pct": sum(self.recent_wins) / recent_count if recent_count else 0.5,
            "recent_runs_scored": sum(self.recent_runs_for) / recent_count if recent_count else config.neutral_runs_per_game,
            "recent_runs_allowed": sum(self.recent_runs_against) / recent_count if recent_count else config.neutral_runs_per_game,
            "rest_days": rest_days,
            "has_prior_history": int(self.season_games > 0),
            "history_through_date": self.last_game_date.isoformat() if self.last_game_date else None,
        }

    def update(self, *, won: int, runs_for: int, runs_against: int, game_date: date) -> None:
        self.season_games += 1
        self.season_wins += won
        self.recent_wins.append(won)
        self.recent_runs_for.append(runs_for)
        self.recent_runs_against.append(runs_against)
        self.last_game_date = game_date


def _elo_expected(home_elo: float, away_elo: float, home_advantage: float) -> float:
    exponent = -((home_elo + home_advantage) - away_elo) / 400.0
    return 1.0 / (1.0 + 10.0**exponent)


def _ordered_games(games: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(game) for game in games]
    return sorted(rows, key=lambda row: (str(row["official_date"]), str(row.get("game_start_utc") or ""), int(row["game_id"])))


def build_pregame_features(
    games: Iterable[dict[str, Any]],
    config: FeatureConfig | None = None,
) -> list[dict[str, Any]]:
    """완료 경기에서 공식 날짜 이전 정보만 사용한 피처를 생성합니다."""

    config = config or FeatureConfig()
    materialized = [dict(game) for game in games]
    raise_for_failed_reports(validate_raw_games(materialized))
    ordered = _ordered_games(materialized)
    elo: defaultdict[int, float] = defaultdict(lambda: config.initial_elo)
    states: dict[int, _TeamState] = {}
    current_season: int | None = None
    features: list[dict[str, Any]] = []

    for (season, official_date), daily_iter in groupby(
        ordered,
        key=lambda row: (int(row["season"]), str(row["official_date"])),
    ):
        daily_games = list(daily_iter)
        game_date = date.fromisoformat(official_date)

        if current_season != season:
            if current_season is not None:
                for team_id in list(elo.keys()):
                    elo[team_id] = config.initial_elo + (elo[team_id] - config.initial_elo) * config.offseason_elo_carry
            states = {}
            current_season = season

        def state_for(team_id: int) -> _TeamState:
            if team_id not in states:
                states[team_id] = _TeamState(config.recent_window)
            return states[team_id]

        daily_elo_deltas: defaultdict[int, float] = defaultdict(float)

        for game in daily_games:
            home_id = int(game["home_team_id"])
            away_id = int(game["away_team_id"])
            home_state = state_for(home_id).snapshot(game_date, config)
            away_state = state_for(away_id).snapshot(game_date, config)
            home_elo = float(elo[home_id])
            away_elo = float(elo[away_id])
            expected_home = _elo_expected(home_elo, away_elo, config.home_field_elo_advantage)

            feature_row = {
                "game_id": int(game["game_id"]),
                "season": season,
                "official_date": official_date,
                "game_start_utc": game.get("game_start_utc"),
                "away_team_id": away_id,
                "away_team_name": game.get("away_team_name"),
                "home_team_id": home_id,
                "home_team_name": game.get("home_team_name"),
                "away_probable_pitcher_id": game.get("away_probable_pitcher_id"),
                "away_probable_pitcher_name": game.get("away_probable_pitcher_name"),
                "home_probable_pitcher_id": game.get("home_probable_pitcher_id"),
                "home_probable_pitcher_name": game.get("home_probable_pitcher_name"),
                "venue_id": game.get("venue_id"),
                "day_night": game.get("day_night"),
                "home_elo_pregame": round(home_elo, 6),
                "away_elo_pregame": round(away_elo, 6),
                "home_elo_minus_away": round(home_elo - away_elo, 6),
                "elo_expected_home_win_probability": round(expected_home, 9),
                "home_games_before": home_state["games_before"],
                "away_games_before": away_state["games_before"],
                "home_season_win_pct": round(home_state["season_win_pct"], 9),
                "away_season_win_pct": round(away_state["season_win_pct"], 9),
                "season_win_pct_difference": round(home_state["season_win_pct"] - away_state["season_win_pct"], 9),
                "home_recent_games_count": home_state["recent_games_count"],
                "away_recent_games_count": away_state["recent_games_count"],
                "home_recent_win_pct": round(home_state["recent_win_pct"], 9),
                "away_recent_win_pct": round(away_state["recent_win_pct"], 9),
                "recent_win_pct_difference": round(home_state["recent_win_pct"] - away_state["recent_win_pct"], 9),
                "home_recent_runs_scored": round(home_state["recent_runs_scored"], 6),
                "away_recent_runs_scored": round(away_state["recent_runs_scored"], 6),
                "home_recent_runs_allowed": round(home_state["recent_runs_allowed"], 6),
                "away_recent_runs_allowed": round(away_state["recent_runs_allowed"], 6),
                "recent_run_margin_difference": round(
                    (home_state["recent_runs_scored"] - home_state["recent_runs_allowed"])
                    - (away_state["recent_runs_scored"] - away_state["recent_runs_allowed"]),
                    6,
                ),
                "home_rest_days": home_state["rest_days"],
                "away_rest_days": away_state["rest_days"],
                "rest_days_difference": home_state["rest_days"] - away_state["rest_days"],
                "home_has_prior_history": home_state["has_prior_history"],
                "away_has_prior_history": away_state["has_prior_history"],
                "home_history_through_date": home_state["history_through_date"],
                "away_history_through_date": away_state["history_through_date"],
                "feature_cutoff_policy": "prior_official_date_only",
                "feature_version": config.feature_version,
                "home_win": int(game["home_win"]),
            }
            features.append(feature_row)

            actual_home = int(game["home_win"])
            elo_delta = config.elo_k_factor * (actual_home - expected_home)
            daily_elo_deltas[home_id] += elo_delta
            daily_elo_deltas[away_id] -= elo_delta

        # 같은 날짜 결과는 그날의 다른 경기 피처에 보이지 않도록 모든 피처 계산 후 반영합니다.
        for team_id, delta in daily_elo_deltas.items():
            elo[team_id] += delta

        for game in daily_games:
            home_id = int(game["home_team_id"])
            away_id = int(game["away_team_id"])
            home_win = int(game["home_win"])
            home_score = int(game["home_score"])
            away_score = int(game["away_score"])
            state_for(home_id).update(won=home_win, runs_for=home_score, runs_against=away_score, game_date=game_date)
            state_for(away_id).update(won=1 - home_win, runs_for=away_score, runs_against=home_score, game_date=game_date)

    for row in features:
        probability = float(row["elo_expected_home_win_probability"])
        if not math.isfinite(probability):
            raise ValueError(f"비유한 Elo 확률이 생성되었습니다: game_id={row['game_id']}")
    return features
