from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest


def make_game(
    game_id: int,
    official_date: str,
    home_team_id: int,
    away_team_id: int,
    home_score: int,
    away_score: int,
    *,
    start_hour: int = 18,
    season: int = 2025,
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "season": season,
        "official_date": official_date,
        "game_start_utc": f"{official_date}T{start_hour:02d}:00:00Z",
        "game_type": "R",
        "status": "Final",
        "status_code": "F",
        "detailed_status": "Final",
        "away_team_id": away_team_id,
        "away_team_name": f"Team {away_team_id}",
        "away_team_abbreviation": f"T{away_team_id}",
        "home_team_id": home_team_id,
        "home_team_name": f"Team {home_team_id}",
        "home_team_abbreviation": f"T{home_team_id}",
        "away_score": away_score,
        "home_score": home_score,
        "away_is_winner": away_score > home_score,
        "home_is_winner": home_score > away_score,
        "home_win": int(home_score > away_score),
        "away_probable_pitcher_id": 10_000 + away_team_id,
        "away_probable_pitcher_name": f"Away Pitcher {away_team_id}",
        "home_probable_pitcher_id": 20_000 + home_team_id,
        "home_probable_pitcher_name": f"Home Pitcher {home_team_id}",
        "venue_id": 30_000 + home_team_id,
        "venue_name": f"Park {home_team_id}",
        "day_night": "night",
        "double_header": "N",
        "game_number": 1,
        "scheduled_innings": 9,
        "source_game_link": f"/api/v1.1/game/{game_id}/feed/live",
    }


@pytest.fixture
def game_factory():
    return make_game


@pytest.fixture
def schedule_payload() -> dict[str, Any]:
    def api_game(
        game_id: int,
        *,
        status: str,
        detailed_status: str,
        home_score: int | None,
        away_score: int | None,
        is_tie: bool = False,
    ) -> dict[str, Any]:
        game = {
            "gamePk": game_id,
            "link": f"/api/v1.1/game/{game_id}/feed/live",
            "gameType": "R",
            "season": "2025",
            "gameDate": "2025-04-01T22:40:00Z",
            "officialDate": "2025-04-01",
            "status": {
                "abstractGameState": status,
                "detailedState": detailed_status,
                "statusCode": "F" if status == "Final" else "S",
            },
            "teams": {
                "away": {
                    "team": {"id": 2, "name": "Away", "abbreviation": "AWY"},
                    "score": away_score,
                    "isWinner": away_score is not None and home_score is not None and away_score > home_score,
                    "probablePitcher": {"id": 22, "fullName": "Away Starter"},
                },
                "home": {
                    "team": {"id": 1, "name": "Home", "abbreviation": "HME"},
                    "score": home_score,
                    "isWinner": away_score is not None and home_score is not None and home_score > away_score,
                    "probablePitcher": {"id": 11, "fullName": "Home Starter"},
                },
            },
            "venue": {"id": 101, "name": "Test Park"},
            "isTie": is_tie,
            "dayNight": "night",
            "doubleHeader": "N",
            "gameNumber": 1,
            "scheduledInnings": 9,
        }
        return game

    return {
        "dates": [
            {
                "date": "2025-04-01",
                "games": [
                    api_game(100, status="Final", detailed_status="Final", home_score=5, away_score=3),
                    api_game(101, status="Preview", detailed_status="Scheduled", home_score=None, away_score=None),
                    api_game(102, status="Final", detailed_status="Final", home_score=2, away_score=2, is_tie=True),
                    api_game(103, status="Final", detailed_status="Completed Early", home_score=None, away_score=None),
                ],
            }
        ]
    }


@pytest.fixture
def cloned_payload(schedule_payload):
    return deepcopy(schedule_payload)

