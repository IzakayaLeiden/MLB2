from __future__ import annotations

from datetime import date
from typing import Any, Iterable

from .models import SkippedGame


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def normalize_schedule_payloads(
    payloads: Iterable[dict[str, Any]],
    *,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> tuple[list[dict[str, Any]], list[SkippedGame]]:
    lower_bound = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    upper_bound = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    if lower_bound and upper_bound and lower_bound > upper_bound:
        raise ValueError("start_date는 end_date보다 늦을 수 없습니다.")
    rows: list[dict[str, Any]] = []
    skipped: list[SkippedGame] = []
    for payload in payloads:
        for date_bucket in payload.get("dates", []):
            if not isinstance(date_bucket, dict):
                continue
            for game in date_bucket.get("games", []):
                if not isinstance(game, dict):
                    continue
                row, skip = _normalize_game(game, lower_bound=lower_bound, upper_bound=upper_bound)
                if row is not None:
                    rows.append(row)
                elif skip is not None:
                    skipped.append(skip)
    return rows, skipped


def _normalize_game(
    game: dict[str, Any],
    *,
    lower_bound: date | None = None,
    upper_bound: date | None = None,
) -> tuple[dict[str, Any] | None, SkippedGame | None]:
    raw_game_id = game.get("gamePk")
    game_id = int(raw_game_id) if isinstance(raw_game_id, (int, str)) and str(raw_game_id).isdigit() else None
    official_date = game.get("officialDate")
    official_date = str(official_date) if official_date is not None else None

    resume_markers = ("resumedFrom", "resumedFromDate", "resumeDate", "resumeGameDate")
    present_resume_markers = [key for key in resume_markers if game.get(key)]
    if present_resume_markers:
        return None, SkippedGame(
            game_id,
            official_date,
            "resumed_game_temporal_ambiguity",
            f"재개 경기 필드: {present_resume_markers}",
        )

    if official_date is not None and (lower_bound is not None or upper_bound is not None):
        try:
            parsed_official_date = date.fromisoformat(official_date)
        except ValueError:
            return None, SkippedGame(game_id, official_date, "invalid_official_date", "officialDate가 ISO 날짜가 아닙니다.")
        if (lower_bound is not None and parsed_official_date < lower_bound) or (
            upper_bound is not None and parsed_official_date > upper_bound
        ):
            return None, SkippedGame(
                game_id,
                official_date,
                "official_date_out_of_range",
                f"요청 범위: {lower_bound or '-'}~{upper_bound or '-'}",
            )

    status = game.get("status") if isinstance(game.get("status"), dict) else {}
    abstract_state = status.get("abstractGameState")
    if abstract_state != "Final":
        return None, SkippedGame(game_id, official_date, "status_not_final", str(status.get("detailedState") or abstract_state))

    away = _nested(game, "teams", "away", default={})
    home = _nested(game, "teams", "home", default={})
    if not isinstance(away, dict) or not isinstance(home, dict):
        return None, SkippedGame(game_id, official_date, "missing_team_container", "home/away 팀 컨테이너가 없습니다.")

    away_score = away.get("score")
    home_score = home.get("score")
    if not isinstance(away_score, int) or not isinstance(home_score, int):
        return None, SkippedGame(game_id, official_date, "missing_score", "완료 경기의 정수 점수가 없습니다.")
    if bool(game.get("isTie")) or away_score == home_score:
        return None, SkippedGame(game_id, official_date, "tie_game", f"{away_score}-{home_score}")

    away_team_id = _nested(away, "team", "id")
    home_team_id = _nested(home, "team", "id")
    if not isinstance(away_team_id, int) or not isinstance(home_team_id, int):
        return None, SkippedGame(game_id, official_date, "missing_team_id", "정수형 팀 ID가 없습니다.")
    if game_id is None or official_date is None:
        return None, SkippedGame(game_id, official_date, "missing_primary_field", "gamePk 또는 officialDate가 없습니다.")

    away_pitcher = away.get("probablePitcher") if isinstance(away.get("probablePitcher"), dict) else {}
    home_pitcher = home.get("probablePitcher") if isinstance(home.get("probablePitcher"), dict) else {}

    row = {
        "game_id": game_id,
        "season": int(game.get("season") or official_date[:4]),
        "official_date": official_date,
        "game_start_utc": game.get("gameDate"),
        "game_type": game.get("gameType"),
        "status": abstract_state,
        "status_code": status.get("statusCode"),
        "detailed_status": status.get("detailedState"),
        "away_team_id": away_team_id,
        "away_team_name": _nested(away, "team", "name"),
        "away_team_abbreviation": _nested(away, "team", "abbreviation"),
        "home_team_id": home_team_id,
        "home_team_name": _nested(home, "team", "name"),
        "home_team_abbreviation": _nested(home, "team", "abbreviation"),
        "away_score": away_score,
        "home_score": home_score,
        "away_is_winner": bool(away.get("isWinner")),
        "home_is_winner": bool(home.get("isWinner")),
        "home_win": int(home_score > away_score),
        "away_probable_pitcher_id": away_pitcher.get("id"),
        "away_probable_pitcher_name": away_pitcher.get("fullName"),
        "home_probable_pitcher_id": home_pitcher.get("id"),
        "home_probable_pitcher_name": home_pitcher.get("fullName"),
        "venue_id": _nested(game, "venue", "id"),
        "venue_name": _nested(game, "venue", "name"),
        "day_night": game.get("dayNight"),
        "double_header": game.get("doubleHeader"),
        "game_number": game.get("gameNumber"),
        "scheduled_innings": game.get("scheduledInnings"),
        "source_game_link": game.get("link"),
    }
    return row, None
