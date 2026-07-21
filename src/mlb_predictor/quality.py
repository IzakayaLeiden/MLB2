from __future__ import annotations

import math
from collections import Counter
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Iterable

from .models import DataQualityError, QualityIssue, QualityReport


RAW_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "official_date",
    "game_start_utc",
    "status",
    "away_team_id",
    "home_team_id",
    "away_score",
    "home_score",
    "home_win",
}

FEATURE_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "official_date",
    "away_team_id",
    "home_team_id",
    "home_elo_pregame",
    "away_elo_pregame",
    "elo_expected_home_win_probability",
    "home_games_before",
    "away_games_before",
    "home_recent_games_count",
    "away_recent_games_count",
    "home_history_through_date",
    "away_history_through_date",
    "feature_cutoff_policy",
    "feature_version",
    "home_win",
}

BLOCKED_FEATURE_COLUMNS = {
    "away_score",
    "home_score",
    "away_is_winner",
    "home_is_winner",
    "final_result",
    "is_tie",
    "winner",
}


def _materialize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        # pandas/NumPy의 NaN, NaT, NA를 코어 검증기가 직접 의존성 없이 처리합니다.
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _missing_columns(rows: list[dict[str, Any]], required: set[str]) -> set[str]:
    if not rows:
        return set(required)
    available = set().union(*(row.keys() for row in rows))
    return required - available


def _duplicate_ids(rows: list[dict[str, Any]]) -> list[int]:
    counts = Counter(row.get("game_id") for row in rows)
    return sorted(int(game_id) for game_id, count in counts.items() if _is_integer(game_id) and count > 1)


def validate_raw_games(rows: Iterable[dict[str, Any]]) -> QualityReport:
    games = _materialize(rows)
    report = QualityReport("normalized_games", len(games))
    if not games:
        report.issues.append(QualityIssue("high", "empty_dataset", "완료된 정규시즌 경기가 한 건도 없습니다."))
        return report

    missing = _missing_columns(games, RAW_REQUIRED_COLUMNS)
    if missing:
        report.issues.append(QualityIssue("critical", "missing_columns", f"필수 열 누락: {sorted(missing)}"))

    duplicate_ids = _duplicate_ids(games)
    if duplicate_ids:
        report.issues.append(QualityIssue("critical", "duplicate_game_id", f"중복 game_id: {duplicate_ids[:20]}"))

    pitcher_missing = 0
    team_ids: set[int] = set()
    dates: list[str] = []
    for row in games:
        raw_game_id = row.get("game_id")
        game_id = int(raw_game_id) if _is_integer(raw_game_id) else None
        if not _is_integer(raw_game_id) or int(raw_game_id) <= 0:
            report.issues.append(QualityIssue("critical", "invalid_game_id", "game_id는 양의 정수여야 합니다.", game_id))
        try:
            official_date = date.fromisoformat(str(row.get("official_date")))
            dates.append(official_date.isoformat())
        except ValueError:
            report.issues.append(QualityIssue("critical", "invalid_official_date", "official_date가 ISO 날짜가 아닙니다.", game_id))
            continue

        start = row.get("game_start_utc")
        try:
            parsed_start = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            if parsed_start.tzinfo is None:
                raise ValueError("timezone missing")
        except ValueError:
            report.issues.append(QualityIssue("high", "invalid_game_start", "game_start_utc가 시간대 포함 ISO 시각이 아닙니다.", game_id))

        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")
        if not _is_integer(home_id) or int(home_id) <= 0 or not _is_integer(away_id) or int(away_id) <= 0:
            report.issues.append(QualityIssue("critical", "invalid_team_id", "홈·원정 팀 ID는 양의 정수여야 합니다.", game_id))
        if home_id == away_id:
            report.issues.append(QualityIssue("critical", "same_team_matchup", "홈팀과 원정팀 ID가 같습니다.", game_id))
        if _is_integer(home_id):
            team_ids.add(int(home_id))
        if _is_integer(away_id):
            team_ids.add(int(away_id))

        home_score = row.get("home_score")
        away_score = row.get("away_score")
        target = row.get("home_win")
        if not _is_integer(home_score) or not _is_integer(away_score) or int(home_score) < 0 or int(away_score) < 0:
            report.issues.append(QualityIssue("critical", "invalid_score", "점수는 0 이상의 정수여야 합니다.", game_id))
        elif home_score == away_score:
            report.issues.append(QualityIssue("high", "tie_in_training_rows", "무승부가 학습 행에 포함됐습니다.", game_id))
        elif not _is_integer(target) or int(target) not in {0, 1} or int(target) != int(home_score > away_score):
            report.issues.append(QualityIssue("critical", "target_score_mismatch", "home_win과 최종 점수가 일치하지 않습니다.", game_id))

        if row.get("status") != "Final":
            report.issues.append(QualityIssue("high", "non_final_game", "완료되지 않은 경기가 학습 행에 포함됐습니다.", game_id))
        if _is_missing(row.get("home_probable_pitcher_id")) or _is_missing(row.get("away_probable_pitcher_id")):
            pitcher_missing += 1

    report.metrics = {
        "distinct_game_ids": len({row.get("game_id") for row in games}),
        "distinct_team_ids": len(team_ids),
        "min_official_date": min(dates) if dates else None,
        "max_official_date": max(dates) if dates else None,
        "probable_pitcher_missing_rows": pitcher_missing,
        "probable_pitcher_missing_rate": round(pitcher_missing / len(games), 6),
    }
    if pitcher_missing:
        report.issues.append(
            QualityIssue(
                "low",
                "probable_pitcher_missing",
                f"선발 예정 투수 ID가 한쪽 이상 없는 경기: {pitcher_missing}/{len(games)}. 현재 단계에서는 허용됩니다.",
            )
        )
    return report


def validate_feature_rows(rows: Iterable[dict[str, Any]], *, recent_window: int = 10) -> QualityReport:
    features = _materialize(rows)
    report = QualityReport("pregame_features", len(features))
    if not features:
        report.issues.append(QualityIssue("high", "empty_dataset", "피처 행이 한 건도 없습니다."))
        return report

    missing = _missing_columns(features, FEATURE_REQUIRED_COLUMNS)
    if missing:
        report.issues.append(QualityIssue("critical", "missing_columns", f"필수 열 누락: {sorted(missing)}"))

    available_columns = set().union(*(row.keys() for row in features))
    blocked_columns = sorted(column for column in available_columns if column.lower() in BLOCKED_FEATURE_COLUMNS)
    if blocked_columns:
        report.issues.append(QualityIssue("critical", "post_outcome_columns", f"결과 누수 위험 열: {blocked_columns}"))

    duplicate_ids = _duplicate_ids(features)
    if duplicate_ids:
        report.issues.append(QualityIssue("critical", "duplicate_game_id", f"중복 game_id: {duplicate_ids[:20]}"))

    probability_columns = [column for column in available_columns if "probability" in column]
    numeric_columns = {
        "home_elo_pregame",
        "away_elo_pregame",
        "home_elo_minus_away",
        "home_season_win_pct",
        "away_season_win_pct",
        "home_recent_win_pct",
        "away_recent_win_pct",
    }
    for row in features:
        raw_game_id = row.get("game_id")
        game_id = int(raw_game_id) if _is_integer(raw_game_id) else None
        if not _is_integer(raw_game_id) or int(raw_game_id) <= 0:
            report.issues.append(QualityIssue("critical", "invalid_game_id", "game_id는 양의 정수여야 합니다.", game_id))
        home_team_id = row.get("home_team_id")
        away_team_id = row.get("away_team_id")
        if not _is_integer(home_team_id) or int(home_team_id) <= 0 or not _is_integer(away_team_id) or int(away_team_id) <= 0:
            report.issues.append(QualityIssue("critical", "invalid_team_id", "홈·원정 팀 ID는 양의 정수여야 합니다.", game_id))
        elif int(home_team_id) == int(away_team_id):
            report.issues.append(QualityIssue("critical", "same_team_matchup", "홈팀과 원정팀 ID가 같습니다.", game_id))
        try:
            current_date = date.fromisoformat(str(row.get("official_date")))
        except ValueError:
            report.issues.append(QualityIssue("critical", "invalid_official_date", "official_date가 ISO 날짜가 아닙니다.", game_id))
            continue

        if row.get("feature_cutoff_policy") != "prior_official_date_only":
            report.issues.append(QualityIssue("high", "unknown_cutoff_policy", "지원되지 않는 피처 시점 정책입니다.", game_id))
        for side in ("home", "away"):
            history_value = row.get(f"{side}_history_through_date")
            if not _is_missing(history_value):
                try:
                    history_date = date.fromisoformat(str(history_value))
                    if history_date >= current_date:
                        report.issues.append(
                            QualityIssue("critical", "lookahead_history_date", f"{side} 이력 날짜가 경기 날짜보다 이르지 않습니다.", game_id)
                        )
                except ValueError:
                    report.issues.append(QualityIssue("high", "invalid_history_date", f"{side} 이력 날짜 형식이 잘못됐습니다.", game_id))

            games_before = row.get(f"{side}_games_before")
            recent_count = row.get(f"{side}_recent_games_count")
            if not _is_integer(games_before) or int(games_before) < 0:
                report.issues.append(QualityIssue("critical", "invalid_games_before", f"{side}_games_before가 음이 아닌 정수가 아닙니다.", game_id))
            if not _is_integer(recent_count) or int(recent_count) < 0 or int(recent_count) > recent_window:
                report.issues.append(QualityIssue("high", "invalid_recent_count", f"{side}_recent_games_count가 0~{recent_window} 범위를 벗어났습니다.", game_id))
            if _is_integer(games_before) and _is_integer(recent_count) and int(recent_count) > int(games_before):
                report.issues.append(QualityIssue("critical", "recent_count_exceeds_history", f"{side} 최근 경기 수가 전체 과거 경기 수보다 큽니다.", game_id))

        for column in probability_columns:
            value = row.get(column)
            if not _is_real(value) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                report.issues.append(QualityIssue("critical", "invalid_probability", f"{column}이 0~1 유한 범위가 아닙니다.", game_id))
        for column in numeric_columns & available_columns:
            value = row.get(column)
            if not _is_real(value) or not math.isfinite(float(value)):
                report.issues.append(QualityIssue("critical", "invalid_numeric_feature", f"{column}이 유한 숫자가 아닙니다.", game_id))
        if not _is_integer(row.get("home_win")) or int(row["home_win"]) not in {0, 1}:
            report.issues.append(QualityIssue("critical", "invalid_target", "home_win이 0 또는 1이 아닙니다.", game_id))

    report.metrics = {
        "distinct_game_ids": len({row.get("game_id") for row in features}),
        "feature_column_count": len(available_columns),
        "probability_columns": sorted(probability_columns),
        "cutoff_policies": sorted({str(row.get("feature_cutoff_policy")) for row in features}),
        "feature_versions": sorted({str(row.get("feature_version")) for row in features}),
    }
    return report


def validate_dataset_pair(
    games: Iterable[dict[str, Any]],
    features: Iterable[dict[str, Any]],
    *,
    recent_window: int = 10,
) -> QualityReport:
    raw_rows = _materialize(games)
    feature_rows = _materialize(features)
    report = QualityReport("dataset_pair", len(feature_rows))
    raw_ids = {row.get("game_id") for row in raw_rows}
    feature_ids = {row.get("game_id") for row in feature_rows}
    if raw_ids != feature_ids:
        missing_features = sorted(raw_ids - feature_ids)
        unknown_features = sorted(feature_ids - raw_ids)
        report.issues.append(
            QualityIssue(
                "critical",
                "game_id_coverage_mismatch",
                f"피처 누락 ID={missing_features[:20]}, 원본에 없는 피처 ID={unknown_features[:20]}",
            )
        )
    if len(raw_rows) != len(feature_rows):
        report.issues.append(
            QualityIssue("critical", "row_count_mismatch", f"원본 {len(raw_rows)}행, 피처 {len(feature_rows)}행")
        )
    raw_by_id = {row.get("game_id"): row for row in raw_rows}
    feature_by_id = {row.get("game_id"): row for row in feature_rows}
    identity_columns = (
        "season",
        "official_date",
        "game_start_utc",
        "away_team_id",
        "home_team_id",
        "home_win",
    )
    mismatch_count = 0
    for game_id in sorted(raw_ids & feature_ids):
        raw_row = raw_by_id[game_id]
        feature_row = feature_by_id[game_id]
        mismatched = [column for column in identity_columns if raw_row.get(column) != feature_row.get(column)]
        if mismatched:
            mismatch_count += 1
            if mismatch_count <= 50:
                report.issues.append(
                    QualityIssue(
                        "critical",
                        "raw_feature_value_mismatch",
                        f"원본과 피처 값 불일치 열: {mismatched}",
                        int(game_id) if _is_integer(game_id) else None,
                    )
                )
    report.metrics = {
        "raw_row_count": len(raw_rows),
        "feature_row_count": len(feature_rows),
        "game_id_coverage_rate": round(len(raw_ids & feature_ids) / len(raw_ids), 6) if raw_ids else 0.0,
        "recent_window": recent_window,
        "raw_feature_mismatch_rows": mismatch_count,
    }
    return report


def raise_for_failed_reports(*reports: QualityReport) -> None:
    failed = [report for report in reports if not report.passed]
    if failed:
        summary = "; ".join(f"{report.dataset}: {len(report.issues)}개 이슈" for report in failed)
        raise DataQualityError(f"데이터 품질 게이트 실패: {summary}")
