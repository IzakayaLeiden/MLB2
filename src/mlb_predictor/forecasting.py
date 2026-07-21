from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from .features import FeatureConfig, build_forecast_features
from .io import sha256_file, write_json
from .modeling import ModelBundle


ET = ZoneInfo("America/New_York")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("시각에는 UTC 오프셋이 필요합니다.")
    return parsed.astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_model_hash(payload: Mapping[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("model_sha256", None)
    encoded = json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "model-v1" or payload.get("frozen") is not True:
        raise ValueError("동결된 model-v1 파일이 아닙니다.")
    expected = str(payload.get("model_sha256") or "")
    if not expected or _canonical_model_hash(payload) != expected:
        raise ValueError("model-v1 해시 검증에 실패했습니다.")
    return payload


def _probability(model: Mapping[str, Any], feature: Mapping[str, Any]) -> float:
    model_type = str(model["model_type"])
    if model_type == "elo":
        return float(feature["elo_expected_home_win_probability"])
    if model_type == "constant":
        return float(model["training"]["constant_home_win_rate"])
    runtime_payload = model.get("runtime_model")
    if not isinstance(runtime_payload, Mapping):
        raise ValueError("LR model-v1에 runtime_model이 없습니다.")
    bundle = ModelBundle.from_dict(runtime_payload)
    calibrated = model_type == "logistic_platt"
    return float(bundle.predict_rows([feature], calibrated=calibrated)[0])


def _existing_feeds(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("**/prediction-*.json") if path.is_file())


def _sealed_game_ids(paths: Iterable[Path]) -> set[int]:
    sealed: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for prediction in payload.get("predictions", []):
            sealed.add(int(prediction["game_id"]))
    return sealed


def create_prediction_feed(
    *,
    completed_games: Iterable[dict[str, Any]],
    scheduled_games: Iterable[dict[str, Any]],
    model_path: str | Path,
    output_root: str | Path,
    target_date: str,
    created_at_utc: str | None = None,
    feature_config: FeatureConfig | None = None,
) -> tuple[dict[str, Any], Path]:
    """당일 예측을 새 파일로 봉인하며 기존 경기 예측은 덮어쓰지 않습니다."""

    created = _parse_utc(created_at_utc) if created_at_utc else _now()
    if created.astimezone(ET).date().isoformat() != target_date:
        raise ValueError("target_date는 생성 시각의 America/New_York 날짜와 같아야 합니다.")
    model = load_frozen_model(model_path)
    history = [dict(game) for game in completed_games]
    all_scheduled = [dict(game) for game in scheduled_games if str(game.get("official_date")) == target_date]
    eligible_schedule = [game for game in all_scheduled if bool(game.get("forecast_eligible"))]
    features = build_forecast_features(history, eligible_schedule, feature_config)
    by_id = {int(game["game_id"]): game for game in eligible_schedule}
    root = Path(output_root)
    existing = _existing_feeds(root)
    already_sealed = _sealed_game_ids(existing)
    duplicate_ids = sorted(int(feature["game_id"]) for feature in features if int(feature["game_id"]) in already_sealed)
    if duplicate_ids:
        raise FileExistsError(f"이미 봉인된 경기 예측은 덮어쓸 수 없습니다: {duplicate_ids}")
    predictions: list[dict[str, Any]] = []
    late_game_ids: list[int] = []
    for feature in features:
        game_id = int(feature["game_id"])
        game = by_id[game_id]
        start = _parse_utc(str(game["game_start_utc"]))
        cutoff = start - timedelta(minutes=60)
        if created > cutoff:
            late_game_ids.append(game_id)
            continue
        predictions.append(
            {
                "game_id": game_id,
                "game_start_utc": game["game_start_utc"],
                "official_date": target_date,
                "away_team": {
                    "id": int(game["away_team_id"]),
                    "name": game.get("away_team_name"),
                    "abbreviation": game.get("away_team_abbreviation"),
                    "probable_pitcher": game.get("away_probable_pitcher_name"),
                },
                "home_team": {
                    "id": int(game["home_team_id"]),
                    "name": game.get("home_team_name"),
                    "abbreviation": game.get("home_team_abbreviation"),
                    "probable_pitcher": game.get("home_probable_pitcher_name"),
                },
                "venue": {"id": game.get("venue_id"), "name": game.get("venue_name")},
                "home_win_probability": round(_probability(model, feature), 9),
                "evaluation_eligible": True,
                "sealed_before_start_minutes": int((start - created).total_seconds() // 60),
                "diagnostics": {
                    "elo_home_win_probability": float(feature["elo_expected_home_win_probability"]),
                    "home_elo": float(feature["home_elo_pregame"]),
                    "away_elo": float(feature["away_elo_pregame"]),
                    "home_recent_win_pct": float(feature["home_recent_win_pct"]),
                    "away_recent_win_pct": float(feature["away_recent_win_pct"]),
                    "home_rest_days": int(feature["home_rest_days"]),
                    "away_rest_days": int(feature["away_rest_days"]),
                    "data_through_date": max(
                        str(feature.get("home_history_through_date") or ""),
                        str(feature.get("away_history_through_date") or ""),
                    ) or None,
                },
            }
        )
    previous_sha = sha256_file(existing[-1]) if existing else None
    quality_passed = not late_game_ids and len(predictions) == len(eligible_schedule)
    payload = {
        "schema_version": "prediction-feed-v1",
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "data_through_date": max((str(game["official_date"]) for game in history), default=None),
        "target_date_et": target_date,
        "timezone": "America/New_York",
        "model_version": model["model_version"],
        "model_sha256": model["model_sha256"],
        "quality": {
            "status": "passed" if quality_passed else "failed",
            "eligible_games": len(eligible_schedule),
            "sealed_predictions": len(predictions),
            "late_game_ids": late_game_ids,
        },
        "previous_file_sha256": previous_sha,
        "predictions": predictions,
    }
    destination = root / target_date / f"prediction-{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    if os.path.lexists(destination):
        raise FileExistsError(f"예측 파일이 이미 존재합니다: {destination}")
    write_json(destination, payload)
    return payload, destination


def grade_prediction_feed(
    feed: Mapping[str, Any],
    completed_games: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """봉인 예측과 완료 결과를 별도 결과 파일로 연결합니다."""

    if feed.get("schema_version") != "prediction-feed-v1":
        raise ValueError("prediction-feed-v1이 아닙니다.")
    completed = {int(game["game_id"]): dict(game) for game in completed_games}
    grades: list[dict[str, Any]] = []
    missing: list[int] = []
    for prediction in feed.get("predictions", []):
        game_id = int(prediction["game_id"])
        game = completed.get(game_id)
        if game is None:
            missing.append(game_id)
            continue
        grades.append(
            {
                "game_id": game_id,
                "official_date": prediction["official_date"],
                "home_win_probability": float(prediction["home_win_probability"]),
                "elo_home_win_probability": float(prediction["diagnostics"]["elo_home_win_probability"]),
                "home_win": int(game["home_win"]),
                "evaluation_eligible": bool(prediction.get("evaluation_eligible")),
            }
        )
    return {
        "schema_version": "prediction-grade-v1",
        "created_at_utc": _now().isoformat().replace("+00:00", "Z"),
        "feed_created_at_utc": feed["created_at_utc"],
        "target_date_et": feed["target_date_et"],
        "model_version": feed["model_version"],
        "quality_status": "passed" if not missing else "pending",
        "graded_games": len(grades),
        "missing_game_ids": missing,
        "grades": grades,
    }


__all__ = ["create_prediction_feed", "grade_prediction_feed", "load_frozen_model"]
