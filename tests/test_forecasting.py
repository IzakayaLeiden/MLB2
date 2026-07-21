from __future__ import annotations

import hashlib
import json

import pytest

from mlb_predictor.forecasting import create_prediction_feed, grade_prediction_feed, load_frozen_model


def elo_model(path) -> None:
    payload = {
        "schema_version": "model-v1",
        "model_version": "model-v1",
        "created_at_utc": "2025-01-01T00:00:00Z",
        "frozen": True,
        "retraining_during_shadow": False,
        "model_type": "elo",
        "selected_candidate": "elo",
        "selection_fingerprint": "test",
        "holdout_evaluation": {"season": 2025, "passed": True, "evaluated_at_utc": "2026-01-01T00:00:00Z"},
        "training": {"rows": 100, "cutoff_date": "2026-07-20", "constant_home_win_rate": 0.54},
        "runtime_model": None,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["model_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def scheduled_from_game(game: dict[str, object]) -> dict[str, object]:
    row = dict(game)
    for key in ("home_score", "away_score", "home_win", "home_is_winner", "away_is_winner"):
        row.pop(key)
    row.update({"schema_version": "scheduled-game-v1", "status": "Preview", "schedule_state": "scheduled", "forecast_eligible": True})
    return row


def test_forecast_seals_result_free_feed_and_refuses_overwrite(tmp_path, game_factory) -> None:
    model_path = tmp_path / "model-v1.json"
    elo_model(model_path)
    history = [game_factory(1, "2025-04-01", 1, 2, 5, 3)]
    future = scheduled_from_game(game_factory(2, "2025-04-02", 2, 1, 1, 0, start_hour=23))

    feed, path = create_prediction_feed(
        completed_games=history,
        scheduled_games=[future],
        model_path=model_path,
        output_root=tmp_path / "feeds",
        target_date="2025-04-02",
        created_at_utc="2025-04-02T12:00:00Z",
    )

    assert path.exists()
    assert feed["schema_version"] == "prediction-feed-v1"
    assert feed["previous_file_sha256"] is None
    assert feed["quality"]["status"] == "passed"
    assert '"home_win"' not in json.dumps(feed)
    assert feed["predictions"][0]["home_team"]["probable_pitcher"]
    with pytest.raises(FileExistsError, match="이미 봉인"):
        create_prediction_feed(
            completed_games=history,
            scheduled_games=[future],
            model_path=model_path,
            output_root=tmp_path / "feeds",
            target_date="2025-04-02",
            created_at_utc="2025-04-02T12:01:00Z",
        )


def test_late_forecast_is_fail_closed(tmp_path, game_factory) -> None:
    model_path = tmp_path / "model-v1.json"
    elo_model(model_path)
    history = [game_factory(10, "2025-04-01", 1, 2, 5, 3)]
    future = scheduled_from_game(game_factory(11, "2025-04-02", 2, 1, 1, 0, start_hour=13))

    feed, _ = create_prediction_feed(
        completed_games=history,
        scheduled_games=[future],
        model_path=model_path,
        output_root=tmp_path / "feeds",
        target_date="2025-04-02",
        created_at_utc="2025-04-02T12:30:00Z",
    )

    assert feed["quality"]["status"] == "failed"
    assert feed["quality"]["late_game_ids"] == [11]
    assert feed["predictions"] == []


def test_grade_keeps_results_separate(tmp_path, game_factory) -> None:
    model_path = tmp_path / "model-v1.json"
    elo_model(model_path)
    history = [game_factory(20, "2025-04-01", 1, 2, 5, 3)]
    completed = game_factory(21, "2025-04-02", 2, 1, 4, 2, start_hour=23)
    feed, _ = create_prediction_feed(
        completed_games=history,
        scheduled_games=[scheduled_from_game(completed)],
        model_path=model_path,
        output_root=tmp_path / "feeds",
        target_date="2025-04-02",
        created_at_utc="2025-04-02T12:00:00Z",
    )

    grade = grade_prediction_feed(feed, [completed])

    assert grade["quality_status"] == "passed"
    assert grade["grades"][0]["home_win"] == 1
    assert "home_win" not in feed["predictions"][0]
    assert load_frozen_model(model_path)["model_type"] == "elo"
