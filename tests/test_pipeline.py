from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from mlb_predictor.collector import MlbStatsApiClient
from mlb_predictor.io import read_rows
from mlb_predictor.pipeline import build_dataset
from mlb_predictor.models import DataQualityError
from mlb_predictor.quality import validate_dataset_pair, validate_feature_rows, validate_raw_games


def test_end_to_end_pipeline_writes_reproducible_artifacts(tmp_path, schedule_payload, monkeypatch) -> None:
    monkeypatch.setattr(MlbStatsApiClient, "_request_json", lambda self, url: schedule_payload)
    output_dir = tmp_path / "dataset"

    manifest = build_dataset(
        start_date="2025-04-01",
        end_date="2025-04-01",
        output_dir=output_dir,
        history_start_date="2025-04-01",
        chunk_days=1,
    )

    assert manifest["quality_gate_passed"] is True
    assert manifest["build_status"] == "completed"
    assert manifest["artifacts_valid"] is True
    assert manifest["counts"] == {
        "history_games": 1,
        "warmup_games": 0,
        "normalized_games": 1,
        "pregame_feature_rows": 1,
        "skipped_games": 3,
    }
    assert (output_dir / "raw" / "schedule_2025-04-01_2025-04-01.json").exists()
    assert (output_dir / "processed" / "games.parquet").exists()
    assert (output_dir / "processed" / "history_games.parquet").exists()
    assert (output_dir / "features" / "pregame_features.parquet").exists()
    assert (output_dir / "reports" / "quality.json").exists()
    assert (output_dir / "manifest.json").exists()

    features = pd.read_parquet(output_dir / "features" / "pregame_features.parquet")
    assert list(features["game_id"]) == [100]
    assert "home_score" not in features.columns
    assert "away_score" not in features.columns
    assert int(features.loc[0, "home_win"]) == 1

    quality = json.loads((output_dir / "reports" / "quality.json").read_text(encoding="utf-8"))
    assert all(report["passed"] for report in quality["reports"])

    reloaded_games = read_rows(output_dir / "processed" / "games.parquet")
    reloaded_features = read_rows(output_dir / "features" / "pregame_features.parquet")
    assert validate_raw_games(reloaded_games).passed
    assert validate_feature_rows(reloaded_features).passed
    assert validate_dataset_pair(reloaded_games, reloaded_features).passed


def test_pipeline_uses_history_before_requested_output_start(tmp_path, schedule_payload, monkeypatch) -> None:
    payload = deepcopy(schedule_payload)
    prior = deepcopy(payload["dates"][0]["games"][0])
    prior["gamePk"] = 99
    prior["officialDate"] = "2025-03-31"
    prior["gameDate"] = "2025-03-31T22:40:00Z"
    payload["dates"].insert(0, {"date": "2025-03-31", "games": [prior]})
    monkeypatch.setattr(MlbStatsApiClient, "_request_json", lambda self, url: payload)

    output_dir = tmp_path / "warmup"
    manifest = build_dataset(
        start_date="2025-04-01",
        end_date="2025-04-01",
        history_start_date="2025-03-31",
        output_dir=output_dir,
        chunk_days=2,
    )

    assert manifest["counts"]["history_games"] == 2
    assert manifest["counts"]["warmup_games"] == 1
    assert manifest["counts"]["normalized_games"] == 1
    features = read_rows(output_dir / "features" / "pregame_features.parquet")
    assert features[0]["game_id"] == 100
    assert features[0]["home_games_before"] == 1
    assert features[0]["away_games_before"] == 1


def test_failed_rebuild_quarantines_previous_success_and_writes_truthful_manifest(
    tmp_path, schedule_payload, monkeypatch
) -> None:
    output_dir = tmp_path / "same-output"
    monkeypatch.setattr(MlbStatsApiClient, "_request_json", lambda self, url: schedule_payload)
    build_dataset(
        start_date="2025-04-01",
        end_date="2025-04-01",
        history_start_date="2025-04-01",
        output_dir=output_dir,
        chunk_days=1,
    )
    assert (output_dir / "processed" / "games.parquet").exists()

    monkeypatch.setattr(MlbStatsApiClient, "_request_json", lambda self, url: {"dates": []})
    with pytest.raises(DataQualityError):
        build_dataset(
            start_date="2025-04-01",
            end_date="2025-04-01",
            history_start_date="2025-04-01",
            output_dir=output_dir,
            chunk_days=1,
            refresh=True,
        )

    failed_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert failed_manifest["build_status"] == "failed"
    assert failed_manifest["quality_gate_passed"] is False
    assert failed_manifest["artifacts_valid"] is False
    assert not (output_dir / "processed" / "games.parquet").exists()
    assert list((output_dir / "previous_runs").rglob("games.parquet"))
