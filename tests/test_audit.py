from __future__ import annotations

import json
from pathlib import Path

from mlb_predictor.audit import (
    build_audit_report,
    generate_historical_audit_rows,
    paired_date_block_bootstrap,
    write_audit_bundle,
)
from mlb_predictor.io import write_rows_csv


def feature_row(game_id: int, season: int, positive: bool) -> dict[str, object]:
    x = 1.0 if positive else -1.0
    return {
        "game_id": game_id,
        "season": season,
        "official_date": f"{season}-06-{game_id % 20 + 1:02d}",
        "home_win": int(positive),
        "elo_expected_home_win_probability": 0.8 if positive else 0.2,
        "home_elo_minus_away": 80.0 * x,
        "season_win_pct_difference": 0.2 * x,
        "recent_win_pct_difference": 0.3 * x,
        "recent_run_margin_difference": 1.5 * x,
        "rest_days_difference": x,
        "home_games_before": 50,
        "away_games_before": 50,
        "home_recent_games_count": 10,
        "away_recent_games_count": 10,
        "home_has_prior_history": 1,
        "away_has_prior_history": 1,
        "day_night": "night",
    }


def multi_year_rows() -> list[dict[str, object]]:
    return [
        feature_row(season * 100 + index, season, index % 2 == 0)
        for season in range(2018, 2027)
        for index in range(20)
    ]


def test_generates_game_level_oof_probabilities_for_selection_and_holdout() -> None:
    rows = generate_historical_audit_rows(multi_year_rows(), l2=0.01)

    assert len(rows) == 80
    assert {row["season"] for row in rows} == {2022, 2023, 2024, 2025}
    assert {row["split"] for row in rows} == {"model_selection", "sealed_holdout"}
    assert all(0.0 <= float(row["p_logistic_platt"]) <= 1.0 for row in rows)
    assert all("p_platt_base_raw" in row for row in rows)


def test_paired_date_block_bootstrap_is_deterministic_and_paired() -> None:
    rows = [
        {
            "official_date": f"2025-04-{index + 1:02d}",
            "home_win": index % 2,
            "challenger": 0.9 if index % 2 else 0.1,
            "baseline": 0.5,
        }
        for index in range(20)
    ]

    first = paired_date_block_bootstrap(
        rows,
        challenger="challenger",
        baseline="baseline",
        metric="log_loss",
        iterations=500,
        seed=7,
    )
    second = paired_date_block_bootstrap(
        rows,
        challenger="challenger",
        baseline="baseline",
        metric="log_loss",
        iterations=500,
        seed=7,
    )

    assert first == second
    assert first["point_estimate"] < 0
    assert first["confidence_interval_95"][1] < 0
    assert first["date_blocks"] == 20


def test_report_separates_seasons_and_holdout() -> None:
    rows = generate_historical_audit_rows(multi_year_rows(), l2=0.01)
    report = build_audit_report(rows, iterations=100, seed=11, created_at_utc="2026-07-21T00:00:00Z")

    assert report["schema_version"] == "model-audit-v1"
    assert set(report["season_results"]) == {"2022", "2023", "2024", "2025"}
    assert report["combined_model_selection_2022_2024"]["rows"] == 60
    assert report["sealed_holdout_2025"]["rows"] == 20


def test_writes_hashed_audit_bundle(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    games = tmp_path / "games.parquet"
    exclusions = tmp_path / "exclusions.csv"
    model_path = tmp_path / "model.json"
    output = tmp_path / "audit"
    rows = multi_year_rows()
    write_rows_csv(features, rows)
    games.write_bytes(b"normalized games")
    exclusions.write_text("game_id,reason\n1,test\n", encoding="utf-8")
    model = {
        "model_version": "model-v1",
        "model_type": "logistic_platt",
        "model_sha256": "canonical-model-hash",
        "selection_fingerprint": "selection-hash",
        "training": {"l2": 0.01},
    }
    model_path.write_text(json.dumps(model), encoding="utf-8")

    manifest = write_audit_bundle(
        feature_rows=rows,
        features_path=features,
        games_path=games,
        exclusions_path=exclusions,
        model=model,
        model_path=model_path,
        output_dir=output,
        code_revision="abc123",
        iterations=100,
        seed=13,
    )

    assert manifest["code_revision"] == "abc123"
    assert {
        "selection-predictions.csv",
        "holdout-predictions.csv",
        "audit-report.json",
        "pregame-features.parquet",
        "normalized-games.parquet",
        "exclusions.csv",
        "model-v1.json",
        "environment-lock.txt",
    } == set(manifest["artifacts"])
    assert (output / "manifest.json").exists()
    assert (output / "holdout-predictions.csv").exists()
    assert (output / "sha256sums.txt").exists()


def test_paired_date_block_bootstrap_supports_accuracy_error_rate() -> None:
    rows = [
        {"official_date": "2025-04-01", "home_win": 1, "challenger": 0.6, "baseline": 0.4},
        {"official_date": "2025-04-02", "home_win": 0, "challenger": 0.4, "baseline": 0.6},
    ]
    result = paired_date_block_bootstrap(
        rows,
        challenger="challenger",
        baseline="baseline",
        metric="error_rate",
        iterations=100,
        seed=1,
    )
    assert result["point_estimate"] == -1.0
