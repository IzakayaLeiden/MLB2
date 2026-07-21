from __future__ import annotations

import json
from pathlib import Path

from mlb_predictor import cli


def test_train_forwards_arguments_and_prints_manifest(monkeypatch, capsys, tmp_path: Path) -> None:
    expected_features_path = tmp_path / "pregame_features.parquet"
    expected_output_dir = tmp_path / "model"
    manifest = {"status": "trained", "manifest_path": str(expected_output_dir / "manifest.json")}
    captured: dict[str, object] = {}

    def fake_train_model_artifacts(
        features_path: Path,
        output_dir: Path,
        *,
        train_fraction: float,
        validation_fraction: float,
        l2: float,
        n_bins: int,
    ) -> dict[str, object]:
        captured.update(
            {
                "features_path": features_path,
                "output_dir": output_dir,
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "l2": l2,
                "n_bins": n_bins,
            }
        )
        return manifest

    monkeypatch.setattr(cli, "train_model_artifacts", fake_train_model_artifacts)

    exit_code = cli.main(
        [
            "train",
            "--features",
            str(expected_features_path),
            "--output-dir",
            str(expected_output_dir),
            "--train-fraction",
            "0.65",
            "--validation-fraction",
            "0.15",
            "--l2",
            "2.5",
            "--calibration-bins",
            "8",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "features_path": expected_features_path,
        "output_dir": expected_output_dir,
        "train_fraction": 0.65,
        "validation_fraction": 0.15,
        "l2": 2.5,
        "n_bins": 8,
    }
    assert json.loads(capsys.readouterr().out) == manifest


def test_validate_remains_an_independent_command(monkeypatch, capsys, tmp_path: Path) -> None:
    games_path = tmp_path / "games.parquet"
    features_path = tmp_path / "pregame_features.parquet"
    games = [{"game_id": 1}]
    features = [{"game_id": 1}]
    reports = [_Report("raw"), _Report("features"), _Report("pair")]
    reads: list[Path] = []
    raised_with: list[_Report] = []

    def fake_read_rows(path: Path) -> list[dict[str, int]]:
        reads.append(path)
        return games if path == games_path else features

    def fail_if_training_runs(*args, **kwargs):
        raise AssertionError("validate 명령이 학습 함수를 호출했습니다.")

    monkeypatch.setattr(cli, "read_rows", fake_read_rows)
    monkeypatch.setattr(cli, "validate_raw_games", lambda rows: reports[0])
    monkeypatch.setattr(cli, "validate_feature_rows", lambda rows, *, recent_window: reports[1])
    monkeypatch.setattr(cli, "validate_dataset_pair", lambda left, right, *, recent_window: reports[2])
    monkeypatch.setattr(cli, "raise_for_failed_reports", lambda *items: raised_with.extend(items))
    monkeypatch.setattr(cli, "train_model_artifacts", fail_if_training_runs)

    exit_code = cli.main(
        [
            "validate",
            "--games",
            str(games_path),
            "--features",
            str(features_path),
            "--recent-window",
            "7",
        ]
    )

    assert exit_code == 0
    assert reads == [games_path, features_path]
    assert raised_with == reports
    assert json.loads(capsys.readouterr().out) == {
        "reports": [report.to_dict() for report in reports]
    }


def test_audit_forwards_reproducibility_inputs(monkeypatch, capsys, tmp_path: Path) -> None:
    features = tmp_path / "features.parquet"
    games = tmp_path / "games.parquet"
    exclusions = tmp_path / "exclusions.csv"
    model_path = tmp_path / "model-v1.json"
    output = tmp_path / "audit"
    model = {"model_version": "model-v1"}
    rows = [{"game_id": 1}]
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_frozen_model", lambda path: model)
    monkeypatch.setattr(cli, "read_rows", lambda path: rows)
    monkeypatch.setattr(
        cli,
        "write_audit_bundle",
        lambda **kwargs: captured.update(kwargs) or {"schema_version": "model-audit-manifest-v1"},
    )

    exit_code = cli.main(
        [
            "audit",
            "--features",
            str(features),
            "--games",
            str(games),
            "--exclusions",
            str(exclusions),
            "--model",
            str(model_path),
            "--output-dir",
            str(output),
            "--code-revision",
            "abc123",
            "--bootstrap-iterations",
            "500",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "feature_rows": rows,
        "features_path": features,
        "games_path": games,
        "exclusions_path": exclusions,
        "model": model,
        "model_path": model_path,
        "output_dir": output,
        "code_revision": "abc123",
        "iterations": 500,
        "seed": 7,
    }
    assert json.loads(capsys.readouterr().out)["schema_version"] == "model-audit-manifest-v1"


class _Report:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name}
