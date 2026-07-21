from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable

from .collector import MlbStatsApiClient
from .features import FeatureConfig, build_pregame_features
from .io import sha256_file, write_json, write_rows_csv, write_rows_parquet
from .normalizer import normalize_schedule_payloads
from .quality import raise_for_failed_reports, validate_dataset_pair, validate_feature_rows, validate_raw_games


_CANONICAL_GENERATED_FILES = (
    "processed/history_games.csv",
    "processed/history_games.parquet",
    "processed/games.csv",
    "processed/games.parquet",
    "features/pregame_features.csv",
    "features/pregame_features.parquet",
    "reports/skipped_games.csv",
    "reports/quality.json",
    "manifest.json",
)

_PARTIAL_DATA_FILES = (
    "processed/history_games.csv",
    "processed/history_games.parquet",
    "processed/games.csv",
    "processed/games.parquet",
    "features/pregame_features.csv",
    "features/pregame_features.parquet",
)


def _archive_files(root: Path, relative_paths: Iterable[str], destination: Path) -> Path | None:
    moved = False
    for relative_path in relative_paths:
        source = root / relative_path
        if not source.exists():
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        moved = True
    return destination if moved else None


def _parse_build_dates(start_date: str, end_date: str, history_start_date: str | None) -> tuple[date, date, date, str]:
    output_start = date.fromisoformat(start_date)
    output_end = date.fromisoformat(end_date)
    if output_start > output_end:
        raise ValueError("start_date는 end_date보다 늦을 수 없습니다.")

    if history_start_date is None:
        history_start = date(output_start.year, 1, 1)
        history_policy = "calendar_year_start_default"
    else:
        history_start = date.fromisoformat(history_start_date)
        history_policy = "explicit"
    if history_start > output_start:
        raise ValueError("history_start_date는 start_date보다 늦을 수 없습니다.")
    return output_start, output_end, history_start, history_policy


def build_dataset(
    *,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    history_start_date: str | None = None,
    chunk_days: int = 7,
    refresh: bool = False,
    feature_config: FeatureConfig | None = None,
) -> dict[str, Any]:
    feature_config = feature_config or FeatureConfig()
    output_start, output_end, history_start, history_policy = _parse_build_dates(start_date, end_date, history_start_date)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(cache_dir) if cache_dir is not None else root / "raw"
    processed_dir = root / "processed"
    feature_dir = root / "features"
    report_dir = root / "reports"
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")

    previous_archive = _archive_files(
        root,
        _CANONICAL_GENERATED_FILES,
        root / "previous_runs" / run_id,
    )
    manifest_base: dict[str, Any] = {
        "pipeline_version": "0.2.0",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "source": {
            "name": "MLB Stats API schedule",
            "history_start_date": history_start.isoformat(),
            "start_date": output_start.isoformat(),
            "end_date": output_end.isoformat(),
            "history_policy": history_policy,
            "chunk_days": chunk_days,
        },
        "previous_run_archive": str(previous_archive.relative_to(root)) if previous_archive else None,
    }
    write_json(
        root / "manifest.json",
        {
            **manifest_base,
            "build_status": "running",
            "quality_gate_passed": False,
            "artifacts_valid": False,
        },
    )

    try:
        client = MlbStatsApiClient(raw_dir)
        cached_payloads = client.fetch_schedule(
            history_start,
            output_end,
            chunk_days=chunk_days,
            refresh=refresh,
        )
        history_games, skipped = normalize_schedule_payloads(
            (item.payload for item in cached_payloads),
            start_date=history_start,
            end_date=output_end,
        )

        history_report = validate_raw_games(history_games)
        all_features = build_pregame_features(history_games, feature_config) if history_report.passed else []
        games = [row for row in history_games if str(row["official_date"]) >= output_start.isoformat()]
        features = [row for row in all_features if str(row["official_date"]) >= output_start.isoformat()]
        output_report = validate_raw_games(games)
        output_report.dataset = "output_games"
        feature_report = validate_feature_rows(features, recent_window=feature_config.recent_window)
        pair_report = validate_dataset_pair(games, features, recent_window=feature_config.recent_window)

        skipped_path = write_rows_csv(report_dir / "skipped_games.csv", (item.to_dict() for item in skipped))
        quality_path = write_json(
            report_dir / "quality.json",
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "reports": [
                    history_report.to_dict(),
                    output_report.to_dict(),
                    feature_report.to_dict(),
                    pair_report.to_dict(),
                ],
            },
        )
        raise_for_failed_reports(history_report, output_report, feature_report, pair_report)

        history_games_csv = write_rows_csv(processed_dir / "history_games.csv", history_games)
        history_games_parquet = write_rows_parquet(processed_dir / "history_games.parquet", history_games)
        games_csv = write_rows_csv(processed_dir / "games.csv", games)
        games_parquet = write_rows_parquet(processed_dir / "games.parquet", games)
        features_csv = write_rows_csv(feature_dir / "pregame_features.csv", features)
        features_parquet = write_rows_parquet(feature_dir / "pregame_features.parquet", features)

        artifact_paths = [
            history_games_csv,
            history_games_parquet,
            games_csv,
            games_parquet,
            features_csv,
            features_parquet,
            skipped_path,
            quality_path,
        ]
        manifest = {
            **manifest_base,
            "build_status": "completed",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                **manifest_base["source"],
                "raw_files": [str(item.cache_path) for item in cached_payloads],
                "requests": [
                    {
                        "start_date": item.start_date,
                        "end_date": item.end_date,
                        "from_cache": item.from_cache,
                        "source_url": item.source_url,
                    }
                    for item in cached_payloads
                ],
            },
            "feature_config": asdict(feature_config),
            "feature_cutoff_policy": "prior_official_date_only",
            "counts": {
                "history_games": len(history_games),
                "warmup_games": len(history_games) - len(games),
                "normalized_games": len(games),
                "pregame_feature_rows": len(features),
                "skipped_games": len(skipped),
            },
            "quality_gate_passed": True,
            "artifacts_valid": True,
            "artifacts": {
                str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in artifact_paths
            },
        }
        manifest_path = write_json(root / "manifest.json", manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest
    except Exception as exc:
        failed_archive = _archive_files(
            root,
            _PARTIAL_DATA_FILES,
            root / "failed_runs" / run_id,
        )
        failure_manifest = {
            **manifest_base,
            "build_status": "failed",
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "quality_gate_passed": False,
            "artifacts_valid": False,
            "failed_artifact_archive": str(failed_archive.relative_to(root)) if failed_archive else None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_json(root / "manifest.json", failure_manifest)
        raise
