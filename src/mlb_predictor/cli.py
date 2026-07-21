from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .audit import write_audit_bundle
from .features import FeatureConfig
from .backtest import DEFAULT_L2_VALUES, run_backtest
from .collector import MlbStatsApiClient
from .forecasting import create_prediction_feed, grade_prediction_feed, load_frozen_model
from .gate import evaluate_public_gate
from .io import read_rows, write_json
from .model_v3_backtest import run_model_v3_backtest
from .model_v3_snapshot import collect_pregame_model_v3_snapshots, write_pregame_model_v3_snapshots
from .normalizer import normalize_future_schedule_payloads
from .pitching import collect_pregame_pitching_snapshots, write_pregame_pitching_snapshots
from .pitching_backtest import DEFAULT_V2_L2_VALUES, run_retrospective_pitching_backtest
from .pipeline import build_dataset
from .quality import raise_for_failed_reports, validate_dataset_pair, validate_feature_rows, validate_raw_games
from .training import train_model_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLB 경기 전 예측 데이터셋 파이프라인")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="일정을 수집하고 정규화·피처 생성·검증을 수행합니다.")
    build.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    build.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    build.add_argument(
        "--history-start-date",
        help="피처 워밍업 시작일입니다. 생략하면 start-date가 속한 해의 1월 1일을 사용합니다.",
    )
    build.add_argument("--output-dir", type=Path, default=Path("data/dataset"))
    build.add_argument("--cache-dir", type=Path, help="재사용할 원본 API 캐시 디렉터리")
    build.add_argument("--chunk-days", type=int, default=7)
    build.add_argument("--refresh", action="store_true")
    build.add_argument("--recent-window", type=int, default=10)

    validate = subparsers.add_parser("validate", help="기존 원본·피처 산출물의 품질 게이트를 다시 실행합니다.")
    validate.add_argument("--games", type=Path, required=True)
    validate.add_argument("--features", type=Path, required=True)
    validate.add_argument("--recent-window", type=int, default=10)

    train = subparsers.add_parser("train", help="시간 순서 분할로 모델을 학습·평가하고 산출물을 저장합니다.")
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--train-fraction", type=float, default=0.6)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--l2", type=float, default=1.0)
    train.add_argument("--calibration-bins", type=int, default=10)

    backtest = subparsers.add_parser("backtest", help="2022~2024 워크포워드 선택과 봉인된 2025 홀드아웃 평가를 수행합니다.")
    backtest.add_argument("--features", type=Path, required=True)
    backtest.add_argument("--output-dir", type=Path, required=True)
    backtest.add_argument("--cutoff-date", default="2026-07-20")
    backtest.add_argument("--l2-values", type=float, nargs="+", default=list(DEFAULT_L2_VALUES))

    forecast = subparsers.add_parser("forecast", help="당일 예정 경기 피처와 봉인 확률 피드를 생성합니다.")
    forecast.add_argument("--history-games", type=Path, required=True)
    forecast.add_argument("--model", type=Path, required=True)
    forecast.add_argument("--target-date", required=True)
    forecast.add_argument("--output-dir", type=Path, required=True)
    forecast.add_argument("--cache-dir", type=Path, default=Path("data/forecast-cache"))
    forecast.add_argument("--schedule-json", type=Path)
    forecast.add_argument("--created-at-utc")

    grade = subparsers.add_parser("grade", help="종료 경기 결과와 봉인 예측을 연결합니다.")
    grade.add_argument("--feed", type=Path, required=True)
    grade.add_argument("--completed-games", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)

    gate = subparsers.add_parser("gate", help="역사·미래 검증과 운영 품질의 공개 게이트를 판정합니다.")
    gate.add_argument("--feeds-dir", type=Path, required=True)
    gate.add_argument("--grades-dir", type=Path, required=True)
    gate.add_argument("--model", type=Path, required=True)
    gate.add_argument("--as-of-date", required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--critical-errors", type=int, default=0)
    gate.add_argument("--high-errors", type=int, default=0)

    audit = subparsers.add_parser("audit", help="경기별 OOF 예측과 날짜 블록 통계 검증 번들을 생성합니다.")
    audit.add_argument("--features", type=Path, required=True)
    audit.add_argument("--games", type=Path, required=True)
    audit.add_argument("--exclusions", type=Path, required=True)
    audit.add_argument("--model", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--code-revision", required=True)
    audit.add_argument("--bootstrap-iterations", type=int, default=10_000)
    audit.add_argument("--seed", type=int, default=20260721)

    pitching = subparsers.add_parser("snapshot-pitching", help="model-v2용 경기 전 선발투수·불펜 스냅샷을 봉인합니다.")
    pitching.add_argument("--target-date", required=True)
    pitching.add_argument("--output-dir", type=Path, default=Path("data/pitching-snapshots"))
    pitching.add_argument("--cache-dir", type=Path, default=Path("data/pitching-cache"))
    pitching.add_argument("--created-at-utc")
    pitching.add_argument("--refresh", action="store_true")

    model_v3_snapshot = subparsers.add_parser(
        "snapshot-model-v3",
        help="model-v3용 선발·라인업·active roster·불펜 피처를 경기 전 봉인합니다.",
    )
    model_v3_snapshot.add_argument("--history-games", type=Path, required=True)
    model_v3_snapshot.add_argument("--target-date", required=True)
    model_v3_snapshot.add_argument("--output-dir", type=Path, default=Path("data/model-v3-snapshots"))
    model_v3_snapshot.add_argument("--cache-dir", type=Path, default=Path("data/model-v3-snapshot-cache"))
    model_v3_snapshot.add_argument("--created-at-utc")
    model_v3_snapshot.add_argument("--refresh", action="store_true")

    pitching_backtest = subparsers.add_parser("backtest-pitching-v2", help="이전 시즌 선발 성적 challenger를 회고적으로 평가합니다.")
    pitching_backtest.add_argument("--features", type=Path, required=True)
    pitching_backtest.add_argument("--output-dir", type=Path, required=True)
    pitching_backtest.add_argument("--cache-dir", type=Path, default=Path("data/pitching-backtest-cache"))
    pitching_backtest.add_argument("--l2-values", type=float, nargs="+", default=list(DEFAULT_V2_L2_VALUES))
    pitching_backtest.add_argument("--bootstrap-iterations", type=int, default=10_000)
    pitching_backtest.add_argument("--seed", type=int, default=20260721)
    pitching_backtest.add_argument("--refresh", action="store_true")

    model_v3 = subparsers.add_parser("backtest-model-v3", help="선발 휴식·예상 이닝 challenger를 회고적으로 평가합니다.")
    model_v3.add_argument("--features", type=Path, required=True)
    model_v3.add_argument("--games", type=Path, required=True)
    model_v3.add_argument("--output-dir", type=Path, required=True)
    model_v3.add_argument("--cache-dir", type=Path, default=Path("data/pitching-backtest-cache"))
    model_v3.add_argument("--l2-values", type=float, nargs="+", default=list(DEFAULT_V2_L2_VALUES))
    model_v3.add_argument("--bootstrap-iterations", type=int, default=10_000)
    model_v3.add_argument("--seed", type=int, default=20260721)
    model_v3.add_argument("--refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    if args.command == "build":
        manifest = build_dataset(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            history_start_date=args.history_start_date,
            chunk_days=args.chunk_days,
            refresh=args.refresh,
            feature_config=FeatureConfig(recent_window=args.recent_window),
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate":
        games = read_rows(args.games)
        features = read_rows(args.features)
        reports = [
            validate_raw_games(games),
            validate_feature_rows(features, recent_window=args.recent_window),
            validate_dataset_pair(games, features, recent_window=args.recent_window),
        ]
        print(json.dumps({"reports": [report.to_dict() for report in reports]}, ensure_ascii=False, indent=2))
        raise_for_failed_reports(*reports)
        return 0

    if args.command == "train":
        manifest = train_model_artifacts(
            features_path=args.features,
            output_dir=args.output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            l2=args.l2,
            n_bins=args.calibration_bins,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "backtest":
        status = run_backtest(
            read_rows(args.features),
            output_dir=args.output_dir,
            cutoff_date=args.cutoff_date,
            l2_values=args.l2_values,
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["passed"] else 2

    if args.command == "forecast":
        if args.schedule_json:
            schedule_payloads = [json.loads(args.schedule_json.read_text(encoding="utf-8"))]
        else:
            schedule_payloads = [
                result.payload
                for result in MlbStatsApiClient(args.cache_dir).fetch_schedule(
                    args.target_date,
                    args.target_date,
                    chunk_days=1,
                )
            ]
        scheduled, skipped = normalize_future_schedule_payloads(schedule_payloads, target_date=args.target_date)
        feed, path = create_prediction_feed(
            completed_games=read_rows(args.history_games),
            scheduled_games=scheduled,
            model_path=args.model,
            output_root=args.output_dir,
            target_date=args.target_date,
            created_at_utc=args.created_at_utc,
        )
        print(json.dumps({"path": str(path), "feed": feed, "skipped": [item.to_dict() for item in skipped]}, ensure_ascii=False, indent=2))
        return 0 if feed["quality"]["status"] == "passed" else 2

    if args.command == "grade":
        feed = json.loads(args.feed.read_text(encoding="utf-8"))
        grade = grade_prediction_feed(feed, read_rows(args.completed_games))
        write_json(args.output, grade)
        print(json.dumps(grade, ensure_ascii=False, indent=2))
        return 0 if grade["quality_status"] == "passed" else 2

    if args.command == "gate":
        feeds = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.feeds_dir.glob("**/prediction-*.json"))]
        grades = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.grades_dir.glob("**/grade-*.json"))]
        result = evaluate_public_gate(
            feeds=feeds,
            grades=grades,
            model=load_frozen_model(args.model),
            as_of_date=args.as_of_date,
            critical_errors=args.critical_errors,
            high_errors=args.high_errors,
        )
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 2

    if args.command == "audit":
        model = load_frozen_model(args.model)
        feature_rows = read_rows(args.features)
        manifest = write_audit_bundle(
            feature_rows=feature_rows,
            features_path=args.features,
            games_path=args.games,
            exclusions_path=args.exclusions,
            model=model,
            model_path=args.model,
            output_dir=args.output_dir,
            code_revision=args.code_revision,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "snapshot-pitching":
        client = MlbStatsApiClient(args.cache_dir)
        schedule_payloads = client.fetch_schedule(args.target_date, args.target_date, chunk_days=1, refresh=args.refresh)
        scheduled_games, skipped = normalize_future_schedule_payloads(
            [payload.payload for payload in schedule_payloads],
            target_date=args.target_date,
        )
        snapshots = collect_pregame_pitching_snapshots(
            client=client,
            scheduled_games=[game for game in scheduled_games if game["forecast_eligible"]],
            schedule_sources=schedule_payloads,
            target_date=args.target_date,
            created_at_utc=args.created_at_utc,
            refresh=args.refresh,
        )
        paths = write_pregame_pitching_snapshots(snapshots, output_dir=args.output_dir)
        failed_game_ids = [snapshot["game_id"] for snapshot in snapshots if snapshot["quality"]["status"] != "passed"]
        result = {
            "paths": [str(path) for path in paths],
            "snapshot_count": len(snapshots),
            "quality_status": "passed" if not failed_game_ids else "failed",
            "failed_game_ids": failed_game_ids,
            "skipped": [item.to_dict() for item in skipped],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not failed_game_ids else 2

    if args.command == "snapshot-model-v3":
        snapshots = collect_pregame_model_v3_snapshots(
            client=MlbStatsApiClient(args.cache_dir),
            completed_games=read_rows(args.history_games),
            target_date=args.target_date,
            created_at_utc=args.created_at_utc,
            refresh=args.refresh,
        )
        paths = write_pregame_model_v3_snapshots(snapshots, output_dir=args.output_dir)
        failed_game_ids = [
            int(snapshot["game_id"])
            for snapshot in snapshots
            if snapshot["quality"]["status"] != "passed"
        ]
        result = {
            "paths": [str(path) for path in paths],
            "snapshot_count": len(snapshots),
            "quality_status": "passed" if snapshots and not failed_game_ids else "failed",
            "failed_game_ids": failed_game_ids,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["quality_status"] == "passed" else 2

    if args.command == "backtest-pitching-v2":
        result = run_retrospective_pitching_backtest(
            feature_rows=read_rows(args.features),
            client=MlbStatsApiClient(args.cache_dir),
            output_dir=args.output_dir,
            refresh=args.refresh,
            l2_values=args.l2_values,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        report = result["report"]
        print(json.dumps({
            "selected_candidate": report["selected_candidate"],
            "development_holdout_metrics": {
                name: evaluation["metrics"]
                for name, evaluation in report["development_holdout"].items()
            },
            "score_gate_criteria": report["score_gate_criteria"],
            "retrospective_score_gate_passed": report["retrospective_score_gate_passed"],
            "promotion_allowed": report["promotion_allowed"],
            "output_dir": str(args.output_dir),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "backtest-model-v3":
        result = run_model_v3_backtest(
            feature_rows=read_rows(args.features),
            completed_games=read_rows(args.games),
            client=MlbStatsApiClient(args.cache_dir),
            output_dir=args.output_dir,
            refresh=args.refresh,
            l2_values=args.l2_values,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        report = result["report"]
        print(json.dumps({
            "selected_candidate": report["selected_candidate"],
            "development_holdout_metrics": {
                name: evaluation["metrics"]
                for name, evaluation in report["development_holdout"].items()
            },
            "score_gate_criteria": report["score_gate_criteria"],
            "retrospective_score_gate_passed": report["retrospective_score_gate_passed"],
            "promotion_allowed": report["promotion_allowed"],
            "output_dir": str(args.output_dir),
        }, ensure_ascii=False, indent=2))
        return 0

    raise AssertionError(f"처리되지 않은 명령: {args.command}")
