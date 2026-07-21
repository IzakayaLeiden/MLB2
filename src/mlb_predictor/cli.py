from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .features import FeatureConfig
from .io import read_rows
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

    raise AssertionError(f"처리되지 않은 명령: {args.command}")
