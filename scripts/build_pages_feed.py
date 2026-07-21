from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PUBLICATION_SAFETY_CHECKS = (
    "historical_holdout_passed",
    "model_is_frozen",
    "coverage_at_least_95_percent",
    "critical_high_data_errors_zero",
    "post_start_or_late_seals_zero",
)


def _validate_gate_for_publication(gate: dict[str, object]) -> None:
    checks = gate.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("운영 안전 검사가 없어 Pages 산출물을 만들지 않습니다.")
    failed = [name for name in PUBLICATION_SAFETY_CHECKS if checks.get(name) is not True]
    if failed:
        raise RuntimeError(f"운영 안전 검사가 실패했습니다: {', '.join(failed)}")


def _read_safe_feed(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "prediction-feed-v1":
        raise RuntimeError(f"지원하지 않는 예측 피드 스키마입니다: {path.name}")
    quality = payload.get("quality")
    if not isinstance(quality, dict) or quality.get("status") != "passed":
        raise RuntimeError(f"품질 검사를 통과하지 못한 예측 피드입니다: {path.name}")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError(f"예측 목록이 올바르지 않습니다: {path.name}")
    forbidden = {"result", "home_win", "winner", "home_score", "away_score"}
    for prediction in predictions:
        if not isinstance(prediction, dict) or forbidden.intersection(prediction):
            raise RuntimeError(f"결과 정보가 포함된 예측 피드입니다: {path.name}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feeds-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--model-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    _validate_gate_for_publication(gate)
    feeds = sorted(args.feeds_dir.glob("**/prediction-*.json"))
    if not feeds:
        raise RuntimeError("게시할 예측 피드가 없습니다.")
    destination = args.output_dir
    archive = destination / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    latest = feeds[-1]
    _read_safe_feed(latest)
    shutil.copy2(latest, destination / "latest.json")
    for feed in feeds:
        payload = _read_safe_feed(feed)
        target_date = str(payload["target_date_et"])
        shutil.copy2(feed, archive / f"{target_date}.json")
    shutil.copy2(args.gate, destination / "status.json")
    shutil.copy2(args.model_summary, destination / "model-validation.json")
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
