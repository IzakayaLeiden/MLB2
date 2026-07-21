from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feeds-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--model-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate.get("passed") is not True:
        raise RuntimeError("공개 게이트가 실패하여 Pages 산출물을 만들지 않습니다.")
    feeds = sorted(args.feeds_dir.glob("**/prediction-*.json"))
    if not feeds:
        raise RuntimeError("게시할 예측 피드가 없습니다.")
    destination = args.output_dir
    archive = destination / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    latest = feeds[-1]
    shutil.copy2(latest, destination / "latest.json")
    for feed in feeds:
        payload = json.loads(feed.read_text(encoding="utf-8"))
        target_date = str(payload["target_date_et"])
        shutil.copy2(feed, archive / f"{target_date}.json")
    shutil.copy2(args.gate, destination / "status.json")
    shutil.copy2(args.model_summary, destination / "model-validation.json")
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
