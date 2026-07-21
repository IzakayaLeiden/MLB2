from __future__ import annotations

import json
import sys

import pytest

from scripts.build_pages_feed import main as build_pages_main


def test_daily_shadow_workflow_contract() -> None:
    workflow = open(".github/workflows/daily-shadow.yml", encoding="utf-8").read()
    assert 'cron: "17 8 * * *"' in workflow
    assert 'timezone: "America/New_York"' in workflow
    assert "--draft" in workflow
    assert 'test "$status" = "404"' in workflow
    assert "existing=$(find" in workflow
    assert "gh release upload" in workflow
    assert "actions: write" in workflow
    assert 'gh workflow run pages.yml --ref "$GITHUB_REF_NAME"' in workflow
    assert "git add" not in workflow
    assert "git commit" not in workflow


def test_pages_workflow_publishes_while_future_validation_is_monitoring() -> None:
    workflow = open(".github/workflows/pages.yml", encoding="utf-8").read()
    assert "future_validation=monitoring" in workflow
    assert "publish=false" not in workflow
    assert "needs.build.outputs.publish == 'true'" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "dist/pages" in workflow
    assert "contents: write" in workflow
    assert "workflow_run:" not in workflow


def _safe_feed() -> dict[str, object]:
    return {
        "schema_version": "prediction-feed-v1",
        "target_date_et": "2026-07-21",
        "quality": {"status": "passed"},
        "predictions": [],
    }


def _run_builder(tmp_path, gate_payload: dict[str, object]) -> None:
    feeds = tmp_path / "feeds"
    feeds.mkdir()
    (feeds / "prediction-20260721T120000Z.json").write_text(json.dumps(_safe_feed()), encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(gate_payload), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"

    argv = [
            sys.executable,
            "--feeds-dir",
            str(feeds),
            "--gate",
            str(gate),
            "--model-summary",
            str(summary),
            "--output-dir",
            str(output),
        ]
    original = sys.argv
    sys.argv = argv
    try:
        build_pages_main()
    finally:
        sys.argv = original


def test_pages_builder_publishes_safe_feed_while_future_gate_is_pending(tmp_path) -> None:
    gate = {
        "passed": False,
        "checks": {
            "historical_holdout_passed": True,
            "model_is_frozen": True,
            "coverage_at_least_95_percent": True,
            "critical_high_data_errors_zero": True,
            "post_start_or_late_seals_zero": True,
        },
    }

    _run_builder(tmp_path, gate)

    assert (tmp_path / "output" / "latest.json").exists()
    assert json.loads((tmp_path / "output" / "status.json").read_text(encoding="utf-8"))["passed"] is False


def test_pages_builder_refuses_unsafe_feed(tmp_path) -> None:
    gate = {
        "passed": False,
        "checks": {
            "historical_holdout_passed": True,
            "model_is_frozen": True,
            "coverage_at_least_95_percent": True,
            "critical_high_data_errors_zero": False,
            "post_start_or_late_seals_zero": True,
        },
    }

    with pytest.raises(RuntimeError, match="운영 안전 검사"):
        _run_builder(tmp_path, gate)

    assert not (tmp_path / "output" / "latest.json").exists()
