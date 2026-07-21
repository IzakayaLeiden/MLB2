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
    assert "git add" not in workflow
    assert "git commit" not in workflow


def test_pages_workflow_preserves_last_good_feed_on_failed_gate() -> None:
    workflow = open(".github/workflows/pages.yml", encoding="utf-8").read()
    assert "publish=false" in workflow
    assert "needs.build.outputs.publish == 'true'" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "dist/pages" in workflow


def test_pages_builder_refuses_failed_gate(tmp_path) -> None:
    feeds = tmp_path / "feeds"
    feeds.mkdir()
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"passed": False}), encoding="utf-8")
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
        with pytest.raises(RuntimeError, match="공개 게이트"):
            build_pages_main()
    finally:
        sys.argv = original

    assert not output.exists()
