from __future__ import annotations

import json

import pytest

from mlb_predictor.backtest import evaluate_sealed_holdout, freeze_model_v1, select_model


def feature_row(game_id: int, season: int, positive: bool) -> dict[str, object]:
    x = 1.0 if positive else -1.0
    return {
        "game_id": game_id,
        "season": season,
        "official_date": f"{season}-06-{game_id % 20 + 1:02d}",
        "home_win": int(positive),
        "elo_expected_home_win_probability": 0.9 if positive else 0.1,
        "home_elo_minus_away": 100.0 * x,
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


def test_walk_forward_selects_by_log_loss_then_brier() -> None:
    selection = select_model(multi_year_rows(), l2_values=[0.1, 1.0])

    assert selection["selection_seasons"] == [2022, 2023, 2024]
    assert selection["ranking_policy"] == ["log_loss", "brier_score"]
    assert selection["selected"]["candidate"] in selection["candidates"]
    assert set(selection["candidates"]) == {
        "constant",
        "elo",
        "logistic:l2=0.1",
        "logistic_platt:l2=0.1",
        "logistic:l2=1",
        "logistic_platt:l2=1",
    }


def test_holdout_is_used_once_and_freezes_only_after_pass(tmp_path) -> None:
    rows = multi_year_rows()
    selection = select_model(rows, l2_values=[1.0])
    state = tmp_path / "holdout-state.json"

    result = evaluate_sealed_holdout(rows, selection, state_path=state)

    assert result["usage_count"] == 1
    assert result["passed"] is True
    assert json.loads(state.read_text(encoding="utf-8"))["usage_count"] == 1
    with pytest.raises(RuntimeError, match="이미 사용"):
        evaluate_sealed_holdout(rows, selection, state_path=state)

    model = freeze_model_v1(
        rows,
        selection,
        result,
        cutoff_date="2026-07-20",
        output_path=tmp_path / "model-v1.json",
    )
    assert model["frozen"] is True
    assert model["retraining_during_shadow"] is False
    assert model["training"]["cutoff_date"] == "2026-07-20"


def test_failed_holdout_cannot_freeze_model(tmp_path) -> None:
    selection = {"selected": {"candidate": "constant", "model_type": "constant"}}
    with pytest.raises(RuntimeError, match="통과하지 않아"):
        freeze_model_v1(
            multi_year_rows(),
            selection,
            {"passed": False},
            cutoff_date="2026-07-20",
            output_path=tmp_path / "model-v1.json",
        )
