from __future__ import annotations

from copy import deepcopy

import pytest

from mlb_predictor.features import FeatureConfig, build_pregame_features
from mlb_predictor.models import DataQualityError
from mlb_predictor.quality import (
    raise_for_failed_reports,
    validate_dataset_pair,
    validate_feature_rows,
    validate_raw_games,
)


def issue_codes(report):
    return {issue.code for issue in report.issues}


def test_raw_quality_rejects_duplicates_and_target_mismatch(game_factory) -> None:
    valid = game_factory(800, "2025-07-01", 1, 2, 3, 2)
    duplicate = deepcopy(valid)
    duplicate["home_score"] = 1
    duplicate["away_score"] = 5
    report = validate_raw_games([valid, duplicate])

    assert report.passed is False
    assert "duplicate_game_id" in issue_codes(report)
    assert "target_score_mismatch" in issue_codes(report)
    with pytest.raises(DataQualityError):
        raise_for_failed_reports(report)


def test_feature_builder_fails_before_state_update_on_duplicate_game_id(game_factory) -> None:
    game = game_factory(805, "2025-07-06", 1, 2, 3, 2)
    with pytest.raises(DataQualityError):
        build_pregame_features([game, deepcopy(game)])


@pytest.mark.parametrize("game_id", [None, 0, -1, "abc", 3.5, True])
def test_raw_quality_requires_positive_integer_game_id(game_factory, game_id) -> None:
    game = game_factory(806, "2025-07-07", 1, 2, 3, 2)
    game["game_id"] = game_id
    report = validate_raw_games([game])
    assert "invalid_game_id" in issue_codes(report)


def test_feature_quality_detects_outcome_columns_and_lookahead(game_factory) -> None:
    game = game_factory(801, "2025-07-02", 1, 2, 3, 2)
    feature = build_pregame_features([game])[0]
    feature["home_score"] = 3
    feature["home_history_through_date"] = "2025-07-02"
    report = validate_feature_rows([feature])

    assert report.passed is False
    assert "post_outcome_columns" in issue_codes(report)
    assert "lookahead_history_date" in issue_codes(report)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_feature_quality_rejects_invalid_probability(game_factory, value) -> None:
    feature = build_pregame_features([game_factory(802, "2025-07-03", 1, 2, 3, 2)])[0]
    feature["elo_expected_home_win_probability"] = value
    report = validate_feature_rows([feature])
    assert "invalid_probability" in issue_codes(report)


def test_pair_quality_requires_exact_game_coverage(game_factory) -> None:
    games = [
        game_factory(803, "2025-07-04", 1, 2, 3, 2),
        game_factory(804, "2025-07-05", 3, 4, 5, 1),
    ]
    features = build_pregame_features(games, FeatureConfig())[:1]
    report = validate_dataset_pair(games, features)
    assert report.passed is False
    assert issue_codes(report) == {"game_id_coverage_mismatch", "row_count_mismatch"}


def test_pair_quality_rejects_target_or_identity_mismatch(game_factory) -> None:
    games = [game_factory(807, "2025-07-08", 1, 2, 3, 2)]
    features = build_pregame_features(games)
    features[0]["home_win"] = 0
    features[0]["away_team_id"] = 99

    report = validate_dataset_pair(games, features)

    assert report.passed is False
    assert "raw_feature_value_mismatch" in issue_codes(report)
    assert report.metrics["raw_feature_mismatch_rows"] == 1
