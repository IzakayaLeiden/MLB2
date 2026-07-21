from __future__ import annotations

from datetime import date
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audit import generate_historical_audit_rows, paired_date_block_bootstrap
from .bullpen_backtest import (
    PROBABILISTIC_RELIEVER_FEATURE_NAMES,
    RELIEVER_FEATURE_NAMES,
    add_neutral_probabilistic_reliever_features,
    add_neutral_reliever_features,
    add_probabilistic_reliever_features,
    add_reliever_availability_features,
    collect_full_season_pitcher_rosters,
    collect_reliever_pitching_data,
    ensure_probabilistic_reliever_feature_values,
)
from .collector import MlbStatsApiClient
from .evaluation import evaluate_prediction_set
from .io import write_json, write_rows_csv
from .lineup import (
    LEAGUE_OBP_PRIOR,
    LEAGUE_SLG_PRIOR,
    LINEUP_FEATURE_NAMES,
    LINEUP_WEIGHTS,
    add_lineup_features,
    add_neutral_lineup_features,
    collect_current_season_batting_game_logs,
    collect_historical_lineups,
    collect_prior_season_batting_platoon_stats,
    collect_prior_season_batting_stats,
    smoothed_platoon_ops,
)
from .modeling import DEFAULT_FEATURE_SPECS, FeatureSpec, StandardizedLogisticRegression, extract_feature_matrix
from .pitching import PITCHER_RATE_PRIORS
from .pitching_backtest import (
    BACKTEST_SEASONS,
    DEFAULT_V2_L2_VALUES,
    DEVELOPMENT_HOLDOUT_SEASON,
    STARTER_FEATURE_NAMES,
    _predict_season as _predict_v2_season,
    add_bullpen_workload_features,
    add_rolling_starter_features,
    collect_current_season_pitching_game_logs,
    collect_prior_season_pitching_stats,
    collect_team_pitching_game_logs,
)
from .run_strength import (
    RUN_STRENGTH_FEATURE_NAMES,
    add_dynamic_run_strength_features,
    add_neutral_run_strength_features,
)
from .schedule_context import (
    FATIGUE_CONTEXT_FEATURE_NAMES,
    SCHEDULE_CONTEXT_FEATURE_NAMES,
    SCHEDULE_CONTEXT_INTERACTION_NAMES,
    VENUE_CONTEXT_FEATURE_NAMES,
    add_neutral_schedule_context_features,
    add_schedule_context_features,
    add_schedule_context_interactions,
    ensure_schedule_context_feature_values,
    select_schedule_context_feature_block,
)


DEFAULT_EXPECTED_START_OUTS = 15.0
DEFAULT_STARTER_REST_DAYS = 5
PRIOR_START_WEIGHT = 3.0
RECENT_START_WINDOW = 5
RECENT_TRAINING_WINDOWS = (3, 4, 5)
LDA_SHRINKAGE_VALUES = (0.1, 0.5, 0.9)
BLEND_WEIGHTS = (0.25, 0.5, 0.75)
BLEND_FEATURE_MODES = {
    "starter_readiness_lineup_reliever_interactions",
    "starter_readiness_lineup_reliever_schedule_interactions",
    "starter_readiness_lineup_reliever_venue_interactions",
    "starter_readiness_lineup_reliever_fatigue_interactions",
    "starter_readiness_lineup_reliever_schedule_probabilistic_interactions",
}
READINESS_FEATURE_NAMES = (
    "starter_rest_days_difference",
    "starter_expected_innings_advantage",
    "away_starter_readiness_missing",
    "home_starter_readiness_missing",
)
CONTEXT_FEATURE_NAMES = (
    "recent_offense_difference",
    "recent_defense_advantage",
    "season_win_pct_signed_square",
    "recent_win_pct_signed_square",
)
STARTER_TREND_FEATURE_NAMES = (
    "starter_recent_k_minus_bb_difference",
    "starter_recent_earned_run_advantage",
    "starter_recent_outs_advantage",
    "starter_recent_pitch_efficiency_advantage",
    "away_starter_recent_form_missing",
    "home_starter_recent_form_missing",
)
PLATOON_FEATURE_NAMES = (
    "lineup_platoon_advantage",
    "lineup_same_side_exposure_advantage",
    "away_lineup_handedness_missing_rate",
    "home_lineup_handedness_missing_rate",
    "away_opposing_starter_hand_missing",
    "home_opposing_starter_hand_missing",
)
PLATOON_PERFORMANCE_FEATURE_NAMES = (
    "lineup_platoon_ops_advantage",
    "lineup_platoon_top4_ops_advantage",
    "away_lineup_platoon_stats_missing_rate",
    "home_lineup_platoon_stats_missing_rate",
)
INTERACTION_FEATURE_NAMES = (
    "elo_signed_square",
    "starter_kbb_expected_innings_interaction",
    "starter_era_expected_innings_interaction",
    "lineup_ops_expected_innings_interaction",
    "starter_bullpen_availability_interaction",
)
MODEL_V3_FEATURE_SPECS = DEFAULT_FEATURE_SPECS + tuple(
    FeatureSpec(name=name, source=name, transform="numeric", value=None)
    for name in (
        *STARTER_FEATURE_NAMES,
        *CONTEXT_FEATURE_NAMES,
        *STARTER_TREND_FEATURE_NAMES,
        *PLATOON_FEATURE_NAMES,
        *PLATOON_PERFORMANCE_FEATURE_NAMES,
        *RUN_STRENGTH_FEATURE_NAMES,
        *SCHEDULE_CONTEXT_FEATURE_NAMES,
        *SCHEDULE_CONTEXT_INTERACTION_NAMES,
        *READINESS_FEATURE_NAMES,
        *LINEUP_FEATURE_NAMES,
        *RELIEVER_FEATURE_NAMES,
        *PROBABILISTIC_RELIEVER_FEATURE_NAMES,
        *INTERACTION_FEATURE_NAMES,
    )
)


def _target(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["home_win"]) for row in rows], dtype=float)


def blend_with_elo(
    rows: Sequence[Mapping[str, Any]],
    model_probability: Sequence[float],
    *,
    model_weight: float,
) -> np.ndarray:
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be between zero and one.")
    if len(rows) != len(model_probability):
        raise ValueError("rows and model_probability must have equal length.")
    elo = np.asarray(
        [float(row["elo_expected_home_win_probability"]) for row in rows],
        dtype=float,
    )
    model = np.asarray(model_probability, dtype=float)
    return model_weight * model + (1.0 - model_weight) * elo


def _fit(rows: Sequence[Mapping[str, Any]], *, l2: float) -> StandardizedLogisticRegression:
    return StandardizedLogisticRegression(l2=l2).fit(
        extract_feature_matrix(rows, MODEL_V3_FEATURE_SPECS),
        _target(rows),
        feature_names=[spec.name for spec in MODEL_V3_FEATURE_SPECS],
    )


def _prior_expected_outs(stats: Mapping[str, Any] | None) -> tuple[float, bool]:
    materialized = stats or {}
    starts = int(materialized.get("games_started", 0) or 0)
    outs = int(materialized.get("innings_pitched_outs", 0) or 0)
    if starts <= 0 or outs <= 0:
        return DEFAULT_EXPECTED_START_OUTS, False
    return float(outs) / float(starts), True


def add_starter_readiness_features(
    rows: Iterable[Mapping[str, Any]],
    prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
    game_logs: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Add past-only starter rest and expected-innings features.

    Current-season appearances on the target official date are always excluded.
    Expected innings blend the previous-season innings per start with at most the
    five most recent current-season starts. The prior carries three-start weight.
    """

    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        season = int(row["season"])
        official_date = str(row["official_date"])
        target_date = date.fromisoformat(official_date)
        readiness: dict[str, dict[str, float | int]] = {}
        for side in ("away", "home"):
            pitcher_id = row.get(f"{side}_probable_pitcher_id")
            if pitcher_id is None:
                readiness[side] = {
                    "rest_days": DEFAULT_STARTER_REST_DAYS,
                    "expected_outs": DEFAULT_EXPECTED_START_OUTS,
                    "recent_start_count": 0,
                    "missing": 1,
                }
                continue

            numeric_pitcher_id = int(pitcher_id)
            prior_outs, has_prior = _prior_expected_outs(prior_stats.get((season - 1, numeric_pitcher_id)))
            starts = [
                appearance
                for appearance in game_logs.get((season, numeric_pitcher_id), [])
                if str(appearance["date"]) < official_date
                and int(appearance.get("stats", {}).get("games_started", 0) or 0) > 0
            ]
            recent_starts = starts[-RECENT_START_WINDOW:]
            recent_outs = [
                int(appearance.get("stats", {}).get("innings_pitched_outs", 0) or 0)
                for appearance in recent_starts
            ]
            expected_outs = (
                prior_outs * PRIOR_START_WEIGHT + float(sum(recent_outs))
            ) / (PRIOR_START_WEIGHT + len(recent_outs))
            if starts:
                days_since_start = (target_date - date.fromisoformat(str(starts[-1]["date"]))).days - 1
                rest_days = min(max(days_since_start, 0), 10)
            else:
                rest_days = DEFAULT_STARTER_REST_DAYS
            readiness[side] = {
                "rest_days": rest_days,
                "expected_outs": expected_outs,
                "recent_start_count": len(recent_starts),
                "missing": int(not has_prior and not recent_starts),
            }

        row.update(
            {
                "starter_readiness_provenance": "retrospective_prior_plus_past_game_logs_v1",
                "starter_readiness_through_policy": "strictly_before_official_date",
                "away_starter_rest_days": int(readiness["away"]["rest_days"]),
                "home_starter_rest_days": int(readiness["home"]["rest_days"]),
                "starter_rest_days_difference": int(readiness["home"]["rest_days"]) - int(readiness["away"]["rest_days"]),
                "away_starter_expected_innings": float(readiness["away"]["expected_outs"]) / 3.0,
                "home_starter_expected_innings": float(readiness["home"]["expected_outs"]) / 3.0,
                "starter_expected_innings_advantage": (
                    float(readiness["home"]["expected_outs"]) - float(readiness["away"]["expected_outs"])
                ) / 3.0,
                "away_starter_recent_start_count": int(readiness["away"]["recent_start_count"]),
                "home_starter_recent_start_count": int(readiness["home"]["recent_start_count"]),
                "away_starter_readiness_missing": int(readiness["away"]["missing"]),
                "home_starter_readiness_missing": int(readiness["home"]["missing"]),
            }
        )
        augmented.append(row)
    return augmented


def add_context_features(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Split existing past-only team form into offense and defense components."""

    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        season_difference = float(row["season_win_pct_difference"])
        recent_difference = float(row["recent_win_pct_difference"])
        row.update(
            {
                "recent_offense_difference": (
                    float(row["home_recent_runs_scored"])
                    - float(row["away_recent_runs_scored"])
                ),
                "recent_defense_advantage": (
                    float(row["away_recent_runs_allowed"])
                    - float(row["home_recent_runs_allowed"])
                ),
                "season_win_pct_signed_square": season_difference * abs(season_difference),
                "recent_win_pct_signed_square": recent_difference * abs(recent_difference),
            }
        )
        augmented.append(row)
    return augmented


def add_neutral_context_features(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row, **{name: 0.0 for name in CONTEXT_FEATURE_NAMES}) for row in rows]


def add_recent_starter_form_features(
    rows: Iterable[Mapping[str, Any]],
    game_logs: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    *,
    window: int = 3,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        season = int(row["season"])
        official_date = str(row["official_date"])
        form: dict[str, dict[str, float]] = {}
        for side in ("away", "home"):
            pitcher_id = row.get(f"{side}_probable_pitcher_id")
            starts = [] if pitcher_id is None else [
                appearance
                for appearance in game_logs.get((season, int(pitcher_id)), [])
                if str(appearance["date"]) < official_date
                and int(appearance.get("stats", {}).get("games_started", 0) or 0) > 0
            ][-window:]
            totals = {
                field: sum(int(appearance.get("stats", {}).get(field, 0) or 0) for appearance in starts)
                for field in ("batters_faced", "strikeouts", "walks", "earned_runs", "innings_pitched_outs", "pitches_thrown")
            }
            batters = float(totals["batters_faced"])
            prior_batters = 50.0
            k_rate = (totals["strikeouts"] + PITCHER_RATE_PRIORS["strikeouts"] * prior_batters) / (batters + prior_batters)
            bb_rate = (totals["walks"] + PITCHER_RATE_PRIORS["walks"] * prior_batters) / (batters + prior_batters)
            er_rate = (totals["earned_runs"] + PITCHER_RATE_PRIORS["earned_runs"] * prior_batters) / (batters + prior_batters)
            outs = float(totals["innings_pitched_outs"]) / len(starts) if starts else DEFAULT_EXPECTED_START_OUTS
            pitches = float(totals["pitches_thrown"])
            form[side] = {
                "k_minus_bb": k_rate - bb_rate,
                "earned_runs": er_rate,
                "outs": outs,
                "efficiency": float(totals["innings_pitched_outs"]) / pitches if pitches > 0 else DEFAULT_EXPECTED_START_OUTS / 85.0,
                "missing": float(not starts),
            }
        row.update(
            {
                "starter_recent_form_provenance": "past_three_starts_strictly_before_official_date_v1",
                "starter_recent_k_minus_bb_difference": form["home"]["k_minus_bb"] - form["away"]["k_minus_bb"],
                "starter_recent_earned_run_advantage": form["away"]["earned_runs"] - form["home"]["earned_runs"],
                "starter_recent_outs_advantage": (form["home"]["outs"] - form["away"]["outs"]) / 3.0,
                "starter_recent_pitch_efficiency_advantage": form["home"]["efficiency"] - form["away"]["efficiency"],
                "away_starter_recent_form_missing": form["away"]["missing"],
                "home_starter_recent_form_missing": form["home"]["missing"],
            }
        )
        augmented.append(row)
    return augmented


def add_neutral_starter_trend_features(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row, **{name: 0.0 for name in STARTER_TREND_FEATURE_NAMES}) for row in rows]


def add_platoon_features(
    rows: Iterable[Mapping[str, Any]],
    lineups: Mapping[int, Mapping[str, Any]],
    batter_prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
    pitcher_prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        season = int(row["season"])
        lineup = lineups.get(int(row["game_id"]), {})
        values: dict[str, dict[str, float]] = {}
        for side, opposing_side in (("away", "home"), ("home", "away")):
            pitcher_id = row.get(f"{opposing_side}_probable_pitcher_id")
            pitcher = pitcher_prior_stats.get((season - 1, int(pitcher_id)), {}) if pitcher_id is not None else {}
            pitch_hand = str(pitcher.get("pitch_hand") or "")
            pitch_hand_known = pitch_hand in {"L", "R"}
            player_ids = [int(value) for value in lineup.get(f"{side}_player_ids", [])][:9]
            favorable = 0
            same_side = 0
            missing = max(0, 9 - len(player_ids))
            for player_id in player_ids:
                bat_side = str(batter_prior_stats.get((season - 1, player_id), {}).get("bat_side") or "")
                if bat_side not in {"L", "R", "S"} or not pitch_hand_known:
                    missing += 1
                elif bat_side == "S" or bat_side != pitch_hand:
                    favorable += 1
                else:
                    same_side += 1
            values[side] = {
                "favorable_rate": favorable / 9.0,
                "same_side_rate": same_side / 9.0,
                "missing_rate": min(missing, 9) / 9.0,
                "pitcher_hand_missing": float(not pitch_hand_known),
            }
        row.update(
            {
                "platoon_feature_provenance": "retrospective_roster_handedness_v1",
                "lineup_platoon_advantage": values["home"]["favorable_rate"] - values["away"]["favorable_rate"],
                "lineup_same_side_exposure_advantage": values["away"]["same_side_rate"] - values["home"]["same_side_rate"],
                "away_lineup_handedness_missing_rate": values["away"]["missing_rate"],
                "home_lineup_handedness_missing_rate": values["home"]["missing_rate"],
                "away_opposing_starter_hand_missing": values["away"]["pitcher_hand_missing"],
                "home_opposing_starter_hand_missing": values["home"]["pitcher_hand_missing"],
            }
        )
        augmented.append(row)
    return augmented


def add_neutral_platoon_features(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row, **{name: 0.0 for name in PLATOON_FEATURE_NAMES}) for row in rows]


def add_platoon_performance_features(
    rows: Iterable[Mapping[str, Any]],
    lineups: Mapping[int, Mapping[str, Any]],
    batter_prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
    batter_platoon_stats: Mapping[tuple[int, int], Mapping[str, Mapping[str, Any]]],
    pitcher_prior_stats: Mapping[tuple[int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        season = int(row["season"])
        lineup = lineups.get(int(row["game_id"]), {})
        values: dict[str, dict[str, float]] = {}
        for side, opposing_side in (("away", "home"), ("home", "away")):
            pitcher_id = row.get(f"{opposing_side}_probable_pitcher_id")
            pitcher = pitcher_prior_stats.get((season - 1, int(pitcher_id)), {}) if pitcher_id is not None else {}
            pitch_hand = str(pitcher.get("pitch_hand") or "")
            split_code = "vl" if pitch_hand == "L" else "vr" if pitch_hand == "R" else ""
            player_ids = [int(value) for value in lineup.get(f"{side}_player_ids", [])][:9]
            ops_values: list[float] = []
            missing = max(0, 9 - len(player_ids))
            for player_id in player_ids:
                overall = batter_prior_stats.get((season - 1, player_id), {})
                available_splits = batter_platoon_stats.get((season - 1, player_id), {})
                split = available_splits.get(split_code) if split_code else None
                missing += int(split is None)
                ops_values.append(smoothed_platoon_ops(split, overall))
            if len(ops_values) < 9:
                ops_values.extend([LEAGUE_OBP_PRIOR + LEAGUE_SLG_PRIOR] * (9 - len(ops_values)))
            values[side] = {
                "weighted_ops": sum(value * weight for value, weight in zip(ops_values, LINEUP_WEIGHTS)) / sum(LINEUP_WEIGHTS),
                "top4_ops": sum(ops_values[:4]) / 4.0,
                "missing_rate": min(missing, 9) / 9.0,
            }
        row.update(
            {
                "platoon_performance_provenance": "prior_season_stat_splits_vl_vr_v1",
                "lineup_platoon_ops_advantage": values["home"]["weighted_ops"] - values["away"]["weighted_ops"],
                "lineup_platoon_top4_ops_advantage": values["home"]["top4_ops"] - values["away"]["top4_ops"],
                "away_lineup_platoon_stats_missing_rate": values["away"]["missing_rate"],
                "home_lineup_platoon_stats_missing_rate": values["home"]["missing_rate"],
            }
        )
        augmented.append(row)
    return augmented


def add_neutral_platoon_performance_features(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row, **{name: 0.0 for name in PLATOON_PERFORMANCE_FEATURE_NAMES})
        for row in rows
    ]


def _neutralize_trend_and_platoon(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return add_neutral_probabilistic_reliever_features(
        add_neutral_schedule_context_features(
            add_neutral_run_strength_features(
                add_neutral_platoon_performance_features(
                    add_neutral_platoon_features(add_neutral_starter_trend_features(rows))
                )
            )
        )
    )


def add_run_strength_features_from_rows(
    rows: Iterable[Mapping[str, Any]],
    run_strength_rows: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        source = run_strength_rows.get(int(row["game_id"]), {})
        row.update({name: float(source.get(name, 0.0)) for name in RUN_STRENGTH_FEATURE_NAMES})
        augmented.append(row)
    return augmented


def add_neutral_interaction_features(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row, **{name: 0.0 for name in INTERACTION_FEATURE_NAMES}) for row in rows]


def add_interaction_features(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add a small, preregistered nonlinear block without future information."""

    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        elo = float(row["home_elo_minus_away"])
        average_expected_innings = (
            float(row["away_starter_expected_innings"])
            + float(row["home_starter_expected_innings"])
        ) / 2.0
        row.update(
            {
                "elo_signed_square": elo * abs(elo),
                "starter_kbb_expected_innings_interaction": (
                    float(row["starter_k_minus_bb_rate_difference"]) * average_expected_innings
                ),
                "starter_era_expected_innings_interaction": (
                    float(row["starter_earned_run_rate_advantage"]) * average_expected_innings
                ),
                "lineup_ops_expected_innings_interaction": (
                    float(row["lineup_ops_advantage"]) * average_expected_innings
                ),
                "starter_bullpen_availability_interaction": (
                    float(row["starter_expected_innings_advantage"])
                    * float(row["bullpen_core_unavailable_count_advantage"])
                ),
            }
        )
        augmented.append(row)
    return augmented


def _predict_season(
    rows: Sequence[Mapping[str, Any]],
    *,
    season: int,
    l2: float,
    training_years: int | None = None,
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    first_training_season = None if training_years is None else int(season) - int(training_years)
    training = [
        row
        for row in rows
        if int(row["season"]) < int(season)
        and (first_training_season is None or int(row["season"]) >= first_training_season)
    ]
    validation = [row for row in rows if int(row["season"]) == int(season)]
    if not training or not validation:
        raise ValueError(f"Insufficient rows for model-v3 season {season}.")
    model = _fit(training, l2=l2)
    probability = model.predict_proba(extract_feature_matrix(validation, MODEL_V3_FEATURE_SPECS))
    return validation, probability


def _predict_lda_season(
    rows: Sequence[Mapping[str, Any]],
    *,
    season: int,
    shrinkage: float,
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    training = [row for row in rows if int(row["season"]) < int(season)]
    validation = [row for row in rows if int(row["season"]) == int(season)]
    if not training or not validation:
        raise ValueError(f"Insufficient rows for model-v3 LDA season {season}.")
    matrix = extract_feature_matrix(training, MODEL_V3_FEATURE_SPECS)
    target = _target(training)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (matrix - mean) / scale
    negative = standardized[target == 0.0]
    positive = standardized[target == 1.0]
    if not len(negative) or not len(positive):
        raise ValueError(f"Both classes are required for model-v3 LDA season {season}.")
    negative_mean = negative.mean(axis=0)
    positive_mean = positive.mean(axis=0)
    centered = np.vstack((negative - negative_mean, positive - positive_mean))
    covariance = centered.T @ centered / max(len(centered) - 2, 1)
    diagonal = np.diag(np.diag(covariance))
    regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    regularized += np.eye(regularized.shape[0]) * 1e-6
    coefficients = np.linalg.solve(regularized, positive_mean - negative_mean)
    positive_rate = float(target.mean())
    intercept = (
        -0.5 * float((positive_mean + negative_mean) @ coefficients)
        + math.log(positive_rate / (1.0 - positive_rate))
    )
    validation_matrix = (extract_feature_matrix(validation, MODEL_V3_FEATURE_SPECS) - mean) / scale
    scores = intercept + validation_matrix @ coefficients
    probability = np.empty_like(scores, dtype=float)
    nonnegative = scores >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
    exponential = np.exp(scores[~nonnegative])
    probability[~nonnegative] = exponential / (1.0 + exponential)
    return validation, probability


def _candidate_ranking_key(key: str, candidates: Mapping[str, Any]) -> tuple[Any, ...]:
    candidate = candidates[key]
    season_accuracies = [
        float(evaluation["metrics"]["accuracy"])
        for evaluation in candidate["season_evaluations"].values()
    ]
    metrics = candidate["evaluation"]["metrics"]
    return (
        -min(season_accuracies),
        -float(metrics["accuracy"]),
        float(metrics["log_loss"]),
        float(metrics["brier_score"]),
        key,
    )


def _prediction_is_correct(row: Mapping[str, Any], column: str) -> bool:
    return (float(row[column]) >= 0.5) == (int(row["home_win"]) == 1)


def _holdout_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Holdout diagnostics require at least one row.")
    total = len(rows)
    correct = sum(_prediction_is_correct(row, "p_model_v3") for row in rows)
    required_for_sixty = math.ceil(0.60 * total)
    agreement = [
        row
        for row in rows
        if (float(row["p_model_v3"]) >= 0.5) == (float(row["p_elo"]) >= 0.5)
    ]
    disagreement = [row for row in rows if row not in agreement]

    def group_summary(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "row_count": len(group),
            "model_v3_accuracy": (
                sum(_prediction_is_correct(row, "p_model_v3") for row in group) / len(group)
                if group else None
            ),
            "elo_accuracy": (
                sum(_prediction_is_correct(row, "p_elo") for row in group) / len(group)
                if group else None
            ),
        }

    accuracy_at_confidence = []
    for threshold in (0.0, 0.025, 0.05, 0.10, 0.15):
        eligible = [
            row for row in rows
            if abs(float(row["p_model_v3"]) - 0.5) >= threshold
        ]
        accuracy_at_confidence.append(
            {
                "minimum_distance_from_fifty": threshold,
                "row_count": len(eligible),
                "coverage": len(eligible) / total,
                "accuracy": (
                    sum(_prediction_is_correct(row, "p_model_v3") for row in eligible) / len(eligible)
                    if eligible else None
                ),
            }
        )
    return {
        "row_count": total,
        "correct_predictions": correct,
        "required_correct_for_sixty_percent": required_for_sixty,
        "additional_correct_needed_for_sixty_percent": max(required_for_sixty - correct, 0),
        "elo_agreement": group_summary(agreement),
        "elo_disagreement": group_summary(disagreement),
        "accuracy_at_confidence": accuracy_at_confidence,
    }


def evaluate_model_v3_challenger(
    base_rows: Sequence[Mapping[str, Any]],
    v2_rows: Sequence[Mapping[str, Any]],
    augmented_variants: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    l2_values: Sequence[float] = DEFAULT_V2_L2_VALUES,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260721,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    augmented_variants = {
        mode: ensure_probabilistic_reliever_feature_values(
            ensure_schedule_context_feature_values(rows)
        )
        for mode, rows in augmented_variants.items()
    }
    candidate_configs = [
        (mode, float(l2), None)
        for mode in augmented_variants
        for l2 in l2_values
    ]
    recent_window_modes = {
        "starter_readiness_lineup_reliever",
        "starter_readiness_lineup_reliever_interactions",
    }
    candidate_configs.extend(
        (mode, float(l2), training_years)
        for mode in recent_window_modes
        if mode in augmented_variants
        for l2 in l2_values
        if float(l2) <= 0.1
        for training_years in RECENT_TRAINING_WINDOWS
    )
    candidate_rows: dict[tuple[str, float, int | None], list[dict[str, Any]]] = {
        config: [] for config in candidate_configs
    }
    for (mode, l2, training_years), output_rows in candidate_rows.items():
        for season in BACKTEST_SEASONS:
            validation, probability = _predict_season(
                augmented_variants[mode],
                season=season,
                l2=l2,
                training_years=training_years,
            )
            output_rows.extend(
                {
                    "game_id": int(row["game_id"]),
                    "season": season,
                    "home_win": int(row["home_win"]),
                    "probability": float(value),
                }
                for row, value in zip(validation, probability)
            )

    candidates: dict[str, Any] = {}
    candidate_prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for (mode, l2, training_years), rows in candidate_rows.items():
        window_label = "all" if training_years is None else f"{training_years}y"
        key = f"model_v3:{mode}:l2={l2:g}:window={window_label}"
        candidates[key] = {
            "model_type": "logistic",
            "feature_mode": mode,
            "l2": l2,
            "training_years": training_years,
            "evaluation": evaluate_prediction_set(
                [row["home_win"] for row in rows],
                [row["probability"] for row in rows],
            ),
            "season_evaluations": {
                str(season): evaluate_prediction_set(
                    [row["home_win"] for row in rows if int(row["season"]) == season],
                    [row["probability"] for row in rows if int(row["season"]) == season],
                )
                for season in BACKTEST_SEASONS
            },
        }
        candidate_prediction_rows[key] = rows

    for base_key, base_candidate in list(candidates.items()):
        if str(base_candidate["feature_mode"]) not in BLEND_FEATURE_MODES:
            continue
        base_rows_for_blend = candidate_prediction_rows[base_key]
        elo_by_season = {
            season: {
                int(row["game_id"]): float(row["elo_expected_home_win_probability"])
                for row in base_rows
                if int(row["season"]) == season
            }
            for season in BACKTEST_SEASONS
        }
        for model_weight in BLEND_WEIGHTS:
            blended_rows = [
                {
                    **row,
                    "probability": (
                        model_weight * float(row["probability"])
                        + (1.0 - model_weight)
                        * elo_by_season[int(row["season"])][int(row["game_id"])]
                    ),
                }
                for row in base_rows_for_blend
            ]
            key = f"model_v3:blend:model_weight={model_weight:g}:{base_key}"
            candidates[key] = {
                **base_candidate,
                "model_type": "blend",
                "base_model_type": "logistic",
                "model_weight": model_weight,
                "base_candidate": base_key,
                "evaluation": evaluate_prediction_set(
                    [row["home_win"] for row in blended_rows],
                    [row["probability"] for row in blended_rows],
                ),
                "season_evaluations": {
                    str(season): evaluate_prediction_set(
                        [row["home_win"] for row in blended_rows if int(row["season"]) == season],
                        [row["probability"] for row in blended_rows if int(row["season"]) == season],
                    )
                    for season in BACKTEST_SEASONS
                },
            }
    lda_modes = {
        "starter_readiness_lineup_reliever",
        "starter_readiness_lineup_reliever_interactions",
        "starter_readiness_lineup_reliever_context",
        "starter_readiness_lineup_reliever_context_interactions",
        "starter_readiness_lineup_reliever_trend_platoon",
        "starter_readiness_lineup_reliever_trend_platoon_interactions",
        "starter_readiness_lineup_reliever_platoon_ops",
        "starter_readiness_lineup_reliever_platoon_ops_interactions",
        "starter_readiness_lineup_reliever_platoon_full",
        "starter_readiness_lineup_reliever_platoon_full_interactions",
        "starter_readiness_lineup_reliever_run_strength",
        "starter_readiness_lineup_reliever_run_strength_interactions",
    }
    for mode in sorted(lda_modes.intersection(augmented_variants)):
        for shrinkage in LDA_SHRINKAGE_VALUES:
            rows: list[dict[str, Any]] = []
            for season in BACKTEST_SEASONS:
                validation, probability = _predict_lda_season(
                    augmented_variants[mode],
                    season=season,
                    shrinkage=shrinkage,
                )
                rows.extend(
                    {"season": season, "home_win": int(row["home_win"]), "probability": float(value)}
                    for row, value in zip(validation, probability)
                )
            key = f"model_v3:lda:{mode}:shrinkage={shrinkage:g}:window=all"
            candidates[key] = {
                "model_type": "lda",
                "feature_mode": mode,
                "shrinkage": shrinkage,
                "training_years": None,
                "evaluation": evaluate_prediction_set(
                    [row["home_win"] for row in rows],
                    [row["probability"] for row in rows],
                ),
                "season_evaluations": {
                    str(season): evaluate_prediction_set(
                        [row["home_win"] for row in rows if int(row["season"]) == season],
                        [row["probability"] for row in rows if int(row["season"]) == season],
                    )
                    for season in BACKTEST_SEASONS
                },
            }
    selected_key = min(
        candidates,
        key=lambda key: _candidate_ranking_key(key, candidates),
    )
    selected = candidates[selected_key]
    selected_rows = augmented_variants[str(selected["feature_mode"])]
    selected_model_type = str(selected["model_type"])
    selected_l2 = float(selected.get("l2", 0.0))
    selected_training_years = selected["training_years"]

    official_rows = generate_historical_audit_rows(
        base_rows,
        l2=0.01,
        seasons=(*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON),
    )
    official_by_game = {int(row["game_id"]): row for row in official_rows}
    prediction_rows: list[dict[str, Any]] = []
    for season in (*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON):
        if selected_model_type in {"logistic", "blend"}:
            validation, v3_probability = _predict_season(
                selected_rows,
                season=season,
                l2=selected_l2,
                training_years=selected_training_years,
            )
            if selected_model_type == "blend":
                v3_probability = blend_with_elo(
                    validation,
                    v3_probability,
                    model_weight=float(selected["model_weight"]),
                )
        else:
            validation, v3_probability = _predict_lda_season(
                selected_rows,
                season=season,
                shrinkage=float(selected["shrinkage"]),
            )
        v2_validation, v2_probability = _predict_v2_season(v2_rows, season=season, l2=0.01)
        if [int(row["game_id"]) for row in validation] != [int(row["game_id"]) for row in v2_validation]:
            raise ValueError(f"model-v2 and model-v3 validation rows differ for {season}.")
        for row, v2_value, v3_value in zip(validation, v2_probability, v3_probability):
            official = official_by_game[int(row["game_id"])]
            prediction_rows.append(
                {
                    "game_id": int(row["game_id"]),
                    "official_date": str(row["official_date"]),
                    "season": season,
                    "home_win": int(row["home_win"]),
                    "p_elo": float(official["p_elo"]),
                    "p_model_v1": float(official["p_logistic_platt"]),
                    "p_model_v2": float(v2_value),
                    "p_model_v3": float(v3_value),
                }
            )

    model_columns = (
        ("elo", "p_elo"),
        ("model_v1", "p_model_v1"),
        ("model_v2", "p_model_v2"),
        ("model_v3", "p_model_v3"),
    )
    season_results = {
        str(season): {
            name: evaluate_prediction_set(
                [row["home_win"] for row in prediction_rows if int(row["season"]) == season],
                [row[column] for row in prediction_rows if int(row["season"]) == season],
            )
            for name, column in model_columns
        }
        for season in (*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON)
    }
    holdout_rows = [row for row in prediction_rows if int(row["season"]) == DEVELOPMENT_HOLDOUT_SEASON]
    holdout = {
        name: evaluate_prediction_set(
            [row["home_win"] for row in holdout_rows],
            [row[column] for row in holdout_rows],
        )
        for name, column in model_columns
    }
    v3_metrics = holdout["model_v3"]["metrics"]
    v1_metrics = holdout["model_v1"]["metrics"]
    v2_metrics = holdout["model_v2"]["metrics"]
    elo_metrics = holdout["elo"]["metrics"]
    criteria = {
        "beats_model_v1_on_accuracy_log_loss_brier": (
            float(v3_metrics["accuracy"]) > float(v1_metrics["accuracy"])
            and float(v3_metrics["log_loss"]) < float(v1_metrics["log_loss"])
            and float(v3_metrics["brier_score"]) < float(v1_metrics["brier_score"])
        ),
        "beats_model_v2_on_accuracy": float(v3_metrics["accuracy"]) > float(v2_metrics["accuracy"]),
        "beats_elo_on_accuracy": float(v3_metrics["accuracy"]) > float(elo_metrics["accuracy"]),
        "reaches_sixty_percent_accuracy": float(v3_metrics["accuracy"]) >= 0.60,
    }
    paired = {
        baseline: {
            metric: paired_date_block_bootstrap(
                holdout_rows,
                challenger="p_model_v3",
                baseline=column,
                metric=metric,
                iterations=bootstrap_iterations,
                seed=seed + baseline_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(("error_rate", "log_loss", "brier_score"))
        }
        for baseline_index, (baseline, column) in enumerate(
            (("model_v1", "p_model_v1"), ("model_v2", "p_model_v2"), ("elo", "p_elo"))
        )
    }
    coverage_rows = [
        row for row in selected_rows
        if int(row["season"]) in {*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON}
    ]
    report = {
        "schema_version": "model-v3-retrospective-evaluation-v1",
        "selection_seasons": list(BACKTEST_SEASONS),
        "development_holdout_season": DEVELOPMENT_HOLDOUT_SEASON,
        "ranking_policy": [
            "minimum_season_accuracy_desc",
            "aggregate_accuracy_desc",
            "log_loss",
            "brier_score",
        ],
        "selected_candidate": selected_key,
        "candidates": candidates,
        "season_results": season_results,
        "development_holdout": holdout,
        "development_holdout_diagnostics": _holdout_diagnostics(holdout_rows),
        "paired_date_block_bootstrap": paired,
        "coverage": {
            "rows": len(coverage_rows),
            "away_readiness_missing_rate": float(np.mean([row["away_starter_readiness_missing"] for row in coverage_rows])),
            "home_readiness_missing_rate": float(np.mean([row["home_starter_readiness_missing"] for row in coverage_rows])),
        },
        "score_gate_criteria": criteria,
        "retrospective_score_gate_passed": all(criteria.values()),
        "promotion_allowed": False,
        "promotion_blocker": "Historical starter and lineup identities lack pregame observation timestamps; prospective model-v3 snapshots are required.",
    }
    return report, prediction_rows


def run_model_v3_backtest(
    *,
    feature_rows: Iterable[Mapping[str, Any]],
    completed_games: Iterable[Mapping[str, Any]],
    client: MlbStatsApiClient,
    output_dir: str | Path,
    refresh: bool = False,
    l2_values: Sequence[float] = DEFAULT_V2_L2_VALUES,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260721,
) -> dict[str, Any]:
    rows = sorted((dict(row) for row in feature_rows), key=lambda row: (str(row["official_date"]), int(row["game_id"])))
    games = [dict(game) for game in completed_games]
    run_strength_materialized = add_dynamic_run_strength_features(rows, games)
    run_strength_by_game = {int(row["game_id"]): row for row in run_strength_materialized}
    target_seasons = sorted({int(row["season"]) for row in rows if int(row["season"]) <= DEVELOPMENT_HOLDOUT_SEASON})
    prior_stats, sources = collect_prior_season_pitching_stats(
        client=client,
        rows=rows,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    game_logs, game_log_sources = collect_current_season_pitching_game_logs(
        client=client,
        rows=rows,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    team_logs, team_log_sources = collect_team_pitching_game_logs(
        client=client,
        rows=rows,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    sources.extend(game_log_sources)
    sources.extend(team_log_sources)

    lineups, lineup_sources = collect_historical_lineups(
        client=client,
        rows=rows,
        refresh=refresh,
    )
    batting_stats, batting_stats_sources = collect_prior_season_batting_stats(
        client=client,
        rows=rows,
        lineups=lineups,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    batting_logs, batting_log_sources = collect_current_season_batting_game_logs(
        client=client,
        rows=rows,
        lineups=lineups,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    batting_platoon_stats, batting_platoon_sources = collect_prior_season_batting_platoon_stats(
        client=client,
        rows=rows,
        lineups=lineups,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    sources.extend(lineup_sources)
    sources.extend(batting_stats_sources)
    sources.extend(batting_log_sources)
    sources.extend(batting_platoon_sources)

    pitcher_rosters, roster_sources = collect_full_season_pitcher_rosters(
        client=client,
        rows=rows,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    reliever_prior_stats, reliever_game_logs, reliever_sources = collect_reliever_pitching_data(
        client=client,
        rosters=pitcher_rosters,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    sources.extend(roster_sources)
    sources.extend(reliever_sources)

    neutral_context_rows = add_neutral_context_features(rows)
    rolling = add_rolling_starter_features(neutral_context_rows, prior_stats, game_logs)
    readiness = add_starter_readiness_features(rolling, prior_stats, game_logs)
    bullpen = add_bullpen_workload_features(rolling, game_logs, team_logs)
    readiness_bullpen = add_starter_readiness_features(bullpen, prior_stats, game_logs)
    readiness_lineup = add_lineup_features(readiness, lineups, batting_stats, batting_logs)
    readiness_lineup_bullpen = add_lineup_features(readiness_bullpen, lineups, batting_stats, batting_logs)
    readiness_lineup_reliever = add_reliever_availability_features(
        readiness_lineup,
        pitcher_rosters,
        reliever_prior_stats,
        reliever_game_logs,
    )
    readiness_lineup_reliever_bullpen = add_reliever_availability_features(
        readiness_lineup_bullpen,
        pitcher_rosters,
        reliever_prior_stats,
        reliever_game_logs,
    )
    readiness_lineup_reliever_context = add_context_features(readiness_lineup_reliever)
    readiness_lineup_reliever_trend = add_neutral_run_strength_features(
        add_recent_starter_form_features(
            add_neutral_platoon_performance_features(
                add_neutral_platoon_features(readiness_lineup_reliever)
            ),
            game_logs,
        ),
    )
    readiness_lineup_reliever_platoon = add_neutral_run_strength_features(
        add_neutral_platoon_performance_features(
            add_platoon_features(
                add_neutral_starter_trend_features(readiness_lineup_reliever),
                lineups,
                batting_stats,
                prior_stats,
            )
        )
    )
    readiness_lineup_reliever_trend_platoon = add_neutral_run_strength_features(
        add_neutral_platoon_performance_features(
            add_platoon_features(
                add_recent_starter_form_features(readiness_lineup_reliever, game_logs),
                lineups,
                batting_stats,
                prior_stats,
            )
        )
    )
    readiness_lineup_reliever_platoon_ops = add_neutral_run_strength_features(
        add_platoon_performance_features(
            add_neutral_platoon_features(add_neutral_starter_trend_features(readiness_lineup_reliever)),
            lineups,
            batting_stats,
            batting_platoon_stats,
            prior_stats,
        )
    )
    readiness_lineup_reliever_platoon_full = add_platoon_performance_features(
        readiness_lineup_reliever_platoon,
        lineups,
        batting_stats,
        batting_platoon_stats,
        prior_stats,
    )
    readiness_lineup_reliever_run_strength = add_run_strength_features_from_rows(
        add_neutral_platoon_performance_features(
            add_neutral_platoon_features(add_neutral_starter_trend_features(readiness_lineup_reliever))
        ),
        run_strength_by_game,
    )
    readiness_lineup_reliever_schedule = add_schedule_context_features(
        _neutralize_trend_and_platoon(readiness_lineup_reliever),
        games,
    )
    readiness_lineup_reliever_probabilistic = add_probabilistic_reliever_features(
        add_neutral_schedule_context_features(
            add_neutral_run_strength_features(
                add_neutral_platoon_performance_features(
                    add_neutral_platoon_features(
                        add_neutral_starter_trend_features(readiness_lineup_reliever)
                    )
                )
            )
        ),
        pitcher_rosters,
        reliever_prior_stats,
        reliever_game_logs,
    )
    readiness_lineup_reliever_schedule_probabilistic = add_schedule_context_features(
        readiness_lineup_reliever_probabilistic,
        games,
    )
    readiness_lineup_reliever_venue = select_schedule_context_feature_block(
        readiness_lineup_reliever_schedule,
        VENUE_CONTEXT_FEATURE_NAMES,
    )
    readiness_lineup_reliever_fatigue = select_schedule_context_feature_block(
        readiness_lineup_reliever_schedule,
        FATIGUE_CONTEXT_FEATURE_NAMES,
    )
    report, predictions = evaluate_model_v3_challenger(
        rows,
        rolling,
        {
            "starter_readiness": add_neutral_interaction_features(
                _neutralize_trend_and_platoon(
                    add_neutral_reliever_features(add_neutral_lineup_features(readiness))
                )
            ),
            "starter_readiness_bullpen": add_neutral_interaction_features(
                _neutralize_trend_and_platoon(
                    add_neutral_reliever_features(add_neutral_lineup_features(readiness_bullpen))
                )
            ),
            "starter_readiness_lineup": add_neutral_interaction_features(
                _neutralize_trend_and_platoon(add_neutral_reliever_features(readiness_lineup))
            ),
            "starter_readiness_lineup_bullpen": add_neutral_interaction_features(
                _neutralize_trend_and_platoon(add_neutral_reliever_features(readiness_lineup_bullpen))
            ),
            "starter_readiness_lineup_reliever": add_neutral_interaction_features(
                _neutralize_trend_and_platoon(readiness_lineup_reliever)
            ),
            "starter_readiness_lineup_reliever_bullpen": add_neutral_interaction_features(
                _neutralize_trend_and_platoon(readiness_lineup_reliever_bullpen)
            ),
            "starter_readiness_lineup_reliever_interactions": add_interaction_features(
                _neutralize_trend_and_platoon(readiness_lineup_reliever)
            ),
            "starter_readiness_lineup_reliever_bullpen_interactions": add_interaction_features(
                _neutralize_trend_and_platoon(readiness_lineup_reliever_bullpen)
            ),
            "starter_readiness_lineup_reliever_context": add_neutral_interaction_features(
                _neutralize_trend_and_platoon(readiness_lineup_reliever_context)
            ),
            "starter_readiness_lineup_reliever_context_interactions": add_interaction_features(
                _neutralize_trend_and_platoon(readiness_lineup_reliever_context)
            ),
            "starter_readiness_lineup_reliever_trend": add_neutral_interaction_features(
                readiness_lineup_reliever_trend
            ),
            "starter_readiness_lineup_reliever_trend_interactions": add_interaction_features(
                readiness_lineup_reliever_trend
            ),
            "starter_readiness_lineup_reliever_platoon": add_neutral_interaction_features(
                readiness_lineup_reliever_platoon
            ),
            "starter_readiness_lineup_reliever_platoon_interactions": add_interaction_features(
                readiness_lineup_reliever_platoon
            ),
            "starter_readiness_lineup_reliever_trend_platoon": add_neutral_interaction_features(
                readiness_lineup_reliever_trend_platoon
            ),
            "starter_readiness_lineup_reliever_trend_platoon_interactions": add_interaction_features(
                readiness_lineup_reliever_trend_platoon
            ),
            "starter_readiness_lineup_reliever_platoon_ops": add_neutral_interaction_features(
                readiness_lineup_reliever_platoon_ops
            ),
            "starter_readiness_lineup_reliever_platoon_ops_interactions": add_interaction_features(
                readiness_lineup_reliever_platoon_ops
            ),
            "starter_readiness_lineup_reliever_platoon_full": add_neutral_interaction_features(
                readiness_lineup_reliever_platoon_full
            ),
            "starter_readiness_lineup_reliever_platoon_full_interactions": add_interaction_features(
                readiness_lineup_reliever_platoon_full
            ),
            "starter_readiness_lineup_reliever_run_strength": add_neutral_interaction_features(
                readiness_lineup_reliever_run_strength
            ),
            "starter_readiness_lineup_reliever_run_strength_interactions": add_interaction_features(
                readiness_lineup_reliever_run_strength
            ),
            "starter_readiness_lineup_reliever_schedule_interactions": add_interaction_features(
                readiness_lineup_reliever_schedule
            ),
            "starter_readiness_lineup_reliever_schedule_full_interactions": add_schedule_context_interactions(
                add_interaction_features(readiness_lineup_reliever_schedule)
            ),
            "starter_readiness_lineup_reliever_venue_interactions": add_interaction_features(
                readiness_lineup_reliever_venue
            ),
            "starter_readiness_lineup_reliever_fatigue_interactions": add_interaction_features(
                readiness_lineup_reliever_fatigue
            ),
            "starter_readiness_lineup_reliever_probabilistic_interactions": add_interaction_features(
                readiness_lineup_reliever_probabilistic
            ),
            "starter_readiness_lineup_reliever_schedule_probabilistic_interactions": add_interaction_features(
                readiness_lineup_reliever_schedule_probabilistic
            ),
        },
        l2_values=l2_values,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "evaluation.json", report)
    write_rows_csv(destination / "predictions.csv", predictions)
    manifest = {
        "schema_version": "model-v3-retrospective-manifest-v1",
        "evaluation": "evaluation.json",
        "predictions": "predictions.csv",
        "source_count": len(sources),
        "sources": sources,
        "promotion_allowed": False,
    }
    write_json(destination / "manifest.json", manifest)
    return {"report": report, "manifest": manifest}
