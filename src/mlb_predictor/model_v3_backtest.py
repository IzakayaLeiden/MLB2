from __future__ import annotations

from datetime import date
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audit import generate_historical_audit_rows, paired_date_block_bootstrap
from .bullpen_backtest import (
    RELIEVER_FEATURE_NAMES,
    add_neutral_reliever_features,
    add_reliever_availability_features,
    collect_full_season_pitcher_rosters,
    collect_reliever_pitching_data,
)
from .collector import MlbStatsApiClient
from .evaluation import evaluate_prediction_set
from .io import write_json, write_rows_csv
from .lineup import (
    LINEUP_FEATURE_NAMES,
    add_lineup_features,
    add_neutral_lineup_features,
    collect_current_season_batting_game_logs,
    collect_historical_lineups,
    collect_prior_season_batting_stats,
)
from .modeling import DEFAULT_FEATURE_SPECS, FeatureSpec, StandardizedLogisticRegression, extract_feature_matrix
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


DEFAULT_EXPECTED_START_OUTS = 15.0
DEFAULT_STARTER_REST_DAYS = 5
PRIOR_START_WEIGHT = 3.0
RECENT_START_WINDOW = 5
RECENT_TRAINING_WINDOWS = (3, 4, 5)
READINESS_FEATURE_NAMES = (
    "starter_rest_days_difference",
    "starter_expected_innings_advantage",
    "away_starter_readiness_missing",
    "home_starter_readiness_missing",
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
        *READINESS_FEATURE_NAMES,
        *LINEUP_FEATURE_NAMES,
        *RELIEVER_FEATURE_NAMES,
        *INTERACTION_FEATURE_NAMES,
    )
)


def _target(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["home_win"]) for row in rows], dtype=float)


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
                {"season": season, "home_win": int(row["home_win"]), "probability": float(value)}
                for row, value in zip(validation, probability)
            )

    candidates: dict[str, Any] = {}
    for (mode, l2, training_years), rows in candidate_rows.items():
        window_label = "all" if training_years is None else f"{training_years}y"
        key = f"model_v3:{mode}:l2={l2:g}:window={window_label}"
        candidates[key] = {
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
    selected_key = min(
        candidates,
        key=lambda key: _candidate_ranking_key(key, candidates),
    )
    selected = candidates[selected_key]
    selected_rows = augmented_variants[str(selected["feature_mode"])]
    selected_l2 = float(selected["l2"])
    selected_training_years = selected["training_years"]

    official_rows = generate_historical_audit_rows(
        base_rows,
        l2=0.01,
        seasons=(*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON),
    )
    official_by_game = {int(row["game_id"]): row for row in official_rows}
    prediction_rows: list[dict[str, Any]] = []
    for season in (*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON):
        validation, v3_probability = _predict_season(
            selected_rows,
            season=season,
            l2=selected_l2,
            training_years=selected_training_years,
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
    client: MlbStatsApiClient,
    output_dir: str | Path,
    refresh: bool = False,
    l2_values: Sequence[float] = DEFAULT_V2_L2_VALUES,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260721,
) -> dict[str, Any]:
    rows = sorted((dict(row) for row in feature_rows), key=lambda row: (str(row["official_date"]), int(row["game_id"])))
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
    sources.extend(lineup_sources)
    sources.extend(batting_stats_sources)
    sources.extend(batting_log_sources)

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

    rolling = add_rolling_starter_features(rows, prior_stats, game_logs)
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
    report, predictions = evaluate_model_v3_challenger(
        rows,
        rolling,
        {
            "starter_readiness": add_neutral_interaction_features(
                add_neutral_reliever_features(add_neutral_lineup_features(readiness))
            ),
            "starter_readiness_bullpen": add_neutral_interaction_features(
                add_neutral_reliever_features(add_neutral_lineup_features(readiness_bullpen))
            ),
            "starter_readiness_lineup": add_neutral_interaction_features(
                add_neutral_reliever_features(readiness_lineup)
            ),
            "starter_readiness_lineup_bullpen": add_neutral_interaction_features(
                add_neutral_reliever_features(readiness_lineup_bullpen)
            ),
            "starter_readiness_lineup_reliever": add_neutral_interaction_features(
                readiness_lineup_reliever
            ),
            "starter_readiness_lineup_reliever_bullpen": add_neutral_interaction_features(
                readiness_lineup_reliever_bullpen
            ),
            "starter_readiness_lineup_reliever_interactions": add_interaction_features(
                readiness_lineup_reliever
            ),
            "starter_readiness_lineup_reliever_bullpen_interactions": add_interaction_features(
                readiness_lineup_reliever_bullpen
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
