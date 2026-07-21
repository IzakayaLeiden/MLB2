from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audit import generate_historical_audit_rows, paired_date_block_bootstrap
from .collector import CachedPayload, MlbStatsApiClient
from .evaluation import evaluate_prediction_set
from .io import write_json, write_rows_csv
from .modeling import DEFAULT_FEATURE_SPECS, FeatureSpec, StandardizedLogisticRegression, extract_feature_matrix
from .pitching import PITCHER_RATE_PRIORS, normalize_pitcher_stats_payload, smoothed_pitcher_rate


BACKTEST_SEASONS = (2022, 2023, 2024)
DEVELOPMENT_HOLDOUT_SEASON = 2025
DEFAULT_V2_L2_VALUES = (0.01, 0.1, 1.0, 10.0)

STARTER_FEATURE_NAMES = (
    "starter_k_minus_bb_rate_difference",
    "starter_earned_run_rate_advantage",
    "starter_home_run_rate_advantage",
    "away_starter_history_missing",
    "home_starter_history_missing",
)
STARTER_CHALLENGER_FEATURE_SPECS = DEFAULT_FEATURE_SPECS + tuple(
    FeatureSpec(name=name, source=name, transform="numeric", value=None)
    for name in STARTER_FEATURE_NAMES
)


def _target(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["home_win"]) for row in rows], dtype=float)


def _fit(
    rows: Sequence[Mapping[str, Any]],
    *,
    l2: float,
) -> StandardizedLogisticRegression:
    return StandardizedLogisticRegression(l2=l2).fit(
        extract_feature_matrix(rows, STARTER_CHALLENGER_FEATURE_SPECS),
        _target(rows),
        feature_names=[spec.name for spec in STARTER_CHALLENGER_FEATURE_SPECS],
    )


def collect_prior_season_pitching_stats(
    *,
    client: MlbStatsApiClient,
    rows: Sequence[Mapping[str, Any]],
    target_seasons: Sequence[int],
    refresh: bool = False,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    stats_by_season_player: dict[tuple[int, int], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for target_season in target_seasons:
        ids = sorted(
            {
                int(pitcher_id)
                for row in rows
                if int(row["season"]) == int(target_season)
                for pitcher_id in (row.get("away_probable_pitcher_id"), row.get("home_probable_pitcher_id"))
                if pitcher_id is not None
            }
        )
        stats_season = int(target_season) - 1
        responses = client.fetch_people_pitching_season_stats(ids, stats_season, refresh=refresh)
        for response in responses:
            sources.append(
                {
                    "stats_season": stats_season,
                    "cache_path": str(response.cache_path),
                    "fetched_at_utc": response.fetched_at_utc,
                    "source_url": response.source_url,
                    "response_sha256": response.response_sha256,
                }
            )
            for person in response.payload.get("people", []):
                if not isinstance(person, Mapping) or not person.get("id"):
                    continue
                stats_by_season_player[(stats_season, int(person["id"]))] = normalize_pitcher_stats_payload(
                    {"stats": person.get("stats", [])}
                )
    return stats_by_season_player, sources


def add_prior_season_starter_features(
    rows: Iterable[Mapping[str, Any]],
    stats_by_season_player: Mapping[tuple[int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        stats_season = int(row["season"]) - 1
        rates: dict[str, dict[str, float]] = {}
        missing: dict[str, int] = {}
        for side in ("away", "home"):
            pitcher_id = row.get(f"{side}_probable_pitcher_id")
            stats = stats_by_season_player.get((stats_season, int(pitcher_id))) if pitcher_id is not None else None
            materialized = dict(stats) if stats is not None else {"has_history": False, "batters_faced": 0}
            missing[side] = int(not materialized.get("has_history"))
            rates[side] = {
                field: smoothed_pitcher_rate(materialized, field)
                for field in PITCHER_RATE_PRIORS
            }
        row.update(
            {
                "pitching_feature_provenance": "retrospective_prior_season_stats_v1",
                "pitching_stats_season": stats_season,
                "starter_identity_point_in_time_verified": False,
                "starter_k_minus_bb_rate_difference": (
                    rates["home"]["strikeouts"] - rates["home"]["walks"]
                    - rates["away"]["strikeouts"] + rates["away"]["walks"]
                ),
                "starter_earned_run_rate_advantage": rates["away"]["earned_runs"] - rates["home"]["earned_runs"],
                "starter_home_run_rate_advantage": rates["away"]["home_runs"] - rates["home"]["home_runs"],
                "away_starter_history_missing": missing["away"],
                "home_starter_history_missing": missing["home"],
            }
        )
        augmented.append(row)
    return augmented


def _predict_season(
    rows: Sequence[Mapping[str, Any]],
    *,
    season: int,
    l2: float,
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    training = [row for row in rows if int(row["season"]) < int(season)]
    validation = [row for row in rows if int(row["season"]) == int(season)]
    if not training or not validation:
        raise ValueError(f"Insufficient rows for starter challenger season {season}.")
    model = _fit(training, l2=l2)
    probability = model.predict_proba(extract_feature_matrix(validation, STARTER_CHALLENGER_FEATURE_SPECS))
    return validation, probability


def evaluate_retrospective_starter_challenger(
    base_rows: Sequence[Mapping[str, Any]],
    augmented_rows: Sequence[Mapping[str, Any]],
    *,
    l2_values: Sequence[float] = DEFAULT_V2_L2_VALUES,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260721,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_rows: dict[float, list[dict[str, Any]]] = {float(l2): [] for l2 in l2_values}
    for l2 in candidate_rows:
        for season in BACKTEST_SEASONS:
            validation, probability = _predict_season(augmented_rows, season=season, l2=l2)
            candidate_rows[l2].extend(
                {"home_win": int(row["home_win"]), "probability": float(value)}
                for row, value in zip(validation, probability)
            )
    candidates: dict[str, Any] = {}
    for l2, rows in candidate_rows.items():
        candidates[f"starter_logistic:l2={l2:g}"] = {
            "l2": l2,
            "evaluation": evaluate_prediction_set(
                [row["home_win"] for row in rows],
                [row["probability"] for row in rows],
            ),
        }
    selected_key = min(
        candidates,
        key=lambda key: (
            -float(candidates[key]["evaluation"]["metrics"]["accuracy"]),
            float(candidates[key]["evaluation"]["metrics"]["log_loss"]),
            float(candidates[key]["evaluation"]["metrics"]["brier_score"]),
            key,
        ),
    )
    selected_l2 = float(candidates[selected_key]["l2"])

    official_rows = generate_historical_audit_rows(base_rows, l2=0.01, seasons=(*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON))
    official_by_game = {int(row["game_id"]): row for row in official_rows}
    prediction_rows: list[dict[str, Any]] = []
    for season in (*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON):
        validation, probability = _predict_season(augmented_rows, season=season, l2=selected_l2)
        for row, value in zip(validation, probability):
            official = official_by_game[int(row["game_id"])]
            prediction_rows.append(
                {
                    "game_id": int(row["game_id"]),
                    "official_date": str(row["official_date"]),
                    "season": int(season),
                    "home_win": int(row["home_win"]),
                    "p_elo": float(official["p_elo"]),
                    "p_model_v1_raw": float(official["p_logistic_raw"]),
                    "p_model_v1": float(official["p_logistic_platt"]),
                    "p_starter_challenger": float(value),
                }
            )

    holdout_rows = [row for row in prediction_rows if int(row["season"]) == DEVELOPMENT_HOLDOUT_SEASON]
    season_results = {
        str(season): {
            name: evaluate_prediction_set(
                [row["home_win"] for row in prediction_rows if int(row["season"]) == season],
                [row[column] for row in prediction_rows if int(row["season"]) == season],
            )
            for name, column in (
                ("elo", "p_elo"),
                ("model_v1_raw", "p_model_v1_raw"),
                ("model_v1", "p_model_v1"),
                ("starter_challenger", "p_starter_challenger"),
            )
        }
        for season in (*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON)
    }
    evaluations = {
        name: evaluate_prediction_set(
            [row["home_win"] for row in holdout_rows],
            [row[column] for row in holdout_rows],
        )
        for name, column in (
            ("elo", "p_elo"),
            ("model_v1_raw", "p_model_v1_raw"),
            ("model_v1", "p_model_v1"),
            ("starter_challenger", "p_starter_challenger"),
        )
    }
    challenger_metrics = evaluations["starter_challenger"]["metrics"]
    champion_metrics = evaluations["model_v1"]["metrics"]
    elo_metrics = evaluations["elo"]["metrics"]
    beats_model_v1_all_three = (
        float(challenger_metrics["accuracy"]) > float(champion_metrics["accuracy"])
        and float(challenger_metrics["log_loss"]) < float(champion_metrics["log_loss"])
        and float(challenger_metrics["brier_score"]) < float(champion_metrics["brier_score"])
    )
    beats_elo_accuracy = float(challenger_metrics["accuracy"]) > float(elo_metrics["accuracy"])
    retrospective_passed = beats_model_v1_all_three and beats_elo_accuracy
    paired = {
        metric: paired_date_block_bootstrap(
            holdout_rows,
            challenger="p_starter_challenger",
            baseline="p_model_v1",
            metric=metric,
            iterations=bootstrap_iterations,
            seed=seed + index,
        )
        for index, metric in enumerate(("error_rate", "log_loss", "brier_score"))
    }
    missing_rows = [row for row in augmented_rows if int(row["season"]) in {*BACKTEST_SEASONS, DEVELOPMENT_HOLDOUT_SEASON}]
    report = {
        "schema_version": "pitching-v2-retrospective-evaluation-v1",
        "provenance": {
            "starter_stats": "previous_regular_season_only",
            "starter_identity": "historical_schedule_probable_pitcher_posthoc",
            "starter_identity_point_in_time_verified": False,
            "bullpen_features_included": False,
        },
        "ranking_policy": ["accuracy_desc", "log_loss", "brier_score"],
        "selection_seasons": list(BACKTEST_SEASONS),
        "development_holdout_season": DEVELOPMENT_HOLDOUT_SEASON,
        "candidates": candidates,
        "selected_candidate": selected_key,
        "season_results": season_results,
        "development_holdout": evaluations,
        "paired_date_block_bootstrap_vs_model_v1": paired,
        "coverage": {
            "rows": len(missing_rows),
            "away_prior_season_missing_rate": float(np.mean([row["away_starter_history_missing"] for row in missing_rows])),
            "home_prior_season_missing_rate": float(np.mean([row["home_starter_history_missing"] for row in missing_rows])),
        },
        "score_gate_criteria": {
            "beats_model_v1_on_accuracy_log_loss_brier": beats_model_v1_all_three,
            "beats_elo_on_accuracy": beats_elo_accuracy,
        },
        "retrospective_score_gate_passed": retrospective_passed,
        "promotion_allowed": False,
        "promotion_blocker": "Historical starter identity lacks a pregame observation timestamp; prospective snapshot-v2 evidence is required.",
    }
    return report, prediction_rows


def run_retrospective_pitching_backtest(
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
    stats, sources = collect_prior_season_pitching_stats(
        client=client,
        rows=rows,
        target_seasons=target_seasons,
        refresh=refresh,
    )
    augmented = add_prior_season_starter_features(rows, stats)
    report, predictions = evaluate_retrospective_starter_challenger(
        rows,
        augmented,
        l2_values=l2_values,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "evaluation.json", report)
    write_rows_csv(destination / "predictions.csv", predictions)
    manifest = {
        "schema_version": "pitching-v2-retrospective-manifest-v1",
        "evaluation": "evaluation.json",
        "predictions": "predictions.csv",
        "source_count": len(sources),
        "sources": sources,
        "promotion_allowed": False,
    }
    write_json(destination / "manifest.json", manifest)
    return {"report": report, "manifest": manifest}
