from __future__ import annotations

import csv
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


_INTEGER_COLUMNS = {
    "season",
    "home_win",
    "game_number",
    "scheduled_innings",
    "home_rest_days",
    "away_rest_days",
    "rest_days_difference",
}


def _should_be_nullable_integer(column: str) -> bool:
    return (
        column in _INTEGER_COLUMNS
        or column.endswith("_id")
        or column.endswith("_score")
        or column.endswith("_games_before")
        or column.endswith("_games_count")
        or column.endswith("_has_prior_history")
    )


def _python_value(column: str, value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if _should_be_nullable_integer(column) and isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_rows_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        output.write_text("", encoding="utf-8")
        return output
    columns = list(dict.fromkeys(key for row in materialized for key in row.keys()))
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return output


def write_rows_parquet(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    for column in frame.columns:
        if _should_be_nullable_integer(str(column)):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
    frame.to_parquet(output, index=False)
    return output


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("지원 형식은 .csv 또는 .parquet입니다.")
    return [
        {column: _python_value(str(column), value) for column, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
