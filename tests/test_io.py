from __future__ import annotations

import pyarrow.parquet as pq

from mlb_predictor.io import read_rows, write_rows_parquet
from mlb_predictor.quality import validate_raw_games


def test_nullable_integer_ids_survive_parquet_roundtrip(tmp_path, game_factory) -> None:
    first = game_factory(900, "2025-08-01", 1, 2, 3, 2)
    second = game_factory(901, "2025-08-02", 3, 4, 1, 0)
    first["home_probable_pitcher_id"] = None
    path = write_rows_parquet(tmp_path / "games.parquet", [first, second])

    schema = pq.read_schema(path)
    rows = read_rows(path)

    assert str(schema.field("home_probable_pitcher_id").type) == "int64"
    assert rows[0]["home_probable_pitcher_id"] is None
    assert rows[1]["home_probable_pitcher_id"] == 20_003
    assert isinstance(rows[1]["home_probable_pitcher_id"], int)
    report = validate_raw_games(rows)
    assert report.metrics["probable_pitcher_missing_rows"] == 1
    assert report.metrics["probable_pitcher_missing_rate"] == 0.5
