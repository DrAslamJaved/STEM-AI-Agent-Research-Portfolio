"""Tests for raw time-series data loading."""

from pathlib import Path

import pandas as pd
import pytest

from time_series_agent.config import DataConfig
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.exceptions import (
    DataParsingError,
    DatasetNotFoundError,
    MissingColumnsError,
)


def make_config(csv_path: Path) -> DataConfig:
    """Create a small test configuration."""
    return DataConfig(
        raw_file_path=csv_path,
        encoding="utf-8",
        date_column="Date",
        hour_column="Hour",
        timestamp_column="timestamp",
        target_column="target",
        date_format="%d/%m/%Y",
        expected_frequency="h",
    )


def write_test_csv(
    csv_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write a temporary CSV fixture."""
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def valid_rows() -> list[dict[str, object]]:
    """Return a small valid hourly dataset."""
    return [
        {"Date": "01/12/2017", "Hour": 0, "target": 10},
        {"Date": "01/12/2017", "Hour": 1, "target": 12},
        {"Date": "01/12/2017", "Hour": 2, "target": 15},
    ]


def test_load_valid_csv_constructs_timestamp(
    tmp_path: Path,
) -> None:
    """Valid raw values should produce hourly timestamps."""
    csv_path = tmp_path / "valid.csv"
    write_test_csv(csv_path, valid_rows())

    loaded = load_time_series_csv(make_config(csv_path))

    assert len(loaded) == 3
    assert "timestamp" in loaded.columns
    assert loaded["timestamp"].tolist() == [
        pd.Timestamp("2017-12-01 00:00:00"),
        pd.Timestamp("2017-12-01 01:00:00"),
        pd.Timestamp("2017-12-01 02:00:00"),
    ]


def test_loader_preserves_original_row_order(
    tmp_path: Path,
) -> None:
    """Loading should not silently sort disordered rows."""
    csv_path = tmp_path / "unsorted.csv"
    rows = [
        {"Date": "01/12/2017", "Hour": 2, "target": 15},
        {"Date": "01/12/2017", "Hour": 0, "target": 10},
        {"Date": "01/12/2017", "Hour": 1, "target": 12},
    ]
    write_test_csv(csv_path, rows)

    loaded = load_time_series_csv(make_config(csv_path))

    assert loaded["Hour"].tolist() == [2, 0, 1]


def test_missing_csv_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A missing dataset should raise DatasetNotFoundError."""
    csv_path = tmp_path / "missing.csv"

    with pytest.raises(
        DatasetNotFoundError,
        match="does not exist",
    ):
        load_time_series_csv(make_config(csv_path))


def test_missing_required_column_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A dataset lacking its target should be rejected."""
    csv_path = tmp_path / "missing_column.csv"
    rows = [
        {"Date": "01/12/2017", "Hour": 0},
    ]
    write_test_csv(csv_path, rows)

    with pytest.raises(
        MissingColumnsError,
        match="target",
    ):
        load_time_series_csv(make_config(csv_path))


def test_invalid_date_raises_clear_error(
    tmp_path: Path,
) -> None:
    """An invalid date should produce DataParsingError."""
    csv_path = tmp_path / "bad_date.csv"
    rows = valid_rows()
    rows[1]["Date"] = "not-a-date"
    write_test_csv(csv_path, rows)

    with pytest.raises(
        DataParsingError,
        match="invalid or missing date",
    ):
        load_time_series_csv(make_config(csv_path))


def test_nonnumeric_hour_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A nonnumeric hour should produce DataParsingError."""
    csv_path = tmp_path / "bad_hour.csv"
    rows = valid_rows()
    rows[1]["Hour"] = "morning"
    write_test_csv(csv_path, rows)

    with pytest.raises(
        DataParsingError,
        match="invalid or missing hour",
    ):
        load_time_series_csv(make_config(csv_path))


def test_noninteger_hour_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A fractional hour should be rejected."""
    csv_path = tmp_path / "fractional_hour.csv"
    rows = valid_rows()
    rows[1]["Hour"] = 1.5
    write_test_csv(csv_path, rows)

    with pytest.raises(
        DataParsingError,
        match="noninteger hour",
    ):
        load_time_series_csv(make_config(csv_path))


def test_out_of_range_hour_raises_clear_error(
    tmp_path: Path,
) -> None:
    """An hour outside 0 through 23 should be rejected."""
    csv_path = tmp_path / "bad_hour_range.csv"
    rows = valid_rows()
    rows[1]["Hour"] = 24
    write_test_csv(csv_path, rows)

    with pytest.raises(
        DataParsingError,
        match="outside the range 0-23",
    ):
        load_time_series_csv(make_config(csv_path))


def test_invalid_target_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A nonnumeric target should produce DataParsingError."""
    csv_path = tmp_path / "bad_target.csv"
    rows = valid_rows()
    rows[1]["target"] = "unknown"
    write_test_csv(csv_path, rows)

    with pytest.raises(
        DataParsingError,
        match="invalid or missing numeric",
    ):
        load_time_series_csv(make_config(csv_path))


def test_missing_target_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A missing target value should be rejected."""
    csv_path = tmp_path / "missing_target.csv"
    rows = valid_rows()
    rows[1]["target"] = None
    write_test_csv(csv_path, rows)

    with pytest.raises(
        DataParsingError,
        match="invalid or missing numeric",
    ):
        load_time_series_csv(make_config(csv_path))