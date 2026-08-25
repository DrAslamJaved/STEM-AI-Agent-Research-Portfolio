"""Tests for time-series validation and reporting."""

import json
from pathlib import Path

import pandas as pd
import pytest

from time_series_agent.config import DataConfig
from time_series_agent.exceptions import MissingColumnsError
from time_series_agent.validation import (
    save_validation_report,
    validate_time_series,
)


def make_config(expected_rows: int | None = 3) -> DataConfig:
    """Create configuration for small validation examples."""
    return DataConfig(
        raw_file_path=Path("unused.csv"),
        encoding="utf-8",
        date_column="Date",
        hour_column="Hour",
        timestamp_column="timestamp",
        target_column="target",
        date_format="%d/%m/%Y",
        expected_frequency="h",
        expected_rows=expected_rows,
    )


def make_valid_data() -> pd.DataFrame:
    """Create a complete three-hour time series."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=3,
                freq="h",
            ),
            "target": [10, 12, 15],
            "Functioning Day": ["Yes", "Yes", "Yes"],
        }
    )


def test_valid_series_receives_valid_status() -> None:
    """A complete, ordered series should pass validation."""
    report = validate_time_series(
        make_valid_data(),
        make_config(),
        closure_column="Functioning Day",
        closure_value="No",
    )

    assert report.status == "valid"
    assert report.row_count == 3
    assert report.timestamps_sorted
    assert report.duplicate_timestamp_count == 0
    assert report.missing_timestamp_count == 0
    assert report.irregular_interval_count == 0
    assert report.errors == ()
    assert report.warnings == ()


def test_missing_hour_makes_series_invalid() -> None:
    """A gap in an hourly series should be reported."""
    data = make_valid_data().iloc[[0, 2]].reset_index(drop=True)

    report = validate_time_series(
        data,
        make_config(expected_rows=None),
    )

    assert report.status == "invalid"
    assert report.missing_timestamp_count == 1
    assert report.irregular_interval_count == 1
    assert any("missing" in message for message in report.errors)


def test_duplicate_and_disordered_timestamps_are_reported() -> None:
    """Duplicates and temporal disorder should not pass silently."""
    data = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01 01:00:00"),
                pd.Timestamp("2026-01-01 00:00:00"),
                pd.Timestamp("2026-01-01 01:00:00"),
            ],
            "target": [12, 10, 14],
        }
    )

    report = validate_time_series(
        data,
        make_config(),
    )

    assert report.status == "invalid"
    assert not report.timestamps_sorted
    assert report.duplicate_timestamp_count == 1
    assert any(
        "monotonically increasing" in message
        for message in report.warnings
    )


def test_bad_target_values_make_series_invalid() -> None:
    """Missing and negative targets should be reported."""
    data = make_valid_data()
    data.loc[0, "target"] = -1
    data.loc[1, "target"] = None

    report = validate_time_series(
        data,
        make_config(),
    )

    assert report.status == "invalid"
    assert report.negative_target_count == 1
    assert report.missing_target_count == 1


def test_closure_mismatches_produce_warnings() -> None:
    """Unexpected zeros and nonzero closures should be reported."""
    data = make_valid_data()
    data["target"] = [0, 12, 15]
    data["Functioning Day"] = ["Yes", "No", "Yes"]

    report = validate_time_series(
        data,
        make_config(),
        closure_column="Functioning Day",
        closure_value="No",
    )

    assert report.status == "warning"
    assert report.zero_without_closure_count == 1
    assert report.closure_without_zero_count == 1
    assert len(report.warnings) == 2


def test_missing_required_validation_column_raises_error() -> None:
    """Validation requires timestamp and target columns."""
    data = pd.DataFrame({"target": [1, 2, 3]})

    with pytest.raises(
        MissingColumnsError,
        match="timestamp",
    ):
        validate_time_series(data, make_config())


def test_save_validation_report_creates_json_and_markdown(
    tmp_path: Path,
) -> None:
    """A report should be saved in two readable formats."""
    report = validate_time_series(
        make_valid_data(),
        make_config(),
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    save_validation_report(
        report,
        json_path,
        markdown_path,
    )

    saved_json = json.loads(
        json_path.read_text(encoding="utf-8")
    )
    saved_markdown = markdown_path.read_text(encoding="utf-8")

    assert saved_json["status"] == "valid"
    assert "# Time-Series Validation Report" in saved_markdown
    assert "Overall status:** VALID" in saved_markdown