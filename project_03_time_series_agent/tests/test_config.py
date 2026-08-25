"""Tests for project configuration loading."""

from pathlib import Path

import pytest

from time_series_agent.config import load_data_config
from time_series_agent.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_actual_data_config() -> None:
    """The project YAML should produce the documented data settings."""
    config_path = PROJECT_ROOT / "configs" / "default.yaml"

    config = load_data_config(config_path)

    assert config.raw_file_path == Path(
        "data/raw/SeoulBikeData.csv"
    )
    assert config.encoding == "latin-1"
    assert config.date_column == "Date"
    assert config.hour_column == "Hour"
    assert config.timestamp_column == "timestamp"
    assert config.target_column == "Rented Bike Count"
    assert config.date_format == "%d/%m/%Y"
    assert config.expected_frequency == "h"
    assert config.expected_rows == 8760


def test_missing_config_file_raises_clear_error(
    tmp_path: Path,
) -> None:
    """A missing YAML file should raise ConfigurationError."""
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        ConfigurationError,
        match="does not exist",
    ):
        load_data_config(missing_path)


def test_missing_data_section_raises_clear_error(
    tmp_path: Path,
) -> None:
    """YAML without a data section should be rejected."""
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(
        "project:\n  name: Test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="must contain a 'data' mapping",
    ):
        load_data_config(config_path)


def test_missing_required_data_key_raises_clear_error(
    tmp_path: Path,
) -> None:
    """An incomplete data section should report missing keys."""
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(
        "data:\n  raw_file_path: data.csv\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="missing required keys",
    ):
        load_data_config(config_path)


def test_invalid_expected_rows_raises_clear_error(
    tmp_path: Path,
) -> None:
    """Expected row count should be a positive integer."""
    config_path = tmp_path / "invalid_rows.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  raw_file_path: data.csv",
                "  encoding: utf-8",
                "  date_column: Date",
                "  hour_column: Hour",
                "  timestamp_column: timestamp",
                "  target_column: target",
                '  date_format: "%d/%m/%Y"',
                "  expected_rows: -4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="positive integer",
    ):
        load_data_config(config_path)