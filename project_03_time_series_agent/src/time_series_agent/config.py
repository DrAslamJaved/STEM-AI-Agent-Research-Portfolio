"""Load and represent project configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from time_series_agent.exceptions import ConfigurationError


@dataclass(frozen=True)
class DataConfig:
    """Configuration required to load the raw time-series dataset."""

    raw_file_path: Path
    encoding: str
    date_column: str
    hour_column: str
    timestamp_column: str
    target_column: str
    date_format: str
    expected_frequency: str | None = None
    expected_rows: int | None = None


REQUIRED_DATA_KEYS = {
    "raw_file_path",
    "encoding",
    "date_column",
    "hour_column",
    "timestamp_column",
    "target_column",
    "date_format",
}


def _build_data_config(values: Mapping[str, Any]) -> DataConfig:
    """Construct and validate a DataConfig from a mapping."""
    missing_keys = REQUIRED_DATA_KEYS.difference(values)

    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise ConfigurationError(
            f"Data configuration is missing required keys: {missing_text}"
        )

    expected_rows = values.get("expected_rows")

    if expected_rows is not None:
        if not isinstance(expected_rows, int) or expected_rows <= 0:
            raise ConfigurationError(
                "'expected_rows' must be a positive integer or null."
            )

    return DataConfig(
        raw_file_path=Path(str(values["raw_file_path"])),
        encoding=str(values["encoding"]),
        date_column=str(values["date_column"]),
        hour_column=str(values["hour_column"]),
        timestamp_column=str(values["timestamp_column"]),
        target_column=str(values["target_column"]),
        date_format=str(values["date_format"]),
        expected_frequency=values.get("expected_frequency"),
        expected_rows=expected_rows,
    )


def load_data_config(config_path: str | Path) -> DataConfig:
    """Load the data section of a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    DataConfig
        Validated data-loading configuration.

    Raises
    ------
    ConfigurationError
        If the file is missing, malformed, or lacks required settings.
    """
    path = Path(config_path)

    if not path.is_file():
        raise ConfigurationError(
            f"Configuration file does not exist: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as config_file:
            contents = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Configuration file contains invalid YAML: {path}"
        ) from error
    except OSError as error:
        raise ConfigurationError(
            f"Configuration file could not be read: {path}"
        ) from error

    if not isinstance(contents, dict):
        raise ConfigurationError(
            "Configuration must contain a top-level mapping."
        )

    data_section = contents.get("data")

    if not isinstance(data_section, dict):
        raise ConfigurationError(
            "Configuration must contain a 'data' mapping."
        )

    return _build_data_config(data_section)
