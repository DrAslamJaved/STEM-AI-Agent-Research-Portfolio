"""Validate loaded time-series data and generate structured reports."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from time_series_agent.config import DataConfig
from time_series_agent.exceptions import MissingColumnsError


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of time-series validation."""

    status: str
    row_count: int
    column_count: int
    start_timestamp: str | None
    end_timestamp: str | None
    expected_frequency: str | None
    inferred_frequency: str | None
    expected_rows: int | None
    expected_timestamp_count: int
    timestamps_sorted: bool
    duplicate_timestamp_count: int
    missing_timestamp_count: int
    irregular_interval_count: int
    invalid_timestamp_count: int
    missing_target_count: int
    negative_target_count: int
    zero_target_count: int
    zero_without_closure_count: int | None
    closure_without_zero_count: int | None
    target_minimum: float | None
    target_maximum: float | None
    target_mean: float | None
    target_median: float | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _infer_frequency(
    timestamps: pd.DatetimeIndex,
) -> str | None:
    """Infer frequency when enough unique timestamps are available."""
    if len(timestamps) < 3:
        return None

    try:
        return pd.infer_freq(timestamps)
    except ValueError:
        return None


def _target_statistic(
    target: pd.Series,
    statistic: str,
) -> float | None:
    """Calculate a statistic from valid numeric target values."""
    valid_target = target.dropna()

    if valid_target.empty:
        return None

    result = getattr(valid_target, statistic)()
    return float(result)


def validate_time_series(
    data: pd.DataFrame,
    config: DataConfig,
    closure_column: str | None = None,
    closure_value: str | None = None,
) -> ValidationReport:
    """Validate temporal structure and target values.

    This function inspects but does not modify the supplied DataFrame.

    Parameters
    ----------
    data:
        Loaded time-series DataFrame containing the constructed
        timestamp column.
    config:
        Data-loading configuration.
    closure_column:
        Optional column identifying known system closures.
    closure_value:
        Value representing a closure in ``closure_column``.

    Returns
    -------
    ValidationReport
        Structured validation findings.

    Raises
    ------
    MissingColumnsError
        If the timestamp or target column is unavailable.
    """
    required_columns = {
        config.timestamp_column,
        config.target_column,
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise MissingColumnsError(
            f"Validation requires missing column(s): {missing_text}"
        )

    errors: list[str] = []
    warnings: list[str] = []

    row_count = len(data)
    column_count = len(data.columns)

    timestamps = pd.to_datetime(
        data[config.timestamp_column],
        errors="coerce",
    )
    target = pd.to_numeric(
        data[config.target_column],
        errors="coerce",
    )

    invalid_timestamp_count = int(timestamps.isna().sum())
    missing_target_count = int(target.isna().sum())
    negative_target_count = int(target.lt(0).sum())
    zero_target_count = int(target.eq(0).sum())

    if row_count == 0:
        errors.append("The dataset contains no observations.")

    if invalid_timestamp_count:
        errors.append(
            f"{invalid_timestamp_count} timestamp value(s) are invalid."
        )

    if missing_target_count:
        errors.append(
            f"{missing_target_count} target value(s) are missing or invalid."
        )

    if negative_target_count:
        errors.append(
            f"{negative_target_count} target value(s) are negative."
        )

    timestamps_sorted = bool(
        invalid_timestamp_count == 0
        and timestamps.is_monotonic_increasing
    )

    if row_count and not timestamps_sorted:
        warnings.append(
            "Timestamps are not in monotonically increasing order."
        )

    valid_timestamps = timestamps.dropna()
    duplicate_timestamp_count = int(valid_timestamps.duplicated().sum())

    if duplicate_timestamp_count:
        errors.append(
            f"{duplicate_timestamp_count} duplicate timestamp(s) were found."
        )

    sorted_unique_timestamps = pd.DatetimeIndex(
        valid_timestamps.drop_duplicates().sort_values()
    )

    start_timestamp: str | None = None
    end_timestamp: str | None = None
    inferred_frequency: str | None = None
    expected_timestamp_count = 0
    missing_timestamp_count = 0
    irregular_interval_count = 0

    if len(sorted_unique_timestamps):
        start_timestamp = str(sorted_unique_timestamps.min())
        end_timestamp = str(sorted_unique_timestamps.max())
        inferred_frequency = _infer_frequency(sorted_unique_timestamps)

    if (
        len(sorted_unique_timestamps) >= 2
        and config.expected_frequency is not None
    ):
        expected_difference = pd.to_timedelta(
            1,
            unit=config.expected_frequency,
        )
        differences = pd.Series(sorted_unique_timestamps).diff().dropna()

        irregular_interval_count = int(
            differences.ne(expected_difference).sum()
        )

        expected_index = pd.date_range(
            start=sorted_unique_timestamps.min(),
            end=sorted_unique_timestamps.max(),
            freq=config.expected_frequency,
        )
        expected_timestamp_count = len(expected_index)
        missing_timestamp_count = len(
            expected_index.difference(sorted_unique_timestamps)
        )

        if irregular_interval_count:
            errors.append(
                f"{irregular_interval_count} interval(s) differ from "
                f"the expected '{config.expected_frequency}' frequency."
            )

        if missing_timestamp_count:
            errors.append(
                f"{missing_timestamp_count} expected timestamp(s) are missing."
            )

    if (
        config.expected_rows is not None
        and row_count != config.expected_rows
    ):
        errors.append(
            f"Observed row count {row_count} differs from expected "
            f"row count {config.expected_rows}."
        )

    zero_without_closure_count: int | None = None
    closure_without_zero_count: int | None = None

    if closure_column is not None and closure_value is not None:
        if closure_column not in data.columns:
            warnings.append(
                f"Closure column '{closure_column}' is unavailable."
            )
        else:
            closure_mask = data[closure_column].eq(closure_value)
            zero_mask = target.eq(0)

            zero_without_closure_count = int(
                (zero_mask & ~closure_mask).sum()
            )
            closure_without_zero_count = int(
                (closure_mask & ~zero_mask).sum()
            )

            if zero_without_closure_count:
                warnings.append(
                    f"{zero_without_closure_count} zero target value(s) "
                    "occur outside documented closures."
                )

            if closure_without_zero_count:
                warnings.append(
                    f"{closure_without_zero_count} documented closure "
                    "row(s) have nonzero targets."
                )

    if errors:
        status = "invalid"
    elif warnings:
        status = "warning"
    else:
        status = "valid"

    return ValidationReport(
        status=status,
        row_count=row_count,
        column_count=column_count,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        expected_frequency=config.expected_frequency,
        inferred_frequency=inferred_frequency,
        expected_rows=config.expected_rows,
        expected_timestamp_count=expected_timestamp_count,
        timestamps_sorted=timestamps_sorted,
        duplicate_timestamp_count=duplicate_timestamp_count,
        missing_timestamp_count=missing_timestamp_count,
        irregular_interval_count=irregular_interval_count,
        invalid_timestamp_count=invalid_timestamp_count,
        missing_target_count=missing_target_count,
        negative_target_count=negative_target_count,
        zero_target_count=zero_target_count,
        zero_without_closure_count=zero_without_closure_count,
        closure_without_zero_count=closure_without_zero_count,
        target_minimum=_target_statistic(target, "min"),
        target_maximum=_target_statistic(target, "max"),
        target_mean=_target_statistic(target, "mean"),
        target_median=_target_statistic(target, "median"),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def render_validation_markdown(
    report: ValidationReport,
) -> str:
    """Render a validation report as readable Markdown."""
    lines = [
        "# Time-Series Validation Report",
        "",
        f"- **Overall status:** {report.status.upper()}",
        f"- **Rows:** {report.row_count}",
        f"- **Columns:** {report.column_count}",
        f"- **First timestamp:** {report.start_timestamp}",
        f"- **Last timestamp:** {report.end_timestamp}",
        f"- **Expected frequency:** {report.expected_frequency}",
        f"- **Inferred frequency:** {report.inferred_frequency}",
        "",
        "## Temporal checks",
        "",
        f"- Timestamps sorted: {report.timestamps_sorted}",
        (
            "- Duplicate timestamps: "
            f"{report.duplicate_timestamp_count}"
        ),
        f"- Missing timestamps: {report.missing_timestamp_count}",
        (
            "- Irregular intervals: "
            f"{report.irregular_interval_count}"
        ),
        "",
        "## Target checks",
        "",
        f"- Missing targets: {report.missing_target_count}",
        f"- Negative targets: {report.negative_target_count}",
        f"- Zero targets: {report.zero_target_count}",
        f"- Minimum: {report.target_minimum}",
        f"- Maximum: {report.target_maximum}",
        f"- Mean: {report.target_mean}",
        f"- Median: {report.target_median}",
        "",
        "## Errors",
        "",
    ]

    if report.errors:
        lines.extend(f"- {message}" for message in report.errors)
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])

    if report.warnings:
        lines.extend(f"- {message}" for message in report.warnings)
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def save_validation_report(
    report: ValidationReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Save validation results as JSON and Markdown."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)

    json_output.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(
        render_validation_markdown(report),
        encoding="utf-8",
    )