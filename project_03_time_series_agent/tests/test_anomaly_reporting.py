"""Tests for contextual anomaly reporting."""

import numpy as np
import pandas as pd
import pytest

from time_series_agent.anomaly_reporting import (
    build_anomaly_episodes,
    enrich_anomaly_context,
    select_top_actionable_anomalies,
    create_anomaly_report_figures,
    write_anomaly_report,
)
from time_series_agent.exceptions import (
    AnomalyDetectionError,
)


def make_labels() -> pd.DataFrame:
    """Create three deterministic anomaly episodes."""
    rows = 12
    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=rows,
        freq="h",
    )

    forecast = np.full(rows, 100.0)
    actual = np.full(rows, 100.0)
    actionable = np.zeros(rows, dtype=bool)
    anomaly_type = np.full(
        rows,
        "normal",
        dtype=object,
    )
    severity = np.full(
        rows,
        "normal",
        dtype=object,
    )
    scores = np.full(rows, 0.5)

    actionable[[1, 2]] = True
    forecast[[1, 2]] = [0.0, 10.0]
    actual[[1, 2]] = [200.0, 220.0]
    anomaly_type[[1, 2]] = (
        "positive_demand_spike"
    )
    severity[[1, 2]] = ["high", "moderate"]
    scores[[1, 2]] = [6.0, 4.0]

    actionable[[5, 6, 7]] = True
    forecast[[5, 6, 7]] = 250.0
    actual[[5, 6, 7]] = [40.0, 30.0, 20.0]
    anomaly_type[[5, 6, 7]] = (
        "negative_demand_drop"
    )
    severity[[5, 6, 7]] = [
        "moderate",
        "high",
        "extreme",
    ]
    scores[[5, 6, 7]] = [4.0, 6.0, 8.0]

    actionable[10] = True
    forecast[10] = 100.0
    actual[10] = 300.0
    anomaly_type[10] = "positive_demand_spike"
    severity[10] = "high"
    scores[10] = 6.5

    residual = actual - forecast

    signed_scores = scores.copy()
    signed_scores[[5, 6, 7]] *= -1
    known_closures = np.zeros(rows, dtype=bool)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": actual,
            "forecast": forecast,
            "residual": residual,
            "absolute_modified_z_score": scores,
            "is_actionable_anomaly": actionable,
            "anomaly_type": anomaly_type,
            "anomaly_severity": severity,
            "modified_z_score": signed_scores,
            "is_known_closure": known_closures,
        }
    )


def make_source() -> pd.DataFrame:
    """Create contextual source observations."""
    rows = 12
    rainfall = np.zeros(rows)
    rainfall[[5, 6, 7]] = [1.0, 2.0, 1.5]

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=rows,
                freq="h",
            ),
            "Temperature_C": np.full(rows, 20.0),
            "Humidity(%)": np.full(rows, 60.0),
            "Rainfall(mm)": rainfall,
            "Snowfall (cm)": np.zeros(rows),
            "Seasons": np.full(rows, "Winter"),
            "Holiday": np.full(
                rows,
                "No Holiday",
            ),
            "Functioning Day": np.full(rows, "Yes"),
        }
    )


def make_contextual_labels() -> pd.DataFrame:
    """Create enriched labels for episode tests."""
    return enrich_anomaly_context(
        labels=make_labels(),
        source_data=make_source(),
    )


def test_context_is_attached_without_row_loss() -> None:
    """Every label should receive source context."""
    enriched = make_contextual_labels()

    assert len(enriched) == len(make_labels())
    assert enriched.isna().sum().sum() == 0

    assert {
        "temperature_c",
        "humidity_percent",
        "rainfall_mm",
        "season",
        "holiday",
        "functioning_day",
    }.issubset(enriched.columns)


def test_three_episode_types_are_created() -> None:
    """Consecutive anomalies should form three episodes."""
    episodes = build_anomaly_episodes(
        make_contextual_labels()
    )

    assert len(episodes) == 3
    assert episodes["anomaly_hours"].tolist() == [
        2,
        3,
        1,
    ]

    assert episodes["episode_context"].tolist() == [
        "forecast_floor_positive_episode",
        "rain_coincident_negative_episode",
        "other_residual_episode",
    ]


def test_peak_severity_comes_from_peak_score() -> None:
    """Episode severity should follow its strongest hour."""
    episodes = build_anomaly_episodes(
        make_contextual_labels()
    )

    assert episodes.loc[
        1,
        "maximum_severity",
    ] == "extreme"

    assert episodes.loc[
        1,
        "maximum_absolute_modified_z_score",
    ] == 8.0


def test_top_anomalies_are_ranked() -> None:
    """Top rows should be ordered by absolute score."""
    selected = select_top_actionable_anomalies(
        make_contextual_labels(),
        limit=3,
    )

    assert selected["anomaly_rank"].tolist() == [
        1,
        2,
        3,
    ]

    assert selected[
        "absolute_modified_z_score"
    ].is_monotonic_decreasing


def test_no_actionable_rows_returns_empty_episodes() -> None:
    """No alerts should produce an empty episode table."""
    contextual = make_contextual_labels()
    contextual["is_actionable_anomaly"] = False

    episodes = build_anomaly_episodes(contextual)

    assert episodes.empty
    assert list(episodes.columns) == (
        list(episodes.columns)
    )


def test_duplicate_source_timestamp_is_rejected() -> None:
    """Source timestamps must uniquely identify context."""
    source = make_source()
    source.loc[1, "timestamp"] = source.loc[
        0,
        "timestamp",
    ]

    with pytest.raises(
        AnomalyDetectionError,
        match="unique",
    ):
        enrich_anomaly_context(
            make_labels(),
            source,
        )


def test_missing_context_column_is_rejected() -> None:
    """Required weather context must be available."""
    source = make_source().drop(
        columns="Rainfall(mm)"
    )

    with pytest.raises(
        AnomalyDetectionError,
        match="missing context columns",
    ):
        enrich_anomaly_context(
            make_labels(),
            source,
        )


def test_invalid_episode_gap_is_rejected() -> None:
    """Episode gap must be a positive integer."""
    with pytest.raises(
        AnomalyDetectionError,
        match="maximum_gap_hours",
    ):
        build_anomaly_episodes(
            make_contextual_labels(),
            maximum_gap_hours=0,
        )


def test_invalid_top_limit_is_rejected() -> None:
    """Top-anomaly limit must be positive."""
    with pytest.raises(
        AnomalyDetectionError,
        match="limit",
    ):
        select_top_actionable_anomalies(
            make_contextual_labels(),
            limit=0,
        )


def test_report_figures_are_created(
    tmp_path,
) -> None:
    """Three nonempty reporting figures should be saved."""
    contextual = make_contextual_labels()

    paths = create_anomaly_report_figures(
        contextual_labels=contextual,
        output_directory=tmp_path,
        threshold=3.5,
    )

    assert len(paths) == 3
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_markdown_report_is_created(
    tmp_path,
) -> None:
    """Human-readable report should summarize evidence."""
    contextual = make_contextual_labels()
    episodes = build_anomaly_episodes(contextual)
    top = select_top_actionable_anomalies(
        contextual,
        limit=5,
    )

    detection_summary = {
        "modified_z_threshold": 3.5,
        "row_count": 12,
        "reference_row_count": 12,
        "known_closure_count": 0,
        "actionable_anomaly_count": 6,
        "positive_anomaly_count": 3,
        "negative_anomaly_count": 3,
        "actionable_anomaly_rate_percent": 50.0,
    }

    episode_summary = {
        "episode_count": 3,
        "episode_context_counts": {
            "forecast_floor_positive_episode": 1,
            "rain_coincident_negative_episode": 1,
            "other_residual_episode": 1,
        },
        "episode_context_anomaly_hours": {
            "forecast_floor_positive_episode": 2,
            "rain_coincident_negative_episode": 3,
            "other_residual_episode": 1,
        },
        "top_ten_dates_share_percent": 100.0,
    }

    output_path = tmp_path / "anomaly_report.md"

    saved = write_anomaly_report(
        detection_summary=detection_summary,
        episode_summary=episode_summary,
        episodes=episodes,
        top_anomalies=top,
        output_path=output_path,
    )

    contents = saved.read_text(
        encoding="utf-8"
    )

    assert saved.exists()
    assert "# Residual Anomaly Report" in contents
    assert "Forecast-floor positive episodes" in contents
    assert "## Limitations" in contents