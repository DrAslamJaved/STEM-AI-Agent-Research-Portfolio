"""Tests for robust residual anomaly detection."""

import json

import numpy as np
import pandas as pd
import pytest

from time_series_agent.anomalies import (
    detect_residual_anomalies,
    save_anomaly_results,
    select_top_anomaly_candidates,
)
from time_series_agent.exceptions import (
    AnomalyDetectionError,
)


def make_residual_frame() -> pd.DataFrame:
    """Create normal, anomalous, and closure residuals."""
    ordinary = np.tile(
        np.array(
            [-3, -2, -1, 0, 1, 2, 3],
            dtype="float64",
        ),
        10,
    )

    residual_values = np.concatenate(
        [
            ordinary,
            np.array(
                [100, -100, 250],
                dtype="float64",
            ),
        ]
    )

    closure_flags = np.zeros(
        len(residual_values),
        dtype=bool,
    )
    closure_flags[-1] = True

    forecasts = np.full(
        len(residual_values),
        200.0,
    )

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=len(residual_values),
                freq="h",
            ),
            "actual": forecasts + residual_values,
            "forecast": forecasts,
            "residual": residual_values,
            "is_known_closure": closure_flags,
        }
    )


def test_detects_actionable_anomalies() -> None:
    """Large nonclosure residuals should be actionable."""
    labeled, summary = detect_residual_anomalies(
        make_residual_frame(),
        threshold=3.5,
        minimum_reference_rows=20,
    )

    actionable = labeled.loc[
        labeled["is_actionable_anomaly"]
    ]

    assert len(actionable) == 2
    assert summary.actionable_anomaly_count == 2
    assert summary.positive_anomaly_count == 1
    assert summary.negative_anomaly_count == 1
    assert summary.known_closure_count == 1

    assert set(actionable["anomaly_type"]) == {
        "positive_demand_spike",
        "negative_demand_drop",
    }


def test_known_closure_is_not_actionable() -> None:
    """A known closure should remain a separate event."""
    labeled, summary = detect_residual_anomalies(
        make_residual_frame()
    )

    closure = labeled.loc[
        labeled["is_known_closure"]
    ].iloc[0]

    assert closure["is_statistical_anomaly"]
    assert not closure["is_actionable_anomaly"]
    assert closure["anomaly_type"] == "known_closure"
    assert (
        closure["anomaly_severity"]
        == "known_closure"
    )
    assert (
        summary.closure_statistical_anomaly_count
        == 1
    )


def test_input_frame_is_not_mutated() -> None:
    """Detection should not modify the caller's data."""
    original = make_residual_frame()
    before = original.copy(deep=True)

    detect_residual_anomalies(original)

    pd.testing.assert_frame_equal(
        original,
        before,
    )


def test_missing_required_column_is_rejected() -> None:
    """Required residual columns must be present."""
    bad_data = make_residual_frame().drop(
        columns="residual"
    )

    with pytest.raises(
        AnomalyDetectionError,
        match="missing required columns",
    ):
        detect_residual_anomalies(bad_data)


def test_duplicate_timestamp_is_rejected() -> None:
    """Every residual timestamp must be unique."""
    bad_data = make_residual_frame()
    bad_data.loc[1, "timestamp"] = bad_data.loc[
        0,
        "timestamp",
    ]

    with pytest.raises(
        AnomalyDetectionError,
        match="unique",
    ):
        detect_residual_anomalies(bad_data)


def test_disordered_timestamp_is_rejected() -> None:
    """Residual rows must remain chronological."""
    bad_data = make_residual_frame()
    bad_data.loc[
        [0, 1],
        "timestamp",
    ] = bad_data.loc[
        [1, 0],
        "timestamp",
    ].to_numpy()

    with pytest.raises(
        AnomalyDetectionError,
        match="chronological",
    ):
        detect_residual_anomalies(bad_data)


def test_insufficient_reference_rows_are_rejected() -> None:
    """Calibration requires enough nonclosure rows."""
    small_data = make_residual_frame().iloc[:10]

    with pytest.raises(
        AnomalyDetectionError,
        match="Too few",
    ):
        detect_residual_anomalies(
            small_data,
            minimum_reference_rows=20,
        )


def test_zero_mad_is_rejected() -> None:
    """Constant residuals cannot define a MAD threshold."""
    constant_data = make_residual_frame().iloc[:30].copy()
    constant_data["residual"] = 5.0
    constant_data["is_known_closure"] = False

    with pytest.raises(
        AnomalyDetectionError,
        match="MAD",
    ):
        detect_residual_anomalies(constant_data)


def test_invalid_threshold_is_rejected() -> None:
    """Threshold must be positive and finite."""
    with pytest.raises(
        AnomalyDetectionError,
        match="threshold",
    ):
        detect_residual_anomalies(
            make_residual_frame(),
            threshold=0,
        )


def test_anomaly_results_can_be_saved(
    tmp_path,
) -> None:
    """Labels and summary should be reproducible files."""
    labeled, summary = detect_residual_anomalies(
        make_residual_frame()
    )

    labels_path = tmp_path / "labels.csv"
    summary_path = tmp_path / "summary.json"

    saved_labels, saved_summary = (
        save_anomaly_results(
            labeled_anomalies=labeled,
            summary=summary,
            labels_path=labels_path,
            summary_path=summary_path,
        )
    )

    loaded_labels = pd.read_csv(saved_labels)
    loaded_summary = json.loads(
        saved_summary.read_text(
            encoding="utf-8"
        )
    )

    assert len(loaded_labels) == len(labeled)
    assert (
        loaded_summary[
            "actionable_anomaly_count"
        ]
        == 2
    )
    assert (
        loaded_summary[
            "known_closures_used_for_calibration"
        ]
        == "no"
    )

def test_top_candidates_are_sorted_before_columns_are_selected() -> None:
    """Top candidates should be ranked by absolute score."""
    labeled, _ = detect_residual_anomalies(
        make_residual_frame()
    )

    selected = select_top_anomaly_candidates(
        labeled_anomalies=labeled,
        limit=10,
    )

    assert len(selected) == 2
    assert selected[
        "absolute_modified_z_score"
    ].is_monotonic_decreasing
    assert selected[
        "is_actionable_anomaly"
    ].all()
    assert not selected[
        "is_known_closure"
    ].any()


def test_invalid_top_candidate_limit_is_rejected() -> None:
    """Candidate-selection limit must be positive."""
    labeled, _ = detect_residual_anomalies(
        make_residual_frame()
    )

    with pytest.raises(
        AnomalyDetectionError,
        match="limit",
    ):
        select_top_anomaly_candidates(
            labeled_anomalies=labeled,
            limit=0,
        )