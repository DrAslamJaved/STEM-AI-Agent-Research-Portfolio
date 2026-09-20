import json

import numpy as np
import pandas as pd
import pytest

from causal_audit_agent.diagnostics import (
    DiagnosticStatus,
    DiagnosticThresholds,
    diagnose,
    standardized_mean_difference,
)


def balanced_frame(n=96):
    if n % 8:
        raise ValueError("n must be divisible by eight")
    x = np.tile(np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0]), n // 8)
    treatment = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), n // 8)
    return pd.DataFrame({"T": treatment, "X": x, "Z": x + 1.0})


def test_balanced_data_passes_and_serializes():
    frame = balanced_frame()
    report = diagnose(frame, "T", ["X", "Z"], np.full(len(frame), 0.5))
    assert report.status is DiagnosticStatus.PASS
    assert report.passed
    assert report.overlap_fraction == 1.0
    assert report.effective_sample_size == pytest.approx(len(frame))
    assert report.effective_sample_fraction == pytest.approx(1.0)
    assert report.max_weight == pytest.approx(2.0)
    assert report.reasons == ()
    payload = report.to_dict()
    assert payload["status"] == "PASS"
    assert payload["reasons"] == []
    assert payload["requires_human_review"] is True
    json.dumps(payload)


def test_weighting_reduces_measured_imbalance():
    frame = pd.DataFrame(
        {
            "T": [0, 0, 0, 0, 1, 1, 1, 1],
            "X": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    propensity = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8])
    report = diagnose(
        frame,
        "T",
        ["X"],
        propensity,
        thresholds=DiagnosticThresholds(
            max_abs_smd=10.0,
            min_overlap_fraction=0.0,
            min_ess_fraction=0.01,
            max_weight=100.0,
        ),
    )
    assert abs(report.weighted_covariate_smds["X"]) < abs(
        report.covariate_smds["X"]
    )


def test_each_failed_gate_emits_a_deterministic_reason():
    frame = pd.DataFrame(
        {
            "T": [0, 0, 0, 1, 1, 1],
            "X": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        }
    )
    propensity = np.array([1.0, 0.2, 0.2, 0.8, 0.8, 0.0])
    report = diagnose(
        frame,
        "T",
        ["X"],
        propensity,
        thresholds=DiagnosticThresholds(
            min_ess_fraction=0.99,
            max_weight=2.0,
            propensity_clip=0.01,
        ),
    )
    assert report.status is DiagnosticStatus.REVIEW
    assert not report.passed
    assert report.max_abs_smd == np.inf
    assert report.extreme_weight_fraction > 0.0
    assert report.clipped_propensity_fraction > 0.0
    assert report.reasons == (
        "weighted_balance_exceeds_threshold",
        "insufficient_propensity_overlap",
        "low_effective_sample_size",
        "extreme_weights_present",
        "propensity_clipping_exceeds_threshold",
    )


def test_custom_overlap_bounds_are_honored():
    frame = balanced_frame(24)
    propensity = np.full(len(frame), 0.15)
    report = diagnose(
        frame,
        "T",
        [],
        propensity,
        overlap_bounds=(0.2, 0.8),
        thresholds=DiagnosticThresholds(
            min_ess_fraction=0.01,
            max_weight=100.0,
        ),
    )
    assert report.max_abs_smd == 0.0
    assert report.unweighted_max_abs_smd == 0.0
    assert report.overlap_fraction == 0.0
    assert report.reasons == ("insufficient_propensity_overlap",)


def test_att_weights_are_supported():
    frame = balanced_frame(24)
    propensity = np.where(frame["T"].to_numpy() == 1, 0.6, 0.4)
    report = diagnose(frame, "T", ["X"], propensity, estimand="att")
    assert report.estimand == "ATT"
    assert report.n_observations == len(frame)


def test_constant_equal_groups_have_zero_smd():
    treatment = np.array([0, 0, 1, 1])
    values = np.ones(4)
    assert standardized_mean_difference(values, treatment) == 0.0


def test_constant_unequal_groups_have_infinite_smd():
    treatment = np.array([0, 0, 1, 1])
    values = np.array([0.0, 0.0, 1.0, 1.0])
    assert standardized_mean_difference(values, treatment) == np.inf


def test_weighted_smd_accepts_valid_weights():
    treatment = np.array([0, 0, 1, 1])
    values = np.array([0.0, 1.0, 0.5, 1.5])
    result = standardized_mean_difference(
        values, treatment, np.array([1.0, 2.0, 2.0, 1.0])
    )
    assert np.isfinite(result)


@pytest.mark.parametrize(
    ("values", "treatment", "weights", "message"),
    [
        (np.ones((2, 2)), np.array([0, 0, 1, 1]), None, "equal-length vectors"),
        (np.array([0.0, np.nan, 1.0, 2.0]), np.array([0, 0, 1, 1]), None, "finite"),
        (np.arange(4.0), np.array([0, 0, 0, 0]), None, "binary"),
        (np.arange(3.0), np.array([0, 0, 1]), None, "at least two"),
        (np.arange(4.0), np.array([0, 0, 1, 1]), np.ones(3), "equal-length"),
        (
            np.arange(4.0),
            np.array([0, 0, 1, 1]),
            np.array([1.0, 1.0, 0.0, 1.0]),
            "strictly positive",
        ),
    ],
)
def test_smd_rejects_invalid_inputs(values, treatment, weights, message):
    with pytest.raises(ValueError, match=message):
        standardized_mean_difference(values, treatment, weights)


def test_degenerate_weight_variance_is_rejected():
    with pytest.raises(ValueError, match="finite variance"):
        standardized_mean_difference(
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0, 0, 1, 1]),
            np.array([1e300, 1e-300, 1e300, 1e-300]),
        )


@pytest.mark.parametrize("value", [None, [], {}])
def test_diagnose_rejects_non_dataframe(value):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        diagnose(value, "T", [], np.array([]))


def test_invalid_threshold_type_is_rejected():
    with pytest.raises(TypeError, match="DiagnosticThresholds"):
        diagnose(balanced_frame(24), "T", [], np.full(24, 0.5), thresholds={})


@pytest.mark.parametrize(
    ("treatment", "covariates", "message"),
    [
        ("", [], "non-empty string"),
        ("T", "X", "sequence"),
        ("T", [""], "sequence"),
        ("T", ["X", "X"], "duplicates"),
        ("T", ["T"], "cannot be included"),
        ("T", ["MISSING"], "missing columns"),
    ],
)
def test_invalid_column_requests_are_rejected(treatment, covariates, message):
    with pytest.raises(ValueError, match=message):
        diagnose(balanced_frame(24), treatment, covariates, np.full(24, 0.5))


def test_non_numeric_column_is_rejected():
    frame = balanced_frame(24).assign(label="a")
    with pytest.raises(ValueError, match="non-numeric"):
        diagnose(frame, "T", ["label"], np.full(24, 0.5))


def test_nonfinite_diagnostic_column_is_rejected():
    frame = balanced_frame(24)
    frame.loc[0, "X"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        diagnose(frame, "T", ["X"], np.full(24, 0.5))


@pytest.mark.parametrize(
    ("treatment", "message"),
    [
        ([0] * 20, "exactly binary"),
        ([0] * 19 + [1], "at least two"),
    ],
)
def test_invalid_treatment_is_rejected(treatment, message):
    frame = pd.DataFrame({"T": treatment, "X": np.arange(20.0)})
    with pytest.raises(ValueError, match=message):
        diagnose(frame, "T", ["X"], np.full(20, 0.5))


@pytest.mark.parametrize(
    ("propensity", "message"),
    [
        (np.full((24, 1), 0.5), "one-dimensional"),
        (np.full(23, 0.5), "row count"),
        (np.array([0.5] * 23 + [np.nan]), "finite"),
        (np.array([0.5] * 23 + [1.1]), "unit interval"),
    ],
)
def test_invalid_propensity_is_rejected(propensity, message):
    with pytest.raises(ValueError, match=message):
        diagnose(balanced_frame(24), "T", ["X"], propensity)


@pytest.mark.parametrize("estimand", ["CATE", "", None])
def test_unsupported_estimand_is_rejected(estimand):
    with pytest.raises(ValueError, match="ATE or ATT"):
        diagnose(
            balanced_frame(24), "T", [], np.full(24, 0.5), estimand=estimand
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"overlap_bounds": (0.0, 0.9)}, "overlap bounds"),
        ({"max_abs_smd": -0.1}, "max_abs_smd"),
        ({"min_overlap_fraction": 1.1}, "min_overlap_fraction"),
        ({"min_ess_fraction": 0.0}, "min_ess_fraction"),
        ({"max_weight": 0.0}, "max_weight"),
        ({"propensity_clip": 0.5}, "propensity_clip"),
        ({"max_clipped_fraction": -0.1}, "max_clipped_fraction"),
    ],
)
def test_invalid_thresholds_are_rejected(changes, message):
    thresholds = DiagnosticThresholds(**changes)
    with pytest.raises(ValueError, match=message):
        diagnose(
            balanced_frame(24),
            "T",
            [],
            np.full(24, 0.5),
            thresholds=thresholds,
        )
