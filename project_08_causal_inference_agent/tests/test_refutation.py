from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import causal_audit_agent.refutation as module
from causal_audit_agent.estimation import (
    EffectEstimate,
    EstimationRequest,
    EstimationStatus,
)
from causal_audit_agent.identification import (
    IdentificationResult,
    IdentificationStatus,
)
from causal_audit_agent.refutation import (
    RefutationRequest,
    RefutationStatus,
    run_refutation_suite,
)


def identified():
    return IdentificationResult(
        status=IdentificationStatus.IDENTIFIED,
        estimand="ATE",
        method="backdoor_adjustment",
        adjustment_sets=(("X",),),
        conditioning_variables=(),
        expression="identified expression",
        reasons=("backdoor_criterion_satisfied",),
        assumptions=("consistency", "exchangeability", "positivity"),
    )


def synthetic_data(n=400, seed=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    outcome = 1.0 + x + 2.0 * treatment + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"X": x, "T": treatment, "Y": outcome})


def effect(status, estimate):
    return EffectEstimate(
        status=status,
        estimand="ATE",
        method="g_computation",
        estimate=estimate,
        standard_error=None,
        confidence_interval=None,
        n_observations=0,
        adjustment_set=("X",),
        diagnostics={},
        reasons=(),
    )


def ols_estimator(data, treatment, outcome, identification, request):
    del request
    levels = data[treatment].value_counts()
    if len(levels) != 2 or levels.min() < 2:
        return effect(EstimationStatus.INVALID_DATA, None)
    adjustment_set = sorted(
        identification.adjustment_sets, key=lambda items: (len(items), items)
    )[0]
    design = np.column_stack(
        (
            np.ones(len(data)),
            data[treatment].to_numpy(dtype=float),
            *(
                data[name].to_numpy(dtype=float)
                for name in adjustment_set
            ),
        )
    )
    coefficient = np.linalg.lstsq(
        design, data[outcome].to_numpy(dtype=float), rcond=None
    )[0][1]
    return effect(EstimationStatus.ESTIMATED, float(coefficient))


@pytest.fixture(autouse=True)
def patch_estimator(monkeypatch):
    monkeypatch.setattr(module, "estimate_effect", ols_estimator)


def small_request(**changes):
    values = {
        "n_simulations": 20,
        "random_state": 11,
        "max_absolute_shift": 0.30,
        "max_relative_shift": 0.20,
    }
    values.update(changes)
    return RefutationRequest(**values)


def test_stable_effect_passes_all_refuters():
    report = run_refutation_suite(
        synthetic_data(),
        "T",
        "Y",
        identified(),
        EstimationRequest(method="g_computation"),
        small_request(),
    )
    assert report.status is RefutationStatus.PASS
    assert report.passed
    assert report.baseline_estimate == pytest.approx(2.0, abs=0.08)
    assert tuple(check.name for check in report.checks) == (
        "placebo_treatment",
        "random_common_cause",
        "stratified_subset",
    )
    assert all(check.status is RefutationStatus.PASS for check in report.checks)
    assert report.checks[0].empirical_p_value <= 0.10
    assert report.reasons == ("configured_refuters_detected_no_instability",)
    assert report.requires_human_review


def test_seeded_suite_is_reproducible():
    args = (
        synthetic_data(240),
        "T",
        "Y",
        identified(),
        EstimationRequest(method="ipw"),
        small_request(),
    )
    assert run_refutation_suite(*args).to_dict() == run_refutation_suite(*args).to_dict()


def test_report_serialization_is_json_compatible():
    report = run_refutation_suite(
        synthetic_data(240), "T", "Y", identified(), refutation_request=small_request()
    )
    payload = report.to_dict()
    assert payload["status"] == "PASS"
    assert payload["method"] == "g_computation"
    assert len(payload["checks"]) == 3
    assert payload["checks"][0]["status"] == "PASS"
    assert isinstance(payload["checks"][0]["reasons"], list)


def test_placebo_that_reproduces_observed_effect_requires_review(monkeypatch):
    def constant_estimator(*args, **kwargs):
        return effect(EstimationStatus.ESTIMATED, 2.0)

    monkeypatch.setattr(module, "estimate_effect", constant_estimator)
    report = run_refutation_suite(
        synthetic_data(120), "T", "Y", identified(), refutation_request=small_request()
    )
    placebo = report.checks[0]
    assert report.status is RefutationStatus.REVIEW
    assert not report.passed
    assert placebo.status is RefutationStatus.REVIEW
    assert placebo.empirical_p_value == 1.0
    assert placebo.reasons == ("placebo_effect_not_distinguishable_from_observed",)
    assert report.reasons == ("review_required:placebo_treatment",)


def test_unstable_common_cause_and_subset_require_review(monkeypatch):
    original_n = 240

    def shifting_estimator(data, treatment, outcome, identification, request):
        if any(name.startswith("__random_common_cause__") for name in data.columns):
            return effect(EstimationStatus.ESTIMATED, 3.0)
        if len(data) < original_n:
            return effect(EstimationStatus.ESTIMATED, 3.0)
        return ols_estimator(data, treatment, outcome, identification, request)

    monkeypatch.setattr(module, "estimate_effect", shifting_estimator)
    report = run_refutation_suite(
        synthetic_data(original_n),
        "T",
        "Y",
        identified(),
        refutation_request=small_request(),
    )
    assert report.status is RefutationStatus.REVIEW
    assert report.checks[1].reasons == (
        "mean_absolute_shift_exceeds_threshold",
        "relative_shift_exceeds_threshold",
    )
    assert report.checks[2].status is RefutationStatus.REVIEW
    assert report.reasons == (
        "review_required:random_common_cause",
        "review_required:stratified_subset",
    )


def test_failed_refutation_runs_are_reported(monkeypatch):
    calls = 0

    def failing_after_baseline(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return effect(EstimationStatus.ESTIMATED, 2.0)
        return effect(EstimationStatus.INVALID_DATA, None)

    monkeypatch.setattr(module, "estimate_effect", failing_after_baseline)
    report = run_refutation_suite(
        synthetic_data(120), "T", "Y", identified(), refutation_request=small_request()
    )
    assert report.status is RefutationStatus.REVIEW
    for check in report.checks:
        assert check.successful_runs == 0
        assert check.failed_runs == 20
        assert check.mean_refuted_estimate is None
        assert check.reasons == (
            "failed_refutation_fraction_exceeds_threshold",
            "no_successful_refutation_runs",
        )


def test_blocked_baseline_stops_before_refutation(monkeypatch):
    monkeypatch.setattr(
        module,
        "estimate_effect",
        lambda *args, **kwargs: effect(
            EstimationStatus.BLOCKED_IDENTIFICATION, None
        ),
    )
    report = run_refutation_suite(
        synthetic_data(120), "T", "Y", identified(), refutation_request=small_request()
    )
    assert report.status is RefutationStatus.BLOCKED
    assert report.checks == ()
    assert report.baseline_estimate is None
    assert report.reasons == (
        "baseline_estimation_status:BLOCKED_IDENTIFICATION",
    )


def test_random_common_cause_uses_collision_free_column(monkeypatch):
    seen = []

    def recording_estimator(data, treatment, outcome, identification, request):
        seen.extend(name for items in identification.adjustment_sets for name in items)
        return ols_estimator(data, treatment, outcome, identification, request)

    frame = synthetic_data(120).assign(__random_common_cause__=0.0)
    monkeypatch.setattr(module, "estimate_effect", recording_estimator)
    run_refutation_suite(
        frame, "T", "Y", identified(), refutation_request=small_request()
    )
    assert "__random_common_cause__1" in seen


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"n_simulations": 9}, "n_simulations_must_be_an_integer_at_least_ten"),
        ({"n_simulations": True}, "n_simulations_must_be_an_integer_at_least_ten"),
        ({"subset_fraction": 0.49}, "subset_fraction_must_be_in_half_open_interval"),
        ({"subset_fraction": True}, "subset_fraction_must_be_in_half_open_interval"),
        ({"max_relative_shift": 0.0}, "max_relative_shift_must_be_positive"),
        ({"max_relative_shift": np.nan}, "max_relative_shift_must_be_positive"),
        ({"max_relative_shift": True}, "max_relative_shift_must_be_positive"),
        ({"max_absolute_shift": "bad"}, "max_absolute_shift_must_be_positive"),
        ({"min_effect_scale": -1.0}, "min_effect_scale_must_be_positive"),
        ({"random_common_cause_scale": 0.0}, "random_common_cause_scale_must_be_positive"),
        ({"max_failed_fraction": 1.1}, "max_failed_fraction_must_be_in_unit_interval"),
        ({"max_failed_fraction": True}, "max_failed_fraction_must_be_in_unit_interval"),
        ({"max_placebo_p_value": -0.1}, "max_placebo_p_value_must_be_in_unit_interval"),
        ({"random_state": 1.5}, "random_state_must_be_a_non_negative_integer"),
        ({"random_state": -1}, "random_state_must_be_a_non_negative_integer"),
        ({"random_state": True}, "random_state_must_be_a_non_negative_integer"),
    ],
)
def test_invalid_refutation_request_is_rejected(changes, reason):
    report = run_refutation_suite(
        synthetic_data(120),
        "T",
        "Y",
        identified(),
        refutation_request=replace(small_request(), **changes),
    )
    assert report.status is RefutationStatus.INVALID_REQUEST
    assert reason in report.reasons
    assert report.checks == ()


@pytest.mark.parametrize("method", ["difference_in_means", "unknown", ""])
def test_unsupported_estimation_method_is_rejected(method):
    report = run_refutation_suite(
        synthetic_data(120),
        "T",
        "Y",
        identified(),
        EstimationRequest(method=method),
        small_request(),
    )
    assert report.status is RefutationStatus.INVALID_REQUEST
    assert report.reasons == (f"estimation_method_not_refutable:{method or '<blank>'}",)


def test_non_string_estimation_method_is_rejected():
    report = run_refutation_suite(
        synthetic_data(120),
        "T",
        "Y",
        identified(),
        EstimationRequest(method=7),
        small_request(),
    )
    assert report.reasons == ("estimation_method_not_refutable:<blank>",)


@pytest.mark.parametrize("value", [None, [], {}])
def test_invalid_data_type_fails_explicitly(value):
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        run_refutation_suite(value, "T", "Y", identified())


def test_invalid_identification_type_fails_explicitly():
    with pytest.raises(TypeError, match="identification must be an IdentificationResult"):
        run_refutation_suite(synthetic_data(120), "T", "Y", {})


def test_invalid_estimation_request_type_fails_explicitly():
    with pytest.raises(TypeError, match="estimation_request must be an EstimationRequest"):
        run_refutation_suite(synthetic_data(120), "T", "Y", identified(), {})


def test_invalid_refutation_request_type_fails_explicitly():
    with pytest.raises(TypeError, match="refutation_request must be a RefutationRequest"):
        run_refutation_suite(
            synthetic_data(120), "T", "Y", identified(), refutation_request={}
        )


def test_default_requests_are_constructed():
    report = run_refutation_suite(synthetic_data(), "T", "Y", identified())
    assert report.method == "g_computation"
    assert report.random_state == 0


def test_small_effect_uses_absolute_shift_gate_only():
    check = module._summarize_check(
        "small_effect",
        0.01,
        [0.02] * 10,
        0,
        small_request(max_absolute_shift=0.25, min_effect_scale=0.10),
    )
    assert check.status is RefutationStatus.PASS
    assert check.relative_shift == pytest.approx(0.1)


def test_empty_direct_summary_handles_zero_total():
    check = module._summarize_check(
        "empty", 1.0, [], 0, small_request(max_failed_fraction=0.0)
    )
    assert check.status is RefutationStatus.REVIEW
    assert check.empirical_p_value is None
    assert check.reasons == (
        "failed_refutation_fraction_exceeds_threshold",
        "no_successful_refutation_runs",
    )


def test_refutation_configuration_is_parseable():
    path = Path(__file__).parents[1] / "configs" / "refutation" / "default.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["suite"]["checks"] == [
        "placebo_treatment",
        "random_common_cause",
        "stratified_subset",
    ]
    assert config["safety"]["pass_does_not_validate_causal_assumptions"] is True
