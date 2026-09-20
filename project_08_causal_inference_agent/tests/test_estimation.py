import numpy as np
import pandas as pd
import pytest

from causal_audit_agent.causal_graph import (
    REQUIRED_ASSUMPTIONS,
    CausalDAG,
    Variable,
    VariableRole,
)
from causal_audit_agent.estimation import (
    EffectEstimate,
    EstimationRequest,
    EstimationStatus,
    estimate_effect,
)
from causal_audit_agent.identification import (
    IdentificationRequest,
    IdentificationResult,
    IdentificationStatus,
    identify_estimand,
)


def identified_no_adjustment(estimand="ATE"):
    spec = CausalDAG(
        treatment="T",
        outcome="Y",
        variables=(
            Variable("T", VariableRole.TREATMENT),
            Variable("Y", VariableRole.OUTCOME),
        ),
        edges=(("T", "Y"),),
        assumptions=tuple(sorted(REQUIRED_ASSUMPTIONS)),
    )
    return identify_estimand(spec, IdentificationRequest(estimand=estimand))


def identified_with_x(estimand="ATE"):
    spec = CausalDAG(
        treatment="T",
        outcome="Y",
        variables=(
            Variable("X", VariableRole.CONFOUNDER),
            Variable("T", VariableRole.TREATMENT),
            Variable("Y", VariableRole.OUTCOME),
        ),
        edges=(("X", "T"), ("X", "Y"), ("T", "Y")),
        assumptions=tuple(sorted(REQUIRED_ASSUMPTIONS)),
    )
    return identify_estimand(spec, IdentificationRequest(estimand=estimand))


def simple_data():
    return pd.DataFrame(
        {
            "T": [0, 0, 0, 1, 1, 1],
            "Y": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        }
    )


def confounded_data(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-0.8 * x))
    treatment = rng.binomial(1, propensity)
    outcome = 1.0 + 2.0 * treatment + 1.5 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"T": treatment, "Y": outcome, "X": x})


def test_difference_in_means_estimates_unconfounded_ate():
    result = estimate_effect(
        simple_data(),
        "T",
        "Y",
        identified_no_adjustment(),
        EstimationRequest(method="difference_in_means"),
    )
    assert result.status is EstimationStatus.ESTIMATED
    assert result.estimated
    assert result.estimate == pytest.approx(2.0)
    assert result.standard_error > 0
    assert result.confidence_interval[0] < result.estimate < result.confidence_interval[1]
    assert result.adjustment_set == ()
    assert result.requires_human_review


@pytest.mark.parametrize("method", ["g_computation", "ipw", "aipw"])
def test_adjusted_baselines_recover_known_effect(method):
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x(),
        EstimationRequest(method=method),
    )
    assert result.status is EstimationStatus.ESTIMATED
    assert result.estimate == pytest.approx(2.0, abs=0.15)
    assert result.adjustment_set == ("X",)
    assert result.diagnostics["propensity_min"] > 0
    assert result.diagnostics["propensity_max"] < 1


def test_g_computation_records_design_rank():
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x(),
        EstimationRequest(method="g_computation"),
    )
    assert result.diagnostics["outcome_design_rank"] == 3


def test_ipw_records_effective_sample_size():
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x(),
        EstimationRequest(method="ipw"),
    )
    assert 0 < result.diagnostics["effective_sample_size"] <= len(confounded_data())


def test_ipw_att_is_supported():
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x("ATT"),
        EstimationRequest(method="ipw"),
    )
    assert result.status is EstimationStatus.ESTIMATED
    assert result.estimand == "ATT"
    assert result.estimate == pytest.approx(2.0, abs=0.2)


def test_g_computation_att_is_supported():
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x("ATT"),
        EstimationRequest(method="g_computation"),
    )
    assert result.status is EstimationStatus.ESTIMATED


def test_aipw_att_is_explicitly_unsupported():
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x("ATT"),
        EstimationRequest(method="aipw"),
    )
    assert result.status is EstimationStatus.UNSUPPORTED
    assert result.reasons == ("aipw_baseline_supports_ate_only",)


def test_cate_is_not_silently_estimated_as_ate():
    identification = IdentificationResult(
        status=IdentificationStatus.IDENTIFIED,
        estimand="CATE",
        method="backdoor_adjustment",
        adjustment_sets=((),),
        conditioning_variables=("V",),
        expression="expression",
        reasons=(),
        assumptions=tuple(sorted(REQUIRED_ASSUMPTIONS)),
    )
    result = estimate_effect(simple_data(), "T", "Y", identification)
    assert result.status is EstimationStatus.UNSUPPORTED
    assert result.reasons == ("unsupported_estimand:CATE",)


def test_nonidentified_request_is_blocked_before_data_validation():
    identification = IdentificationResult(
        status=IdentificationStatus.NON_IDENTIFIABLE,
        estimand="ATE",
        method=None,
        adjustment_sets=(),
        conditioning_variables=(),
        expression=None,
        reasons=("no_observed_backdoor_adjustment_set",),
        assumptions=(),
    )
    result = estimate_effect(pd.DataFrame(), "missing", "missing", identification)
    assert result.status is EstimationStatus.BLOCKED_IDENTIFICATION
    assert result.reasons == ("identification_status:NON_IDENTIFIABLE",)


def test_identified_result_must_include_adjustment_set():
    identification = IdentificationResult(
        status=IdentificationStatus.IDENTIFIED,
        estimand="ATE",
        method="backdoor_adjustment",
        adjustment_sets=(),
        conditioning_variables=(),
        expression="expression",
        reasons=(),
        assumptions=(),
    )
    result = estimate_effect(simple_data(), "T", "Y", identification)
    assert result.status is EstimationStatus.BLOCKED_IDENTIFICATION
    assert result.reasons == ("identification_has_no_adjustment_set",)


def test_difference_in_means_cannot_ignore_identified_confounder():
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x(),
        EstimationRequest(method="difference_in_means"),
    )
    assert result.status is EstimationStatus.INVALID_REQUEST
    assert result.reasons == (
        "difference_in_means_cannot_ignore_required_adjustment",
    )


def test_unidentified_adjustment_set_is_rejected():
    result = estimate_effect(
        confounded_data(),
        "T",
        "Y",
        identified_with_x(),
        adjustment_set=(),
    )
    assert result.status is EstimationStatus.INVALID_REQUEST
    assert result.reasons == ("adjustment_set_not_identified",)


@pytest.mark.parametrize(
    ("estimation_request", "reason"),
    [
        (EstimationRequest(method="matching"), "unsupported_method:matching"),
        (
            EstimationRequest(confidence_level=1.0),
            "confidence_level_must_be_between_zero_and_one",
        ),
        (EstimationRequest(propensity_clip=(0.9, 0.1)), "invalid_propensity_clip"),
        (
            EstimationRequest(max_clipped_fraction=2.0),
            "max_clipped_fraction_must_be_in_unit_interval",
        ),
        (
            EstimationRequest(random_state="seed"),
            "random_state_must_be_an_integer",
        ),
    ],
)
def test_invalid_estimation_requests_are_rejected(estimation_request, reason):
    result = estimate_effect(
        simple_data(),
        "T",
        "Y",
        identified_no_adjustment(),
        estimation_request,
    )
    assert result.status is EstimationStatus.INVALID_REQUEST
    assert reason in result.reasons


@pytest.mark.parametrize(
    "clip",
    [None, (0.1,), ("low", "high"), (0.0, 0.9), (0.1, 1.0)],
)
def test_malformed_propensity_clip_is_rejected(clip):
    request = EstimationRequest(propensity_clip=clip)
    result = estimate_effect(simple_data(), "T", "Y", identified_no_adjustment(), request)
    assert "invalid_propensity_clip" in result.reasons


@pytest.mark.parametrize("confidence", [None, 0.0, -0.1, "95%"])
def test_malformed_confidence_level_is_rejected(confidence):
    request = EstimationRequest(confidence_level=confidence)
    result = estimate_effect(simple_data(), "T", "Y", identified_no_adjustment(), request)
    assert "confidence_level_must_be_between_zero_and_one" in result.reasons


@pytest.mark.parametrize("fraction", [None, -0.1, "ten percent"])
def test_malformed_clipping_fraction_is_rejected(fraction):
    request = EstimationRequest(max_clipped_fraction=fraction)
    result = estimate_effect(simple_data(), "T", "Y", identified_no_adjustment(), request)
    assert "max_clipped_fraction_must_be_in_unit_interval" in result.reasons


@pytest.mark.parametrize(
    ("frame", "reason"),
    [
        (pd.DataFrame({"T": [0, 0, 1, 1]}), "missing_column:Y"),
        (
            pd.DataFrame({"T": [0, 0, 1, 1], "Y": ["a", "b", "c", "d"]}),
            "non_numeric_column:Y",
        ),
        (
            pd.DataFrame({"T": [0, 0, 1, 1], "Y": [1.0, np.nan, 2.0, 3.0]}),
            "missing_or_non_finite_values",
        ),
        (
            pd.DataFrame({"T": [0, 0, 2, 2], "Y": [1.0, 2.0, 3.0, 4.0]}),
            "treatment_must_be_binary_zero_one",
        ),
        (
            pd.DataFrame({"T": [0, 0, 0, 1], "Y": [1.0, 2.0, 3.0, 4.0]}),
            "each_treatment_group_requires_two_observations",
        ),
        (
            pd.DataFrame({"T": [0, 1, 1], "Y": [1.0, 2.0, 3.0]}),
            "at_least_four_observations_are_required",
        ),
    ],
)
def test_invalid_data_is_rejected(frame, reason):
    result = estimate_effect(frame, "T", "Y", identified_no_adjustment())
    assert result.status is EstimationStatus.INVALID_DATA
    assert reason in result.reasons


def test_missing_adjustment_column_is_rejected():
    result = estimate_effect(simple_data(), "T", "Y", identified_with_x())
    assert result.reasons == ("missing_column:X",)


def test_treatment_and_outcome_arguments_must_be_distinct():
    result = estimate_effect(simple_data(), "T", "T", identified_no_adjustment())
    assert result.status is EstimationStatus.INVALID_DATA
    assert result.reasons == ("treatment_and_outcome_must_be_distinct",)


def test_defensive_check_rejects_treatment_as_adjustment():
    identification = IdentificationResult(
        status=IdentificationStatus.IDENTIFIED,
        estimand="ATE",
        method="backdoor_adjustment",
        adjustment_sets=(("T",),),
        conditioning_variables=(),
        expression="expression",
        reasons=(),
        assumptions=tuple(sorted(REQUIRED_ASSUMPTIONS)),
    )
    result = estimate_effect(simple_data(), "T", "Y", identification)
    assert result.status is EstimationStatus.INVALID_DATA
    assert result.reasons == ("invalid_adjustment_target:T",)


def test_non_numeric_adjustment_column_is_rejected():
    frame = simple_data().assign(X=["a", "b", "c", "d", "e", "f"])
    result = estimate_effect(frame, "T", "Y", identified_with_x())
    assert result.reasons == ("non_numeric_column:X",)


def test_rank_deficient_outcome_model_is_rejected():
    frame = confounded_data(100).assign(X2=lambda item: item["X"])
    identification = IdentificationResult(
        status=IdentificationStatus.IDENTIFIED,
        estimand="ATE",
        method="backdoor_adjustment",
        adjustment_sets=(("X", "X2"),),
        conditioning_variables=(),
        expression="expression",
        reasons=(),
        assumptions=tuple(sorted(REQUIRED_ASSUMPTIONS)),
    )
    result = estimate_effect(frame, "T", "Y", identification)
    assert result.status is EstimationStatus.INVALID_DATA
    assert result.reasons == ("outcome_design_matrix_is_rank_deficient",)


@pytest.mark.parametrize("method", ["g_computation", "ipw", "aipw"])
def test_severe_propensity_separation_blocks_estimation(method):
    x = np.concatenate((np.full(50, -100.0), np.full(50, 100.0)))
    frame = pd.DataFrame(
        {
            "X": x,
            "T": np.concatenate((np.zeros(50), np.ones(50))),
            "Y": 1.0 + 2.0 * np.concatenate((np.zeros(50), np.ones(50))) + x,
        }
    )
    result = estimate_effect(
        frame,
        "T",
        "Y",
        identified_with_x(),
        EstimationRequest(method=method, max_clipped_fraction=0.1),
    )
    assert result.status is EstimationStatus.POSITIVITY_VIOLATION
    assert result.reasons == ("propensity_overlap_below_configured_threshold",)
    assert result.diagnostics["propensity_fraction_clipped"] > 0.1


def test_aipw_without_covariates_uses_group_outcome_means():
    result = estimate_effect(
        simple_data(),
        "T",
        "Y",
        identified_no_adjustment(),
        EstimationRequest(method="aipw"),
    )
    assert result.estimate == pytest.approx(2.0)


def test_ipw_without_covariates_uses_marginal_propensity():
    result = estimate_effect(
        simple_data(),
        "T",
        "Y",
        identified_no_adjustment(),
        EstimationRequest(method="ipw"),
    )
    assert result.status is EstimationStatus.ESTIMATED
    assert result.diagnostics["propensity_min"] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [None, [], {}])
def test_invalid_data_type_fails_explicitly(value):
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        estimate_effect(value, "T", "Y", identified_no_adjustment())


def test_invalid_identification_type_fails_explicitly():
    with pytest.raises(TypeError, match="identification must be an IdentificationResult"):
        estimate_effect(simple_data(), "T", "Y", {})


def test_invalid_request_type_fails_explicitly():
    with pytest.raises(TypeError, match="request must be an EstimationRequest"):
        estimate_effect(simple_data(), "T", "Y", identified_no_adjustment(), {})


def test_default_request_is_created_when_omitted():
    result = estimate_effect(simple_data(), "T", "Y", identified_no_adjustment())
    assert result.method == "g_computation"


def test_result_serialization_is_json_compatible():
    result = estimate_effect(
        simple_data(),
        "T",
        "Y",
        identified_no_adjustment(),
        EstimationRequest(method="difference_in_means"),
    )
    data = result.to_dict()
    assert data["status"] == "ESTIMATED"
    assert isinstance(data["confidence_interval"], list)
    assert data["adjustment_set"] == []
    assert data["requires_human_review"] is True


def test_blocked_result_serialization_preserves_null_estimate():
    blocked = EffectEstimate(
        status=EstimationStatus.INVALID_DATA,
        estimand="ATE",
        method="ipw",
        estimate=None,
        standard_error=None,
        confidence_interval=None,
        n_observations=0,
        adjustment_set=(),
        diagnostics={},
        reasons=("reason",),
    )
    assert blocked.to_dict()["confidence_interval"] is None
