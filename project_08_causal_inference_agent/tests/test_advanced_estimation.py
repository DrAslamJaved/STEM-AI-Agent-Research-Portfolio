from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from causal_audit_agent.advanced_estimation import (
    AdvancedEffectEstimate,
    AdvancedEstimationRequest,
    AdvancedEstimationStatus,
    estimate_advanced_effect,
)
from causal_audit_agent.identification import (
    IdentificationResult,
    IdentificationStatus,
)


def identified(
    estimand="ATE",
    adjustment_sets=(("X", "V"),),
    conditioning_variables=(),
):
    return IdentificationResult(
        status=IdentificationStatus.IDENTIFIED,
        estimand=estimand,
        method="backdoor_adjustment",
        adjustment_sets=adjustment_sets,
        conditioning_variables=conditioning_variables,
        expression="identified expression",
        reasons=("backdoor_criterion_satisfied",),
        assumptions=("consistency", "exchangeability", "positivity"),
    )


def observational_data(n=360, seed=13):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    v = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.35 * x + 0.25 * v)))
    treatment = rng.binomial(1, propensity)
    true_effect = 2.0 + 1.25 * v
    outcome = 1.0 + 0.8 * x + 0.4 * v + treatment * true_effect
    outcome += rng.normal(scale=0.35, size=n)
    return pd.DataFrame({"X": x, "V": v, "T": treatment, "Y": outcome}), true_effect


def request(method="cross_fitted_aipw", **changes):
    values = {
        "method": method,
        "n_splits": 3,
        "n_estimators": 40,
        "min_samples_leaf": 5,
        "random_state": 17,
    }
    values.update(changes)
    return AdvancedEstimationRequest(**values)


def test_cross_fitted_aipw_recovers_known_average_effect():
    frame, true_effect = observational_data()
    result = estimate_advanced_effect(
        frame, "T", "Y", identified(), request()
    )
    assert result.status is AdvancedEstimationStatus.ESTIMATED
    assert result.estimated
    assert result.method == "cross_fitted_aipw"
    assert result.estimate == pytest.approx(float(true_effect.mean()), abs=0.25)
    assert result.standard_error > 0.0
    assert result.confidence_interval[0] < result.estimate < result.confidence_interval[1]
    assert result.unit_effects is None
    assert result.diagnostics["n_splits"] == 3
    assert result.diagnostics["score_standard_deviation"] > 0.0
    assert result.requires_human_review


def test_forest_dr_learner_recovers_effect_ordering():
    frame, true_effect = observational_data()
    result = estimate_advanced_effect(
        frame,
        "T",
        "Y",
        identified(estimand="CATE", conditioning_variables=("V",)),
        request("dr_learner_forest"),
    )
    effects = np.asarray(result.unit_effects)
    assert result.status is AdvancedEstimationStatus.ESTIMATED
    assert len(effects) == len(frame)
    assert np.corrcoef(effects, true_effect)[0, 1] > 0.70
    assert result.estimate == pytest.approx(float(effects.mean()))
    assert result.standard_error is None
    assert result.confidence_interval is None
    assert result.diagnostics["cate_min"] < result.diagnostics["cate_max"]
    assert result.diagnostics["cate_standard_deviation"] > 0.0
    assert result.diagnostics["importance:V"] == pytest.approx(1.0)
    assert "cate_pointwise_uncertainty_not_estimated" in result.reasons


def test_no_covariate_aipw_uses_fold_specific_group_means():
    rng = np.random.default_rng(5)
    treatment = np.tile([0, 1], 60)
    frame = pd.DataFrame(
        {"T": treatment, "Y": 1.0 + 2.0 * treatment + rng.normal(0, 0.1, 120)}
    )
    result = estimate_advanced_effect(
        frame,
        "T",
        "Y",
        identified(adjustment_sets=((),)),
        request(n_splits=5),
    )
    assert result.estimate == pytest.approx(2.0, abs=0.08)
    assert result.diagnostics["propensity_min"] == pytest.approx(0.5)
    assert result.adjustment_set == ()


def test_fixed_seed_is_reproducible():
    frame, _ = observational_data(240)
    identification = identified(estimand="CATE", conditioning_variables=("V",))
    first = estimate_advanced_effect(
        frame, "T", "Y", identification, request("dr_learner_forest")
    )
    second = estimate_advanced_effect(
        frame, "T", "Y", identification, request("dr_learner_forest")
    )
    assert first.to_dict() == second.to_dict()


def test_estimated_result_serialization_is_json_compatible():
    frame, _ = observational_data(240)
    result = estimate_advanced_effect(frame, "T", "Y", identified(), request())
    payload = result.to_dict()
    assert payload["status"] == "ESTIMATED"
    assert isinstance(payload["confidence_interval"], list)
    assert payload["unit_effects"] is None
    assert payload["adjustment_set"] == ["V", "X"]
    assert payload["conditioning_variables"] == []


def test_cate_serialization_contains_unit_effects_and_null_interval():
    frame, _ = observational_data(240)
    result = estimate_advanced_effect(
        frame,
        "T",
        "Y",
        identified(estimand="CATE", conditioning_variables=("V",)),
        request("dr_learner_forest"),
    )
    payload = result.to_dict()
    assert len(payload["unit_effects"]) == len(frame)
    assert payload["confidence_interval"] is None
    assert payload["conditioning_variables"] == ["V"]


@pytest.mark.parametrize(
    "status",
    [
        IdentificationStatus.NON_IDENTIFIABLE,
        IdentificationStatus.INSUFFICIENT_ASSUMPTIONS,
        IdentificationStatus.INVALID_QUERY,
        IdentificationStatus.UNSUPPORTED_QUERY,
    ],
)
def test_nonidentified_requests_are_blocked(status):
    frame, _ = observational_data(120)
    identification = replace(identified(), status=status)
    result = estimate_advanced_effect(frame, "T", "Y", identification, request())
    assert result.status is AdvancedEstimationStatus.BLOCKED_IDENTIFICATION
    assert result.reasons == (f"identification_status:{status.value}",)
    assert not result.estimated


@pytest.mark.parametrize(
    ("method", "estimand"),
    [
        ("cross_fitted_aipw", "ATT"),
        ("cross_fitted_aipw", "CATE"),
        ("dr_learner_forest", "ATE"),
        ("dr_learner_forest", "ATT"),
    ],
)
def test_unsupported_method_estimand_pairs_are_rejected(method, estimand):
    frame, _ = observational_data(120)
    result = estimate_advanced_effect(
        frame, "T", "Y", identified(estimand=estimand), request(method)
    )
    assert result.status is AdvancedEstimationStatus.UNSUPPORTED
    assert result.reasons == (
        f"method_estimand_pair_not_supported:{method}:{estimand}",
    )


def test_forest_requires_identified_effect_modifiers():
    frame, _ = observational_data(120)
    result = estimate_advanced_effect(
        frame,
        "T",
        "Y",
        identified(estimand="CATE", conditioning_variables=()),
        request("dr_learner_forest"),
    )
    assert result.status is AdvancedEstimationStatus.INVALID_REQUEST
    assert result.reasons == ("dr_learner_forest_requires_conditioning_variables",)


def test_missing_identified_adjustment_set_blocks_estimation():
    frame, _ = observational_data(120)
    result = estimate_advanced_effect(
        frame, "T", "Y", identified(adjustment_sets=()), request()
    )
    assert result.status is AdvancedEstimationStatus.BLOCKED_IDENTIFICATION
    assert result.reasons == ("identification_has_no_adjustment_set",)


def test_unapproved_adjustment_set_is_rejected():
    frame, _ = observational_data(120)
    result = estimate_advanced_effect(
        frame,
        "T",
        "Y",
        identified(adjustment_sets=(("X",),)),
        request(),
        adjustment_set=("V",),
    )
    assert result.status is AdvancedEstimationStatus.INVALID_REQUEST
    assert result.adjustment_set == ("V",)
    assert result.reasons == ("adjustment_set_not_identified",)


def test_smallest_identified_adjustment_set_is_selected_deterministically():
    frame, _ = observational_data(120)
    result = estimate_advanced_effect(
        frame,
        "T",
        "Y",
        identified(adjustment_sets=(("X", "V"), ("X",))),
        request(),
    )
    assert result.adjustment_set == ("X",)


def test_severe_separation_triggers_positivity_review():
    x = np.concatenate((np.full(60, -100.0), np.full(60, 100.0)))
    treatment = np.concatenate((np.zeros(60), np.ones(60)))
    frame = pd.DataFrame({"X": x, "T": treatment, "Y": x + 2 * treatment})
    result = estimate_advanced_effect(
        frame,
        "T",
        "Y",
        identified(adjustment_sets=(("X",),)),
        request(max_clipped_fraction=0.05),
    )
    assert result.status is AdvancedEstimationStatus.POSITIVITY_VIOLATION
    assert result.reasons == ("propensity_overlap_below_configured_threshold",)
    assert result.diagnostics["propensity_fraction_clipped"] > 0.05


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"method": "unknown"}, "unsupported_method:unknown"),
        ({"method": ""}, "unsupported_method:<blank>"),
        ({"method": 3}, "unsupported_method:<blank>"),
        ({"confidence_level": 1.0}, "confidence_level_must_be_between_zero_and_one"),
        ({"n_splits": 1}, "n_splits_must_be_an_integer_between_two_and_ten"),
        ({"n_splits": True}, "n_splits_must_be_an_integer_between_two_and_ten"),
        ({"propensity_clip": (0.9, 0.1)}, "invalid_propensity_clip"),
        ({"max_clipped_fraction": 1.1}, "max_clipped_fraction_must_be_in_unit_interval"),
        ({"n_estimators": 9}, "n_estimators_must_be_an_integer_at_least_ten"),
        ({"n_estimators": True}, "n_estimators_must_be_an_integer_at_least_ten"),
        ({"min_samples_leaf": 0}, "min_samples_leaf_must_be_a_positive_integer"),
        ({"min_samples_leaf": True}, "min_samples_leaf_must_be_a_positive_integer"),
        ({"random_state": 1.5}, "random_state_must_be_an_integer"),
        ({"random_state": True}, "random_state_must_be_an_integer"),
    ],
)
def test_invalid_request_settings_are_rejected(changes, reason):
    frame, _ = observational_data(120)
    bad_request = replace(request(), **changes)
    result = estimate_advanced_effect(frame, "T", "Y", identified(), bad_request)
    assert result.status is AdvancedEstimationStatus.INVALID_REQUEST
    assert reason in result.reasons


@pytest.mark.parametrize("value", [None, [], {}])
def test_invalid_data_type_fails_explicitly(value):
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        estimate_advanced_effect(value, "T", "Y", identified(), request())


def test_invalid_identification_type_fails_explicitly():
    frame, _ = observational_data(120)
    with pytest.raises(TypeError, match="identification must be an IdentificationResult"):
        estimate_advanced_effect(frame, "T", "Y", {}, request())


def test_invalid_request_type_fails_explicitly():
    frame, _ = observational_data(120)
    with pytest.raises(TypeError, match="request must be an AdvancedEstimationRequest"):
        estimate_advanced_effect(frame, "T", "Y", identified(), {})


def test_default_request_is_constructed_when_omitted():
    frame, _ = observational_data(120)
    result = estimate_advanced_effect(frame, "T", "Y", identified())
    assert result.method == "cross_fitted_aipw"


@pytest.mark.parametrize(
    ("frame_change", "treatment", "outcome", "adjustment", "reason"),
    [
        (lambda f: f, "T", "T", None, "treatment_and_outcome_must_be_distinct"),
        (lambda f: f, "T", "Y", ("T",), "invalid_covariate:T"),
        (lambda f: f.drop(columns="X"), "T", "Y", None, "missing_column:X"),
        (lambda f: f.assign(X="bad"), "T", "Y", None, "non_numeric_column:X"),
        (
            lambda f: f.assign(X=lambda x: x["X"].mask(x.index == 0, np.nan)),
            "T",
            "Y",
            None,
            "missing_or_non_finite_values",
        ),
        (lambda f: f.assign(T=2), "T", "Y", None, "treatment_must_be_binary_zero_one"),
    ],
)
def test_invalid_analysis_data_is_rejected(
    frame_change, treatment, outcome, adjustment, reason
):
    frame, _ = observational_data(120)
    frame = frame_change(frame)
    identification = (
        identified(adjustment_sets=(adjustment,))
        if adjustment is not None
        else identified()
    )
    result = estimate_advanced_effect(
        frame, treatment, outcome, identification, request()
    )
    assert result.status is AdvancedEstimationStatus.INVALID_DATA
    assert reason in result.reasons


def test_each_group_must_support_cross_fitting():
    frame = pd.DataFrame(
        {
            "X": np.arange(24.0),
            "V": np.arange(24.0),
            "T": [0] * 19 + [1] * 5,
            "Y": np.arange(24.0),
        }
    )
    result = estimate_advanced_effect(
        frame, "T", "Y", identified(), request(n_splits=3)
    )
    assert result.status is AdvancedEstimationStatus.INVALID_DATA
    assert result.reasons == (
        "each_treatment_group_requires_at_least_twice_n_splits_observations",
    )


def test_sample_must_meet_cross_fitting_minimum():
    frame, _ = observational_data(16)
    result = estimate_advanced_effect(
        frame, "T", "Y", identified(), request(n_splits=2)
    )
    assert result.status is AdvancedEstimationStatus.INVALID_DATA
    assert result.reasons == ("sample_size_below_cross_fitting_minimum",)


def test_blocked_result_serializes_null_outputs():
    result = AdvancedEffectEstimate(
        status=AdvancedEstimationStatus.INVALID_DATA,
        estimand="ATE",
        method="cross_fitted_aipw",
        estimate=None,
        standard_error=None,
        confidence_interval=None,
        n_observations=0,
        adjustment_set=(),
        conditioning_variables=(),
        unit_effects=None,
        diagnostics={},
        reasons=("reason",),
    )
    payload = result.to_dict()
    assert payload["confidence_interval"] is None
    assert payload["unit_effects"] is None
    assert payload["status"] == "INVALID_DATA"


def test_advanced_configuration_is_parseable_and_precise():
    path = Path(__file__).parents[1] / "configs" / "estimation" / "advanced.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["methods"]["cross_fitted_aipw"]["cross_fitted"] is True
    assert config["methods"]["dr_learner_forest"]["generalized_causal_forest"] is False
    assert config["safety"]["report_cate_pointwise_interval"] is False
