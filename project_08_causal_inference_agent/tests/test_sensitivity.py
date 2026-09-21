from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import causal_audit_agent.sensitivity as module
from causal_audit_agent.estimation import EffectEstimate, EstimationStatus
from causal_audit_agent.sensitivity import (
    SensitivityRequest,
    SensitivityStatus,
    analyze_hidden_confounding,
)


def effect(**changes):
    values = {
        "status": EstimationStatus.ESTIMATED,
        "estimand": "ATE",
        "method": "g_computation",
        "estimate": 2.0,
        "standard_error": 0.10,
        "confidence_interval": (1.80, 2.20),
        "n_observations": 200,
        "adjustment_set": ("X",),
        "diagnostics": {},
        "reasons": (),
    }
    values.update(changes)
    return EffectEstimate(**values)


def request(**changes):
    values = {
        "analysis": "linear_ovb",
        "null_value": 0.0,
        "reduce_fraction": 1.0,
        "robustness_threshold": 0.10,
        "confounding_scenarios": ((0.01, 0.01), (0.10, 0.10)),
        "e_value_threshold": 2.0,
    }
    values.update(changes)
    return SensitivityRequest(**values)


def test_strong_linear_effect_is_robust_and_reports_bounds():
    report = analyze_hidden_confounding(effect(), request())
    assert report.status is SensitivityStatus.ROBUST
    assert report.robust
    assert report.analysis == "linear_ovb"
    assert report.estimand == "ATE"
    assert report.method == "g_computation"
    assert report.observed_estimate == 2.0
    assert report.standard_error == 0.10
    assert report.degrees_of_freedom == 197
    assert report.robustness_value > 0.10
    assert report.e_value is None
    assert report.confidence_interval_e_value is None
    assert len(report.scenarios) == 2
    assert report.reasons == ("robustness_value_at_or_above_threshold",)
    assert report.requires_human_review


def test_scenario_bias_uses_partial_r2_identity():
    report = analyze_hidden_confounding(
        effect(estimate=0.20, standard_error=0.10, confidence_interval=(0.01, 0.39)),
        request(confounding_scenarios=((0.10, 0.20),)),
    )
    scenario = report.scenarios[0]
    expected = 0.10 * np.sqrt(197 * 0.20 * 0.10 / 0.90)
    assert scenario.treatment_partial_r2 == 0.10
    assert scenario.outcome_partial_r2 == 0.20
    assert scenario.absolute_bias_bound == pytest.approx(expected)
    assert scenario.adjusted_effect_lower == pytest.approx(0.20 - expected)
    assert scenario.adjusted_effect_upper == pytest.approx(0.20 + expected)
    assert scenario.crosses_null


def test_weak_linear_effect_is_sensitive():
    report = analyze_hidden_confounding(
        effect(estimate=0.10, standard_error=0.20, confidence_interval=(0.01, 0.19)),
        request(robustness_threshold=0.10),
    )
    assert report.status is SensitivityStatus.SENSITIVE
    assert not report.robust
    assert report.robustness_value < 0.10
    assert report.reasons == ("robustness_value_below_threshold",)


def test_linear_confidence_interval_crossing_null_is_inconclusive():
    report = analyze_hidden_confounding(
        effect(confidence_interval=(-0.10, 2.20)), request()
    )
    assert report.status is SensitivityStatus.INCONCLUSIVE
    assert report.reasons == ("baseline_confidence_interval_crosses_null",)


def test_missing_linear_confidence_interval_does_not_block_robustness_value():
    report = analyze_hidden_confounding(effect(confidence_interval=None), request())
    assert report.status is SensitivityStatus.ROBUST


def test_custom_null_and_fraction_change_the_robustness_value():
    full = analyze_hidden_confounding(effect(), request())
    partial = analyze_hidden_confounding(
        effect(), request(null_value=1.0, reduce_fraction=0.50)
    )
    assert partial.robustness_value < full.robustness_value


@pytest.mark.parametrize("estimand", ["ATE", "ATT"])
@pytest.mark.parametrize("method", ["g_computation", "linear_regression", "ols"])
def test_supported_linear_estimands_and_methods(estimand, method):
    report = analyze_hidden_confounding(
        effect(estimand=estimand, method=method), request()
    )
    assert report.status is SensitivityStatus.ROBUST


def test_risk_ratio_e_value_above_one_is_robust():
    report = analyze_hidden_confounding(
        effect(
            estimand="RR",
            method="aipw",
            estimate=3.0,
            standard_error=None,
            confidence_interval=(2.0, 4.0),
        ),
        request(analysis="e_value", null_value=1.0, e_value_threshold=2.0),
    )
    assert report.status is SensitivityStatus.ROBUST
    assert report.e_value == pytest.approx(3.0 + np.sqrt(6.0))
    assert report.confidence_interval_e_value == pytest.approx(2.0 + np.sqrt(2.0))
    assert report.standard_error is None
    assert report.degrees_of_freedom is None
    assert report.robustness_value is None
    assert report.scenarios == ()
    assert report.reasons == (
        "confidence_interval_e_value_at_or_above_threshold",
    )


def test_protective_risk_ratio_is_inverted_for_e_value():
    report = analyze_hidden_confounding(
        effect(
            estimand="RISK_RATIO",
            estimate=0.25,
            confidence_interval=(0.10, 0.50),
        ),
        request(analysis="e_value", null_value=1.0),
    )
    assert report.status is SensitivityStatus.ROBUST
    assert report.e_value == pytest.approx(4.0 + np.sqrt(12.0))
    assert report.confidence_interval_e_value == pytest.approx(2.0 + np.sqrt(2.0))


def test_e_value_below_threshold_is_sensitive():
    report = analyze_hidden_confounding(
        effect(estimand="RR", estimate=1.40, confidence_interval=(1.10, 1.80)),
        request(analysis="e_value", null_value=1.0, e_value_threshold=2.0),
    )
    assert report.status is SensitivityStatus.SENSITIVE
    assert report.reasons == ("confidence_interval_e_value_below_threshold",)


def test_e_value_interval_crossing_one_is_inconclusive():
    report = analyze_hidden_confounding(
        effect(estimand="RR", estimate=1.50, confidence_interval=(0.90, 2.10)),
        request(analysis="e_value", null_value=1.0),
    )
    assert report.status is SensitivityStatus.INCONCLUSIVE
    assert report.confidence_interval_e_value == 1.0
    assert report.reasons == ("baseline_confidence_interval_crosses_null",)


def test_e_value_without_interval_is_inconclusive_but_reports_point_value():
    report = analyze_hidden_confounding(
        effect(estimand="RR", estimate=2.0, confidence_interval=None),
        request(analysis="e_value", null_value=1.0),
    )
    assert report.status is SensitivityStatus.INCONCLUSIVE
    assert report.e_value == pytest.approx(2.0 + np.sqrt(2.0))
    assert report.confidence_interval_e_value is None
    assert report.reasons == ("confidence_interval_unavailable",)


def test_exact_null_risk_ratio_has_e_value_one():
    report = analyze_hidden_confounding(
        effect(estimand="RR", estimate=1.0, confidence_interval=(1.0, 1.0)),
        request(analysis="e_value", null_value=1.0),
    )
    assert report.e_value == 1.0
    assert report.status is SensitivityStatus.INCONCLUSIVE


def test_blocked_baseline_stops_sensitivity_analysis():
    blocked = effect(
        status=EstimationStatus.INVALID_DATA,
        estimate=None,
        standard_error=None,
        confidence_interval=None,
    )
    report = analyze_hidden_confounding(blocked, request())
    assert report.status is SensitivityStatus.BLOCKED
    assert report.observed_estimate is None
    assert report.standard_error is None
    assert report.scenarios == ()
    assert report.reasons == ("baseline_estimation_status:INVALID_DATA",)


def test_report_serialization_is_json_compatible():
    payload = analyze_hidden_confounding(effect(), request()).to_dict()
    assert payload["status"] == "ROBUST"
    assert payload["analysis"] == "linear_ovb"
    assert isinstance(payload["reasons"], list)
    assert isinstance(payload["scenarios"], list)
    assert payload["scenarios"][0]["crosses_null"] is False


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"analysis": "unknown"}, "unsupported_analysis:unknown"),
        ({"analysis": 7}, "unsupported_analysis:<blank>"),
        ({"null_value": np.nan}, "null_value_must_be_finite"),
        ({"null_value": True}, "null_value_must_be_finite"),
        ({"reduce_fraction": 0.0}, "reduce_fraction_must_be_in_zero_one_interval"),
        ({"reduce_fraction": 1.01}, "reduce_fraction_must_be_in_zero_one_interval"),
        ({"reduce_fraction": True}, "reduce_fraction_must_be_in_zero_one_interval"),
        ({"robustness_threshold": 0.0}, "robustness_threshold_must_be_in_zero_one_interval"),
        ({"robustness_threshold": 1.0}, "robustness_threshold_must_be_in_zero_one_interval"),
        ({"e_value_threshold": 0.99}, "e_value_threshold_must_be_at_least_one"),
        ({"e_value_threshold": np.inf}, "e_value_threshold_must_be_at_least_one"),
        ({"confounding_scenarios": ()}, "confounding_scenarios_must_be_a_non_empty_sequence"),
        ({"confounding_scenarios": []}, "confounding_scenarios_must_be_a_non_empty_sequence"),
        ({"confounding_scenarios": "bad"}, "confounding_scenarios_must_be_a_non_empty_sequence"),
        ({"confounding_scenarios": ((0.1,),)}, "invalid_confounding_scenario:0"),
        ({"confounding_scenarios": ("bad",)}, "invalid_confounding_scenario:0"),
        ({"confounding_scenarios": ((-0.1, 0.1),)}, "scenario_partial_r2_out_of_range:0"),
        ({"confounding_scenarios": ((0.1, 1.0),)}, "scenario_partial_r2_out_of_range:0"),
        ({"confounding_scenarios": ((False, 0.1),)}, "scenario_partial_r2_out_of_range:0"),
    ],
)
def test_invalid_requests_are_rejected(changes, reason):
    report = analyze_hidden_confounding(effect(), request(**changes))
    assert report.status is SensitivityStatus.INVALID_REQUEST
    assert reason in report.reasons
    assert report.scenarios == ()


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"estimate": np.nan}, "estimated_effect_must_be_finite"),
        ({"estimand": "CATE"}, "linear_ovb_requires_additive_estimand:CATE"),
        ({"estimand": 5}, "linear_ovb_requires_additive_estimand:<blank>"),
        ({"method": "aipw"}, "linear_ovb_unsupported_method:aipw"),
        ({"method": 5}, "linear_ovb_unsupported_method:<blank>"),
        ({"standard_error": None}, "linear_ovb_requires_positive_standard_error"),
        ({"standard_error": 0.0}, "linear_ovb_requires_positive_standard_error"),
        ({"n_observations": 0}, "n_observations_must_be_a_positive_integer"),
        ({"n_observations": True}, "n_observations_must_be_a_positive_integer"),
        ({"n_observations": 4, "adjustment_set": ("X",)}, "insufficient_residual_degrees_of_freedom"),
    ],
)
def test_invalid_linear_effects_are_rejected(changes, reason):
    report = analyze_hidden_confounding(effect(**changes), request())
    assert report.status is SensitivityStatus.INVALID_REQUEST
    assert reason in report.reasons


@pytest.mark.parametrize(
    ("changes", "request_changes", "reason"),
    [
        ({"estimand": "ATE"}, {}, "e_value_requires_risk_ratio_estimand:ATE"),
        ({"estimand": 4}, {}, "e_value_requires_risk_ratio_estimand:<blank>"),
        ({"estimand": "RR", "estimate": 0.0}, {}, "risk_ratio_must_be_positive"),
        ({"estimand": "RR"}, {"null_value": 0.0}, "e_value_requires_unit_null"),
        ({"estimand": "RR", "confidence_interval": [1.0, 2.0]}, {}, "risk_ratio_confidence_interval_is_invalid"),
        ({"estimand": "RR", "confidence_interval": (2.0,)}, {}, "risk_ratio_confidence_interval_is_invalid"),
        ({"estimand": "RR", "confidence_interval": (0.0, 2.0)}, {}, "risk_ratio_confidence_interval_is_invalid"),
        ({"estimand": "RR", "confidence_interval": (2.0, 1.0)}, {}, "risk_ratio_confidence_interval_is_invalid"),
    ],
)
def test_invalid_e_value_effects_are_rejected(changes, request_changes, reason):
    values = {"analysis": "e_value", "null_value": 1.0}
    values.update(request_changes)
    report = analyze_hidden_confounding(
        effect(**changes),
        request(**values),
    )
    assert report.status is SensitivityStatus.INVALID_REQUEST
    assert reason in report.reasons


def test_type_contracts_are_enforced():
    with pytest.raises(TypeError, match="effect must be an EffectEstimate"):
        analyze_hidden_confounding({})
    with pytest.raises(TypeError, match="request must be a SensitivityRequest"):
        analyze_hidden_confounding(effect(), {})


def test_default_request_is_used():
    assert analyze_hidden_confounding(effect()).analysis == "linear_ovb"


def test_empty_report_normalizes_non_string_labels_and_invalid_numbers():
    invalid = effect(
        estimand=7,
        method=9,
        estimate=np.nan,
        standard_error=np.inf,
    )
    report = analyze_hidden_confounding(
        invalid, request(null_value=np.nan, analysis="bad")
    )
    assert report.estimand == ""
    assert report.method == ""
    assert report.observed_estimate is None
    assert report.standard_error is None
    assert report.null_value == 0.0


def test_configuration_and_protocol_are_present():
    root = Path(__file__).parents[1]
    configuration = yaml.safe_load(
        (root / "configs/sensitivity/default.yaml").read_text(encoding="utf-8")
    )
    assert configuration["analysis"] == "linear_ovb"
    assert configuration["robustness_threshold"] == 0.10
    assert len(configuration["confounding_scenarios"]) == 4
    configured_report = analyze_hidden_confounding(
        effect(), SensitivityRequest(**configuration)
    )
    assert configured_report.status is SensitivityStatus.ROBUST
    protocol = (root / "docs/sensitivity_protocol.md").read_text(encoding="utf-8")
    assert "does not claim that hidden confounding is absent" in protocol
    assert "partial R-squared" in protocol
    assert "risk ratios" in protocol
    assert "10.1111/rssb.12348" in protocol
    assert "10.7326/M16-2607" in protocol


def test_internal_interval_helper_covers_closed_right_endpoint():
    assert module._number_in_interval(1.0, 0.0, 1.0, right_closed=True)
    assert not module._number_in_interval("bad", 0.0, 1.0)
