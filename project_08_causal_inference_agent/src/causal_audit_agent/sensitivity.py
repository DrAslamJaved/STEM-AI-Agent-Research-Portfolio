from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import sqrt
from typing import Any

import numpy as np

from causal_audit_agent.estimation import EffectEstimate


class SensitivityStatus(str, Enum):
    ROBUST = "ROBUST"
    SENSITIVE = "SENSITIVE"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED = "BLOCKED"
    INVALID_REQUEST = "INVALID_REQUEST"


@dataclass(frozen=True)
class SensitivityRequest:
    analysis: str = "linear_ovb"
    null_value: float = 0.0
    reduce_fraction: float = 1.0
    robustness_threshold: float = 0.10
    confounding_scenarios: tuple[tuple[float, float], ...] = (
        (0.01, 0.01),
        (0.05, 0.05),
        (0.10, 0.10),
        (0.20, 0.20),
    )
    e_value_threshold: float = 2.0


@dataclass(frozen=True)
class SensitivityScenario:
    treatment_partial_r2: float
    outcome_partial_r2: float
    absolute_bias_bound: float
    adjusted_effect_lower: float
    adjusted_effect_upper: float
    crosses_null: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SensitivityReport:
    status: SensitivityStatus
    analysis: str
    estimand: str
    method: str
    observed_estimate: float | None
    null_value: float
    standard_error: float | None
    degrees_of_freedom: int | None
    robustness_value: float | None
    e_value: float | None
    confidence_interval_e_value: float | None
    scenarios: tuple[SensitivityScenario, ...]
    reasons: tuple[str, ...]
    requires_human_review: bool = True

    @property
    def robust(self) -> bool:
        return self.status is SensitivityStatus.ROBUST

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "analysis": self.analysis,
            "estimand": self.estimand,
            "method": self.method,
            "observed_estimate": self.observed_estimate,
            "null_value": self.null_value,
            "standard_error": self.standard_error,
            "degrees_of_freedom": self.degrees_of_freedom,
            "robustness_value": self.robustness_value,
            "e_value": self.e_value,
            "confidence_interval_e_value": self.confidence_interval_e_value,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "reasons": list(self.reasons),
            "requires_human_review": self.requires_human_review,
        }


def analyze_hidden_confounding(
    effect: EffectEstimate,
    request: SensitivityRequest | None = None,
) -> SensitivityReport:
    """Quantify sensitivity to an omitted common cause.

    ``linear_ovb`` uses the partial-R-squared omitted-variable-bias identity
    for an additive linear treatment coefficient. ``e_value`` is restricted
    to effects already expressed as risk ratios. Neither analysis proves that
    hidden confounding is absent.
    """
    if not isinstance(effect, EffectEstimate):
        raise TypeError("effect must be an EffectEstimate")
    if request is None:
        request = SensitivityRequest()
    if not isinstance(request, SensitivityRequest):
        raise TypeError("request must be a SensitivityRequest")

    analysis = request.analysis.casefold() if isinstance(request.analysis, str) else ""
    request_errors = _validate_request(request, analysis)
    if request_errors:
        return _empty_report(
            SensitivityStatus.INVALID_REQUEST,
            effect,
            request,
            analysis,
            request_errors,
        )
    if not effect.estimated:
        return _empty_report(
            SensitivityStatus.BLOCKED,
            effect,
            request,
            analysis,
            (f"baseline_estimation_status:{effect.status.value}",),
        )

    effect_errors = _validate_effect(effect, request, analysis)
    if effect_errors:
        return _empty_report(
            SensitivityStatus.INVALID_REQUEST,
            effect,
            request,
            analysis,
            effect_errors,
        )
    if analysis == "linear_ovb":
        return _linear_ovb_report(effect, request)
    return _e_value_report(effect, request)


def _validate_request(
    request: SensitivityRequest, analysis: str
) -> tuple[str, ...]:
    errors: list[str] = []
    if analysis not in {"linear_ovb", "e_value"}:
        errors.append(f"unsupported_analysis:{analysis or '<blank>'}")
    if not _finite_number(request.null_value):
        errors.append("null_value_must_be_finite")
    if not _number_in_interval(
        request.reduce_fraction, 0.0, 1.0, left_open=True, right_closed=True
    ):
        errors.append("reduce_fraction_must_be_in_zero_one_interval")
    if not _number_in_interval(
        request.robustness_threshold, 0.0, 1.0, left_open=True
    ):
        errors.append("robustness_threshold_must_be_in_zero_one_interval")
    if not _finite_number(request.e_value_threshold) or request.e_value_threshold < 1.0:
        errors.append("e_value_threshold_must_be_at_least_one")
    scenarios = request.confounding_scenarios
    if not isinstance(scenarios, (tuple, list)) or not scenarios:
        errors.append("confounding_scenarios_must_be_a_non_empty_sequence")
    else:
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, (tuple, list)) or len(scenario) != 2:
                errors.append(f"invalid_confounding_scenario:{index}")
                continue
            if any(
                not _number_in_interval(value, 0.0, 1.0)
                for value in scenario
            ):
                errors.append(f"scenario_partial_r2_out_of_range:{index}")
    return tuple(errors)


def _validate_effect(
    effect: EffectEstimate,
    request: SensitivityRequest,
    analysis: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not _finite_number(effect.estimate):
        errors.append("estimated_effect_must_be_finite")
    estimand = effect.estimand.upper() if isinstance(effect.estimand, str) else ""
    method = effect.method.casefold() if isinstance(effect.method, str) else ""
    if analysis == "linear_ovb":
        if estimand not in {"ATE", "ATT"}:
            errors.append(f"linear_ovb_requires_additive_estimand:{estimand or '<blank>'}")
        if method not in {"g_computation", "linear_regression", "ols"}:
            errors.append(f"linear_ovb_unsupported_method:{method or '<blank>'}")
        if not _finite_number(effect.standard_error) or effect.standard_error <= 0.0:
            errors.append("linear_ovb_requires_positive_standard_error")
        if (
            not isinstance(effect.n_observations, int)
            or isinstance(effect.n_observations, bool)
            or effect.n_observations <= 0
        ):
            errors.append("n_observations_must_be_a_positive_integer")
        elif effect.n_observations - len(effect.adjustment_set) - 2 < 2:
            errors.append("insufficient_residual_degrees_of_freedom")
    else:
        if estimand not in {"RR", "RISK_RATIO"}:
            errors.append(f"e_value_requires_risk_ratio_estimand:{estimand or '<blank>'}")
        if _finite_number(effect.estimate) and effect.estimate <= 0.0:
            errors.append("risk_ratio_must_be_positive")
        if request.null_value != 1.0:
            errors.append("e_value_requires_unit_null")
        if effect.confidence_interval is not None:
            interval = effect.confidence_interval
            if (
                not isinstance(interval, tuple)
                or len(interval) != 2
                or any(not _finite_number(value) or value <= 0.0 for value in interval)
                or interval[0] > interval[1]
            ):
                errors.append("risk_ratio_confidence_interval_is_invalid")
    return tuple(errors)


def _linear_ovb_report(
    effect: EffectEstimate, request: SensitivityRequest
) -> SensitivityReport:
    estimate = float(effect.estimate)
    standard_error = float(effect.standard_error)
    degrees_of_freedom = effect.n_observations - len(effect.adjustment_set) - 2
    distance = abs(estimate - request.null_value)
    t_value = distance / standard_error
    robustness_value = _robustness_value(
        t_value, degrees_of_freedom, request.reduce_fraction
    )
    scenarios = tuple(
        _scenario(
            estimate,
            request.null_value,
            standard_error,
            degrees_of_freedom,
            treatment_r2,
            outcome_r2,
        )
        for treatment_r2, outcome_r2 in request.confounding_scenarios
    )

    if _interval_crosses_null(effect.confidence_interval, request.null_value):
        status = SensitivityStatus.INCONCLUSIVE
        reasons = ("baseline_confidence_interval_crosses_null",)
    elif robustness_value >= request.robustness_threshold:
        status = SensitivityStatus.ROBUST
        reasons = ("robustness_value_at_or_above_threshold",)
    else:
        status = SensitivityStatus.SENSITIVE
        reasons = ("robustness_value_below_threshold",)
    return SensitivityReport(
        status=status,
        analysis="linear_ovb",
        estimand=effect.estimand.upper(),
        method=effect.method.casefold(),
        observed_estimate=estimate,
        null_value=float(request.null_value),
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        robustness_value=robustness_value,
        e_value=None,
        confidence_interval_e_value=None,
        scenarios=scenarios,
        reasons=reasons,
    )


def _e_value_report(
    effect: EffectEstimate, request: SensitivityRequest
) -> SensitivityReport:
    estimate = float(effect.estimate)
    point_e_value = _e_value(estimate)
    interval_e_value = _confidence_interval_e_value(effect.confidence_interval)
    if interval_e_value is None:
        status = SensitivityStatus.INCONCLUSIVE
        reasons = ("confidence_interval_unavailable",)
    elif interval_e_value == 1.0:
        status = SensitivityStatus.INCONCLUSIVE
        reasons = ("baseline_confidence_interval_crosses_null",)
    elif interval_e_value >= request.e_value_threshold:
        status = SensitivityStatus.ROBUST
        reasons = ("confidence_interval_e_value_at_or_above_threshold",)
    else:
        status = SensitivityStatus.SENSITIVE
        reasons = ("confidence_interval_e_value_below_threshold",)
    return SensitivityReport(
        status=status,
        analysis="e_value",
        estimand=effect.estimand.upper(),
        method=effect.method.casefold(),
        observed_estimate=estimate,
        null_value=1.0,
        standard_error=None,
        degrees_of_freedom=None,
        robustness_value=None,
        e_value=point_e_value,
        confidence_interval_e_value=interval_e_value,
        scenarios=(),
        reasons=reasons,
    )


def _scenario(
    estimate: float,
    null_value: float,
    standard_error: float,
    degrees_of_freedom: int,
    treatment_r2: float,
    outcome_r2: float,
) -> SensitivityScenario:
    bias = standard_error * sqrt(
        degrees_of_freedom * outcome_r2 * treatment_r2 / (1.0 - treatment_r2)
    )
    lower, upper = estimate - bias, estimate + bias
    return SensitivityScenario(
        treatment_partial_r2=float(treatment_r2),
        outcome_partial_r2=float(outcome_r2),
        absolute_bias_bound=float(bias),
        adjusted_effect_lower=float(lower),
        adjusted_effect_upper=float(upper),
        crosses_null=lower <= null_value <= upper,
    )


def _robustness_value(t_value: float, degrees_of_freedom: int, fraction: float) -> float:
    f_value = fraction * t_value / sqrt(degrees_of_freedom)
    squared = f_value * f_value
    return float(0.5 * (sqrt(squared * squared + 4.0 * squared) - squared))


def _e_value(risk_ratio: float) -> float:
    harmful_scale = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return float(harmful_scale + sqrt(harmful_scale * (harmful_scale - 1.0)))


def _confidence_interval_e_value(
    interval: tuple[float, float] | None,
) -> float | None:
    if interval is None:
        return None
    lower, upper = interval
    if lower <= 1.0 <= upper:
        return 1.0
    closest = lower if lower > 1.0 else 1.0 / upper
    return _e_value(closest)


def _interval_crosses_null(
    interval: tuple[float, float] | None, null_value: float
) -> bool:
    return interval is not None and interval[0] <= null_value <= interval[1]


def _empty_report(
    status: SensitivityStatus,
    effect: EffectEstimate,
    request: SensitivityRequest,
    analysis: str,
    reasons: tuple[str, ...],
) -> SensitivityReport:
    estimate = float(effect.estimate) if _finite_number(effect.estimate) else None
    standard_error = (
        float(effect.standard_error) if _finite_number(effect.standard_error) else None
    )
    return SensitivityReport(
        status=status,
        analysis=analysis,
        estimand=effect.estimand.upper() if isinstance(effect.estimand, str) else "",
        method=effect.method.casefold() if isinstance(effect.method, str) else "",
        observed_estimate=estimate,
        null_value=float(request.null_value) if _finite_number(request.null_value) else 0.0,
        standard_error=standard_error,
        degrees_of_freedom=None,
        robustness_value=None,
        e_value=None,
        confidence_interval_e_value=None,
        scenarios=(),
        reasons=reasons,
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _number_in_interval(
    value: object,
    lower: float,
    upper: float,
    *,
    left_open: bool = False,
    right_closed: bool = False,
) -> bool:
    if not _finite_number(value):
        return False
    left_valid = value > lower if left_open else value >= lower
    right_valid = value <= upper if right_closed else value < upper
    return bool(left_valid and right_valid)
