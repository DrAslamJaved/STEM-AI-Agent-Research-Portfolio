from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from causal_audit_agent.estimation import (
    EstimationRequest,
    estimate_effect,
)
from causal_audit_agent.identification import IdentificationResult


class RefutationStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    INVALID_REQUEST = "INVALID_REQUEST"


@dataclass(frozen=True)
class RefutationRequest:
    n_simulations: int = 20
    subset_fraction: float = 0.80
    max_relative_shift: float = 0.25
    max_absolute_shift: float = 0.25
    min_effect_scale: float = 0.10
    max_failed_fraction: float = 0.10
    max_placebo_p_value: float = 0.10
    random_common_cause_scale: float = 1.0
    random_state: int = 0


@dataclass(frozen=True)
class RefutationCheck:
    name: str
    status: RefutationStatus
    successful_runs: int
    failed_runs: int
    mean_refuted_estimate: float | None
    mean_absolute_shift: float | None
    max_absolute_shift: float | None
    relative_shift: float | None
    empirical_p_value: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class RefutationReport:
    status: RefutationStatus
    method: str
    estimand: str
    baseline_estimate: float | None
    checks: tuple[RefutationCheck, ...]
    reasons: tuple[str, ...]
    random_state: int
    requires_human_review: bool = True

    @property
    def passed(self) -> bool:
        return self.status is RefutationStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "method": self.method,
            "estimand": self.estimand,
            "baseline_estimate": self.baseline_estimate,
            "checks": [check.to_dict() for check in self.checks],
            "reasons": list(self.reasons),
            "random_state": self.random_state,
            "requires_human_review": self.requires_human_review,
        }


def run_refutation_suite(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    identification: IdentificationResult,
    estimation_request: EstimationRequest | None = None,
    refutation_request: RefutationRequest | None = None,
) -> RefutationReport:
    """Stress-test an identified, successfully estimated ATE or ATT.

    Passing means that the configured refuters did not detect instability. It
    does not prove the DAG or causal assumptions.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(identification, IdentificationResult):
        raise TypeError("identification must be an IdentificationResult")
    if estimation_request is None:
        estimation_request = EstimationRequest()
    if not isinstance(estimation_request, EstimationRequest):
        raise TypeError("estimation_request must be an EstimationRequest")
    if refutation_request is None:
        refutation_request = RefutationRequest()
    if not isinstance(refutation_request, RefutationRequest):
        raise TypeError("refutation_request must be a RefutationRequest")

    method = (
        estimation_request.method.casefold()
        if isinstance(estimation_request.method, str)
        else ""
    )
    estimand = identification.estimand.upper()
    errors = list(_validate_refutation_request(refutation_request))
    if method not in {"g_computation", "ipw", "aipw"}:
        errors.append(f"estimation_method_not_refutable:{method or '<blank>'}")
    if errors:
        return RefutationReport(
            status=RefutationStatus.INVALID_REQUEST,
            method=method,
            estimand=estimand,
            baseline_estimate=None,
            checks=(),
            reasons=tuple(errors),
            random_state=refutation_request.random_state,
        )

    baseline = estimate_effect(
        data, treatment, outcome, identification, estimation_request
    )
    if not baseline.estimated:
        return RefutationReport(
            status=RefutationStatus.BLOCKED,
            method=method,
            estimand=estimand,
            baseline_estimate=None,
            checks=(),
            reasons=(f"baseline_estimation_status:{baseline.status.value}",),
            random_state=refutation_request.random_state,
        )

    observed = float(baseline.estimate)
    seed_sequences = np.random.SeedSequence(
        refutation_request.random_state
    ).spawn(3)
    placebo_values, placebo_failures = _placebo_estimates(
        data,
        treatment,
        outcome,
        identification,
        estimation_request,
        refutation_request,
        np.random.default_rng(seed_sequences[0]),
    )
    common_cause_values, common_cause_failures = _random_common_cause_estimates(
        data,
        treatment,
        outcome,
        identification,
        estimation_request,
        refutation_request,
        np.random.default_rng(seed_sequences[1]),
    )
    subset_values, subset_failures = _subset_estimates(
        data,
        treatment,
        outcome,
        identification,
        estimation_request,
        refutation_request,
        np.random.default_rng(seed_sequences[2]),
    )

    checks = (
        _summarize_check(
            "placebo_treatment",
            observed,
            placebo_values,
            placebo_failures,
            refutation_request,
            placebo=True,
        ),
        _summarize_check(
            "random_common_cause",
            observed,
            common_cause_values,
            common_cause_failures,
            refutation_request,
        ),
        _summarize_check(
            "stratified_subset",
            observed,
            subset_values,
            subset_failures,
            refutation_request,
        ),
    )
    failed_checks = tuple(
        check.name for check in checks if check.status is RefutationStatus.REVIEW
    )
    status = RefutationStatus.PASS if not failed_checks else RefutationStatus.REVIEW
    reasons = (
        ("configured_refuters_detected_no_instability",)
        if status is RefutationStatus.PASS
        else tuple(f"review_required:{name}" for name in failed_checks)
    )
    return RefutationReport(
        status=status,
        method=method,
        estimand=estimand,
        baseline_estimate=observed,
        checks=checks,
        reasons=reasons,
        random_state=refutation_request.random_state,
    )


def _validate_refutation_request(request: RefutationRequest) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(request.n_simulations, int) or isinstance(
        request.n_simulations, bool
    ) or request.n_simulations < 10:
        errors.append("n_simulations_must_be_an_integer_at_least_ten")
    if isinstance(request.subset_fraction, bool) or not isinstance(
        request.subset_fraction, (int, float)
    ) or not (
        0.50 <= request.subset_fraction < 1.0
    ):
        errors.append("subset_fraction_must_be_in_half_open_interval")
    for name in (
        "max_relative_shift",
        "max_absolute_shift",
        "min_effect_scale",
        "random_common_cause_scale",
    ):
        value = getattr(request, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            errors.append(f"{name}_must_be_positive")
    for name in ("max_failed_fraction", "max_placebo_p_value"):
        value = getattr(request, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
        ):
            errors.append(f"{name}_must_be_in_unit_interval")
    if (
        not isinstance(request.random_state, int)
        or isinstance(request.random_state, bool)
        or request.random_state < 0
    ):
        errors.append("random_state_must_be_a_non_negative_integer")
    return tuple(errors)


def _placebo_estimates(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    identification: IdentificationResult,
    estimation_request: EstimationRequest,
    request: RefutationRequest,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    values: list[float] = []
    failures = 0
    original = data[treatment].to_numpy(copy=True)
    for _ in range(request.n_simulations):
        refuted = data.copy()
        refuted[treatment] = rng.permutation(original)
        estimate = estimate_effect(
            refuted, treatment, outcome, identification, estimation_request
        )
        if estimate.estimated:
            values.append(float(estimate.estimate))
        else:
            failures += 1
    return values, failures


def _random_common_cause_estimates(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    identification: IdentificationResult,
    estimation_request: EstimationRequest,
    request: RefutationRequest,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    values: list[float] = []
    failures = 0
    name = _unique_column(data, "__random_common_cause__")
    adjustment_sets = tuple(
        tuple(sorted((*adjustment_set, name)))
        for adjustment_set in identification.adjustment_sets
    )
    refuted_identification = replace(
        identification, adjustment_sets=adjustment_sets
    )
    for _ in range(request.n_simulations):
        refuted = data.copy()
        refuted[name] = rng.normal(
            loc=0.0,
            scale=request.random_common_cause_scale,
            size=len(data),
        )
        estimate = estimate_effect(
            refuted,
            treatment,
            outcome,
            refuted_identification,
            estimation_request,
        )
        if estimate.estimated:
            values.append(float(estimate.estimate))
        else:
            failures += 1
    return values, failures


def _subset_estimates(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    identification: IdentificationResult,
    estimation_request: EstimationRequest,
    request: RefutationRequest,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    values: list[float] = []
    failures = 0
    treatment_values = data[treatment].to_numpy(dtype=float)
    groups = tuple(np.flatnonzero(treatment_values == level) for level in (0.0, 1.0))
    for _ in range(request.n_simulations):
        sampled = []
        for group in groups:
            size = max(2, int(np.floor(len(group) * request.subset_fraction)))
            sampled.extend(rng.choice(group, size=size, replace=False).tolist())
        indices = np.asarray(sampled, dtype=int)
        rng.shuffle(indices)
        refuted = data.iloc[indices].reset_index(drop=True)
        estimate = estimate_effect(
            refuted, treatment, outcome, identification, estimation_request
        )
        if estimate.estimated:
            values.append(float(estimate.estimate))
        else:
            failures += 1
    return values, failures


def _summarize_check(
    name: str,
    observed: float,
    values: list[float],
    failures: int,
    request: RefutationRequest,
    placebo: bool = False,
) -> RefutationCheck:
    successful = len(values)
    total = successful + failures
    failure_fraction = failures / total if total else 1.0
    reasons: list[str] = []
    if failure_fraction > request.max_failed_fraction:
        reasons.append("failed_refutation_fraction_exceeds_threshold")
    if not values:
        reasons.append("no_successful_refutation_runs")
        return RefutationCheck(
            name=name,
            status=RefutationStatus.REVIEW,
            successful_runs=0,
            failed_runs=failures,
            mean_refuted_estimate=None,
            mean_absolute_shift=None,
            max_absolute_shift=None,
            relative_shift=None,
            empirical_p_value=None,
            reasons=tuple(reasons),
        )

    array = np.asarray(values, dtype=float)
    shifts = np.abs(array - observed)
    mean_shift = float(shifts.mean())
    maximum_shift = float(shifts.max())
    relative_shift = float(mean_shift / max(abs(observed), request.min_effect_scale))
    p_value = None
    if placebo:
        p_value = float(
            (1 + np.count_nonzero(np.abs(array) >= abs(observed)))
            / (len(array) + 1)
        )
        if p_value > request.max_placebo_p_value:
            reasons.append("placebo_effect_not_distinguishable_from_observed")
    else:
        if mean_shift > request.max_absolute_shift:
            reasons.append("mean_absolute_shift_exceeds_threshold")
        if (
            abs(observed) >= request.min_effect_scale
            and relative_shift > request.max_relative_shift
        ):
            reasons.append("relative_shift_exceeds_threshold")
    status = RefutationStatus.PASS if not reasons else RefutationStatus.REVIEW
    return RefutationCheck(
        name=name,
        status=status,
        successful_runs=successful,
        failed_runs=failures,
        mean_refuted_estimate=float(array.mean()),
        mean_absolute_shift=mean_shift,
        max_absolute_shift=maximum_shift,
        relative_shift=relative_shift,
        empirical_p_value=p_value,
        reasons=tuple(reasons),
    )


def _unique_column(data: pd.DataFrame, stem: str) -> str:
    name = stem
    counter = 1
    while name in data.columns:
        name = f"{stem}{counter}"
        counter += 1
    return name
