from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_audit_agent.identification import (
    IdentificationResult,
    IdentificationStatus,
)


class EstimationStatus(str, Enum):
    ESTIMATED = "ESTIMATED"
    BLOCKED_IDENTIFICATION = "BLOCKED_IDENTIFICATION"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_DATA = "INVALID_DATA"
    POSITIVITY_VIOLATION = "POSITIVITY_VIOLATION"
    UNSUPPORTED = "UNSUPPORTED"


SUPPORTED_METHODS = frozenset(
    {"difference_in_means", "g_computation", "ipw", "aipw"}
)


@dataclass(frozen=True)
class EstimationRequest:
    method: str = "g_computation"
    confidence_level: float = 0.95
    propensity_clip: tuple[float, float] = (0.01, 0.99)
    max_clipped_fraction: float = 0.10
    random_state: int = 0


@dataclass(frozen=True)
class EffectEstimate:
    status: EstimationStatus
    estimand: str
    method: str
    estimate: float | None
    standard_error: float | None
    confidence_interval: tuple[float, float] | None
    n_observations: int
    adjustment_set: tuple[str, ...]
    diagnostics: dict[str, float | int | str]
    reasons: tuple[str, ...]
    requires_human_review: bool = True

    @property
    def estimated(self) -> bool:
        return self.status is EstimationStatus.ESTIMATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "estimand": self.estimand,
            "method": self.method,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "confidence_interval": (
                list(self.confidence_interval) if self.confidence_interval else None
            ),
            "n_observations": self.n_observations,
            "adjustment_set": list(self.adjustment_set),
            "diagnostics": dict(self.diagnostics),
            "reasons": list(self.reasons),
            "requires_human_review": self.requires_human_review,
        }


def estimate_effect(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    identification: IdentificationResult,
    request: EstimationRequest | None = None,
    adjustment_set: tuple[str, ...] | None = None,
) -> EffectEstimate:
    """Estimate an identified ATE or ATT with transparent baseline methods."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(identification, IdentificationResult):
        raise TypeError("identification must be an IdentificationResult")
    if request is None:
        request = EstimationRequest()
    if not isinstance(request, EstimationRequest):
        raise TypeError("request must be an EstimationRequest")

    method = request.method.casefold() if isinstance(request.method, str) else ""
    estimand = identification.estimand.upper()
    n_observations = len(data)

    if identification.status is not IdentificationStatus.IDENTIFIED:
        return _blocked(
            EstimationStatus.BLOCKED_IDENTIFICATION,
            estimand,
            method,
            n_observations,
            (),
            (f"identification_status:{identification.status.value}",),
        )

    request_errors = _validate_request(request, method)
    if request_errors:
        return _blocked(
            EstimationStatus.INVALID_REQUEST,
            estimand,
            method,
            n_observations,
            (),
            request_errors,
        )

    if estimand not in {"ATE", "ATT"}:
        return _blocked(
            EstimationStatus.UNSUPPORTED,
            estimand,
            method,
            n_observations,
            (),
            (f"unsupported_estimand:{estimand}",),
        )
    if method == "aipw" and estimand != "ATE":
        return _blocked(
            EstimationStatus.UNSUPPORTED,
            estimand,
            method,
            n_observations,
            (),
            ("aipw_baseline_supports_ate_only",),
        )

    allowed_sets = {
        tuple(sorted(items)) for items in identification.adjustment_sets
    }
    if not allowed_sets:
        return _blocked(
            EstimationStatus.BLOCKED_IDENTIFICATION,
            estimand,
            method,
            n_observations,
            (),
            ("identification_has_no_adjustment_set",),
        )
    chosen = (
        tuple(sorted(adjustment_set))
        if adjustment_set is not None
        else sorted(allowed_sets, key=lambda items: (len(items), items))[0]
    )
    if chosen not in allowed_sets:
        return _blocked(
            EstimationStatus.INVALID_REQUEST,
            estimand,
            method,
            n_observations,
            chosen,
            ("adjustment_set_not_identified",),
        )
    if method == "difference_in_means" and chosen:
        return _blocked(
            EstimationStatus.INVALID_REQUEST,
            estimand,
            method,
            n_observations,
            chosen,
            ("difference_in_means_cannot_ignore_required_adjustment",),
        )

    data_errors = _validate_data(data, treatment, outcome, chosen)
    if data_errors:
        return _blocked(
            EstimationStatus.INVALID_DATA,
            estimand,
            method,
            n_observations,
            chosen,
            data_errors,
        )

    treatment_values = data[treatment].to_numpy(dtype=float)
    outcome_values = data[outcome].to_numpy(dtype=float)
    covariates = data.loc[:, list(chosen)].to_numpy(dtype=float)
    diagnostics: dict[str, float | int | str] = {
        "treated_fraction": float(treatment_values.mean()),
        "n_treated": int(treatment_values.sum()),
        "n_control": int(n_observations - treatment_values.sum()),
    }

    if method == "difference_in_means":
        estimate, standard_error = _difference_in_means(
            treatment_values, outcome_values
        )
    elif method == "g_computation":
        propensity, overlap, positivity_reason = _propensity_scores(
            treatment_values, covariates, request
        )
        diagnostics.update(overlap)
        if positivity_reason:
            return _blocked(
                EstimationStatus.POSITIVITY_VIOLATION,
                estimand,
                method,
                n_observations,
                chosen,
                (positivity_reason,),
                diagnostics,
            )
        del propensity
        regression = _g_computation(treatment_values, outcome_values, covariates)
        if regression is None:
            return _blocked(
                EstimationStatus.INVALID_DATA,
                estimand,
                method,
                n_observations,
                chosen,
                ("outcome_design_matrix_is_rank_deficient",),
                diagnostics,
            )
        estimate, standard_error, rank = regression
        diagnostics["outcome_design_rank"] = rank
    else:
        propensity, overlap, positivity_reason = _propensity_scores(
            treatment_values, covariates, request
        )
        diagnostics.update(overlap)
        if positivity_reason:
            return _blocked(
                EstimationStatus.POSITIVITY_VIOLATION,
                estimand,
                method,
                n_observations,
                chosen,
                (positivity_reason,),
                diagnostics,
            )
        if method == "ipw":
            estimate, standard_error, ess = _ipw(
                treatment_values, outcome_values, propensity, estimand
            )
            diagnostics["effective_sample_size"] = ess
        else:
            estimate, standard_error = _aipw(
                treatment_values, outcome_values, covariates, propensity
            )

    interval = _confidence_interval(
        estimate, standard_error, request.confidence_level
    )
    return EffectEstimate(
        status=EstimationStatus.ESTIMATED,
        estimand=estimand,
        method=method,
        estimate=float(estimate),
        standard_error=float(standard_error),
        confidence_interval=interval,
        n_observations=n_observations,
        adjustment_set=chosen,
        diagnostics=diagnostics,
        reasons=(
            "estimate_conditional_on_identification_and_model_assumptions",
        ),
    )


def _validate_request(
    request: EstimationRequest, method: str
) -> tuple[str, ...]:
    errors: list[str] = []
    if method not in SUPPORTED_METHODS:
        errors.append(f"unsupported_method:{method or '<blank>'}")
    if not isinstance(request.confidence_level, (int, float)) or not (
        0.0 < request.confidence_level < 1.0
    ):
        errors.append("confidence_level_must_be_between_zero_and_one")
    clip = request.propensity_clip
    if (
        not isinstance(clip, (list, tuple))
        or len(clip) != 2
        or not all(isinstance(value, (int, float)) for value in clip)
        or not 0.0 < clip[0] < clip[1] < 1.0
    ):
        errors.append("invalid_propensity_clip")
    if not isinstance(request.max_clipped_fraction, (int, float)) or not (
        0.0 <= request.max_clipped_fraction <= 1.0
    ):
        errors.append("max_clipped_fraction_must_be_in_unit_interval")
    if not isinstance(request.random_state, int):
        errors.append("random_state_must_be_an_integer")
    return tuple(errors)


def _validate_data(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: tuple[str, ...],
) -> tuple[str, ...]:
    if treatment == outcome:
        return ("treatment_and_outcome_must_be_distinct",)
    invalid_adjustments = tuple(
        name for name in adjustment_set if name in {treatment, outcome}
    )
    if invalid_adjustments:
        return tuple(
            f"invalid_adjustment_target:{name}" for name in invalid_adjustments
        )
    required = (treatment, outcome, *adjustment_set)
    missing = tuple(name for name in required if name not in data.columns)
    if missing:
        return tuple(f"missing_column:{name}" for name in missing)
    if len(data) < 4:
        return ("at_least_four_observations_are_required",)
    non_numeric = tuple(name for name in required if not is_numeric_dtype(data[name]))
    if non_numeric:
        return tuple(f"non_numeric_column:{name}" for name in non_numeric)
    values = data.loc[:, list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return ("missing_or_non_finite_values",)
    treatment_values = data[treatment].to_numpy(dtype=float)
    levels = set(np.unique(treatment_values))
    if levels != {0.0, 1.0}:
        return ("treatment_must_be_binary_zero_one",)
    counts = np.bincount(treatment_values.astype(int), minlength=2)
    if np.any(counts < 2):
        return ("each_treatment_group_requires_two_observations",)
    return ()


def _difference_in_means(
    treatment: np.ndarray, outcome: np.ndarray
) -> tuple[float, float]:
    treated = outcome[treatment == 1]
    control = outcome[treatment == 0]
    estimate = treated.mean() - control.mean()
    standard_error = np.sqrt(
        treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)
    )
    return float(estimate), float(standard_error)


def _g_computation(
    treatment: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray,
) -> tuple[float, float, int] | None:
    design = np.column_stack((np.ones(len(treatment)), treatment, covariates))
    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank < design.shape[1] or len(treatment) <= rank:
        return None
    residuals = outcome - design @ coefficients
    residual_variance = float(residuals @ residuals / (len(treatment) - rank))
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    return float(coefficients[1]), standard_error, int(rank)


def _propensity_scores(
    treatment: np.ndarray,
    covariates: np.ndarray,
    request: EstimationRequest,
) -> tuple[np.ndarray, dict[str, float], str | None]:
    if covariates.shape[1] == 0:
        raw = np.full(len(treatment), treatment.mean(), dtype=float)
    else:
        model = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=request.random_state
        )
        model.fit(covariates, treatment.astype(int))
        raw = model.predict_proba(covariates)[:, 1]
    lower, upper = request.propensity_clip
    clipped_mask = (raw < lower) | (raw > upper)
    fraction_clipped = float(clipped_mask.mean())
    clipped = np.clip(raw, lower, upper)
    diagnostics = {
        "propensity_min": float(raw.min()),
        "propensity_max": float(raw.max()),
        "propensity_fraction_clipped": fraction_clipped,
    }
    reason = None
    if fraction_clipped > request.max_clipped_fraction:
        reason = "propensity_overlap_below_configured_threshold"
    return clipped, diagnostics, reason


def _ipw(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    estimand: str,
) -> tuple[float, float, float]:
    if estimand == "ATE":
        scores = (
            treatment * outcome / propensity
            - (1.0 - treatment) * outcome / (1.0 - propensity)
        )
        estimate = float(scores.mean())
        standard_error = float(scores.std(ddof=1) / np.sqrt(len(scores)))
        weights = treatment / propensity + (1.0 - treatment) / (1.0 - propensity)
    else:
        treated = treatment == 1
        control = ~treated
        control_weights = propensity[control] / (1.0 - propensity[control])
        treated_mean = float(outcome[treated].mean())
        control_mean = float(
            np.average(outcome[control], weights=control_weights)
        )
        estimate = treated_mean - control_mean
        treated_variance = outcome[treated].var(ddof=1) / treated.sum()
        centered = outcome[control] - control_mean
        control_variance = np.sum((control_weights * centered) ** 2) / (
            control_weights.sum() ** 2
        )
        standard_error = float(np.sqrt(treated_variance + control_variance))
        weights = treatment + (1.0 - treatment) * propensity / (1.0 - propensity)
    ess = float(weights.sum() ** 2 / np.sum(weights**2))
    return float(estimate), standard_error, ess


def _aipw(
    treatment: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray,
    propensity: np.ndarray,
) -> tuple[float, float]:
    treated = treatment == 1
    control = ~treated
    if covariates.shape[1] == 0:
        predicted_treated = np.full(len(outcome), outcome[treated].mean())
        predicted_control = np.full(len(outcome), outcome[control].mean())
    else:
        treated_model = LinearRegression().fit(covariates[treated], outcome[treated])
        control_model = LinearRegression().fit(covariates[control], outcome[control])
        predicted_treated = treated_model.predict(covariates)
        predicted_control = control_model.predict(covariates)
    scores = (
        predicted_treated
        - predicted_control
        + treatment * (outcome - predicted_treated) / propensity
        - (1.0 - treatment) * (outcome - predicted_control) / (1.0 - propensity)
    )
    return float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(len(scores)))


def _confidence_interval(
    estimate: float, standard_error: float, confidence_level: float
) -> tuple[float, float]:
    quantile = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    margin = quantile * standard_error
    return float(estimate - margin), float(estimate + margin)


def _blocked(
    status: EstimationStatus,
    estimand: str,
    method: str,
    n_observations: int,
    adjustment_set: tuple[str, ...],
    reasons: tuple[str, ...],
    diagnostics: dict[str, float | int | str] | None = None,
) -> EffectEstimate:
    return EffectEstimate(
        status=status,
        estimand=estimand,
        method=method,
        estimate=None,
        standard_error=None,
        confidence_interval=None,
        n_observations=n_observations,
        adjustment_set=adjustment_set,
        diagnostics=diagnostics or {},
        reasons=reasons,
    )
