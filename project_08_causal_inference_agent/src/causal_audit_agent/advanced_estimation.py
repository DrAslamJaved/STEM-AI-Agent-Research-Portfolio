from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from causal_audit_agent.identification import (
    IdentificationResult,
    IdentificationStatus,
)


class AdvancedEstimationStatus(str, Enum):
    ESTIMATED = "ESTIMATED"
    BLOCKED_IDENTIFICATION = "BLOCKED_IDENTIFICATION"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_DATA = "INVALID_DATA"
    POSITIVITY_VIOLATION = "POSITIVITY_VIOLATION"
    UNSUPPORTED = "UNSUPPORTED"


SUPPORTED_METHODS = frozenset({"cross_fitted_aipw", "dr_learner_forest"})


@dataclass(frozen=True)
class AdvancedEstimationRequest:
    method: str = "cross_fitted_aipw"
    confidence_level: float = 0.95
    n_splits: int = 5
    propensity_clip: tuple[float, float] = (0.01, 0.99)
    max_clipped_fraction: float = 0.10
    n_estimators: int = 200
    min_samples_leaf: int = 5
    random_state: int = 0


@dataclass(frozen=True)
class AdvancedEffectEstimate:
    status: AdvancedEstimationStatus
    estimand: str
    method: str
    estimate: float | None
    standard_error: float | None
    confidence_interval: tuple[float, float] | None
    n_observations: int
    adjustment_set: tuple[str, ...]
    conditioning_variables: tuple[str, ...]
    unit_effects: tuple[float, ...] | None
    diagnostics: dict[str, float | int | str]
    reasons: tuple[str, ...]
    requires_human_review: bool = True

    @property
    def estimated(self) -> bool:
        return self.status is AdvancedEstimationStatus.ESTIMATED

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
            "conditioning_variables": list(self.conditioning_variables),
            "unit_effects": list(self.unit_effects) if self.unit_effects else None,
            "diagnostics": dict(self.diagnostics),
            "reasons": list(self.reasons),
            "requires_human_review": self.requires_human_review,
        }


def estimate_advanced_effect(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    identification: IdentificationResult,
    request: AdvancedEstimationRequest | None = None,
    adjustment_set: tuple[str, ...] | None = None,
) -> AdvancedEffectEstimate:
    """Estimate an identified ATE or CATE with cross-fitted learners.

    ``dr_learner_forest`` is a forest final-stage doubly robust learner. It is
    intentionally named precisely: it is not presented as a generalized
    random forest implementation.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(identification, IdentificationResult):
        raise TypeError("identification must be an IdentificationResult")
    if request is None:
        request = AdvancedEstimationRequest()
    if not isinstance(request, AdvancedEstimationRequest):
        raise TypeError("request must be an AdvancedEstimationRequest")

    method = request.method.casefold() if isinstance(request.method, str) else ""
    estimand = identification.estimand.upper()
    n_observations = len(data)
    conditioning = tuple(sorted(identification.conditioning_variables))

    if identification.status is not IdentificationStatus.IDENTIFIED:
        return _blocked(
            AdvancedEstimationStatus.BLOCKED_IDENTIFICATION,
            estimand,
            method,
            n_observations,
            (),
            conditioning,
            (f"identification_status:{identification.status.value}",),
        )

    request_errors = _validate_request(request, method)
    if request_errors:
        return _blocked(
            AdvancedEstimationStatus.INVALID_REQUEST,
            estimand,
            method,
            n_observations,
            (),
            conditioning,
            request_errors,
        )

    supported_estimand = (
        method == "cross_fitted_aipw" and estimand == "ATE"
    ) or (method == "dr_learner_forest" and estimand == "CATE")
    if not supported_estimand:
        return _blocked(
            AdvancedEstimationStatus.UNSUPPORTED,
            estimand,
            method,
            n_observations,
            (),
            conditioning,
            (f"method_estimand_pair_not_supported:{method}:{estimand}",),
        )
    if method == "dr_learner_forest" and not conditioning:
        return _blocked(
            AdvancedEstimationStatus.INVALID_REQUEST,
            estimand,
            method,
            n_observations,
            (),
            conditioning,
            ("dr_learner_forest_requires_conditioning_variables",),
        )

    allowed_sets = {
        tuple(sorted(items)) for items in identification.adjustment_sets
    }
    if not allowed_sets:
        return _blocked(
            AdvancedEstimationStatus.BLOCKED_IDENTIFICATION,
            estimand,
            method,
            n_observations,
            (),
            conditioning,
            ("identification_has_no_adjustment_set",),
        )
    chosen = (
        tuple(sorted(adjustment_set))
        if adjustment_set is not None
        else sorted(allowed_sets, key=lambda items: (len(items), items))[0]
    )
    if chosen not in allowed_sets:
        return _blocked(
            AdvancedEstimationStatus.INVALID_REQUEST,
            estimand,
            method,
            n_observations,
            chosen,
            conditioning,
            ("adjustment_set_not_identified",),
        )

    nuisance_variables = tuple(sorted(set(chosen) | set(conditioning)))
    data_errors = _validate_data(
        data, treatment, outcome, nuisance_variables, request.n_splits
    )
    if data_errors:
        return _blocked(
            AdvancedEstimationStatus.INVALID_DATA,
            estimand,
            method,
            n_observations,
            chosen,
            conditioning,
            data_errors,
        )

    treatment_values = data[treatment].to_numpy(dtype=float)
    outcome_values = data[outcome].to_numpy(dtype=float)
    nuisance_features = data.loc[:, list(nuisance_variables)].to_numpy(dtype=float)
    propensity, predicted_treated, predicted_control = _cross_fitted_nuisance(
        treatment_values, outcome_values, nuisance_features, request
    )
    raw_propensity = propensity.copy()
    lower, upper = request.propensity_clip
    clipped_mask = (propensity < lower) | (propensity > upper)
    clipped_fraction = float(clipped_mask.mean())
    propensity = np.clip(propensity, lower, upper)
    diagnostics: dict[str, float | int | str] = {
        "n_splits": request.n_splits,
        "n_treated": int(treatment_values.sum()),
        "n_control": int(n_observations - treatment_values.sum()),
        "propensity_min": float(raw_propensity.min()),
        "propensity_max": float(raw_propensity.max()),
        "propensity_fraction_clipped": clipped_fraction,
    }
    if clipped_fraction > request.max_clipped_fraction:
        return _blocked(
            AdvancedEstimationStatus.POSITIVITY_VIOLATION,
            estimand,
            method,
            n_observations,
            chosen,
            conditioning,
            ("propensity_overlap_below_configured_threshold",),
            diagnostics,
        )

    scores = _doubly_robust_scores(
        treatment_values,
        outcome_values,
        propensity,
        predicted_treated,
        predicted_control,
    )
    if method == "cross_fitted_aipw":
        unit_effects = None
        estimate = float(scores.mean())
        standard_error = float(scores.std(ddof=1) / np.sqrt(n_observations))
        diagnostics["score_standard_deviation"] = float(scores.std(ddof=1))
    else:
        modifiers = data.loc[:, list(conditioning)].to_numpy(dtype=float)
        effects, importance = _cross_fitted_forest(scores, modifiers, request)
        unit_effects = tuple(float(value) for value in effects)
        estimate = float(effects.mean())
        standard_error = None
        diagnostics["cate_min"] = float(effects.min())
        diagnostics["cate_max"] = float(effects.max())
        diagnostics["cate_standard_deviation"] = float(effects.std(ddof=1))
        for name, value in zip(conditioning, importance, strict=True):
            diagnostics[f"importance:{name}"] = float(value)

    interval = (
        _confidence_interval(estimate, standard_error, request.confidence_level)
        if standard_error is not None
        else None
    )
    return AdvancedEffectEstimate(
        status=AdvancedEstimationStatus.ESTIMATED,
        estimand=estimand,
        method=method,
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval=interval,
        n_observations=n_observations,
        adjustment_set=chosen,
        conditioning_variables=conditioning,
        unit_effects=unit_effects,
        diagnostics=diagnostics,
        reasons=(
            "estimate_conditional_on_identification_and_model_assumptions",
            "cross_fitting_reduces_nuisance_overfit_bias_not_causal_assumption_bias",
            *(
                ("cate_pointwise_uncertainty_not_estimated",)
                if method == "dr_learner_forest"
                else ()
            ),
        ),
    )


def _validate_request(
    request: AdvancedEstimationRequest, method: str
) -> tuple[str, ...]:
    errors: list[str] = []
    if method not in SUPPORTED_METHODS:
        errors.append(f"unsupported_method:{method or '<blank>'}")
    if not isinstance(request.confidence_level, (int, float)) or not (
        0.0 < request.confidence_level < 1.0
    ):
        errors.append("confidence_level_must_be_between_zero_and_one")
    if not isinstance(request.n_splits, int) or isinstance(request.n_splits, bool) or not (
        2 <= request.n_splits <= 10
    ):
        errors.append("n_splits_must_be_an_integer_between_two_and_ten")
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
    if not isinstance(request.n_estimators, int) or isinstance(
        request.n_estimators, bool
    ) or request.n_estimators < 10:
        errors.append("n_estimators_must_be_an_integer_at_least_ten")
    if not isinstance(request.min_samples_leaf, int) or isinstance(
        request.min_samples_leaf, bool
    ) or request.min_samples_leaf < 1:
        errors.append("min_samples_leaf_must_be_a_positive_integer")
    if not isinstance(request.random_state, int) or isinstance(
        request.random_state, bool
    ):
        errors.append("random_state_must_be_an_integer")
    return tuple(errors)


def _validate_data(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    n_splits: int,
) -> tuple[str, ...]:
    if treatment == outcome:
        return ("treatment_and_outcome_must_be_distinct",)
    invalid_covariates = tuple(
        name for name in covariates if name in {treatment, outcome}
    )
    if invalid_covariates:
        return tuple(f"invalid_covariate:{name}" for name in invalid_covariates)
    required = (treatment, outcome, *covariates)
    missing = tuple(name for name in required if name not in data.columns)
    if missing:
        return tuple(f"missing_column:{name}" for name in missing)
    non_numeric = tuple(name for name in required if not is_numeric_dtype(data[name]))
    if non_numeric:
        return tuple(f"non_numeric_column:{name}" for name in non_numeric)
    values = data.loc[:, list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return ("missing_or_non_finite_values",)
    treatment_values = data[treatment].to_numpy(dtype=float)
    if set(np.unique(treatment_values)) != {0.0, 1.0}:
        return ("treatment_must_be_binary_zero_one",)
    counts = np.bincount(treatment_values.astype(int), minlength=2)
    if np.any(counts < 2 * n_splits):
        return ("each_treatment_group_requires_at_least_twice_n_splits_observations",)
    if len(data) < max(20, 4 * n_splits):
        return ("sample_size_below_cross_fitting_minimum",)
    return ()


def _cross_fitted_nuisance(
    treatment: np.ndarray,
    outcome: np.ndarray,
    features: np.ndarray,
    request: AdvancedEstimationRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    propensity = np.empty(len(treatment), dtype=float)
    predicted_treated = np.empty(len(treatment), dtype=float)
    predicted_control = np.empty(len(treatment), dtype=float)
    splitter = StratifiedKFold(
        n_splits=request.n_splits,
        shuffle=True,
        random_state=request.random_state,
    )
    for train, test in splitter.split(features, treatment):
        train_treatment = treatment[train].astype(int)
        train_outcome = outcome[train]
        if features.shape[1] == 0:
            propensity[test] = train_treatment.mean()
            predicted_treated[test] = train_outcome[train_treatment == 1].mean()
            predicted_control[test] = train_outcome[train_treatment == 0].mean()
            continue
        propensity_model = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=request.random_state
        ).fit(features[train], train_treatment)
        propensity[test] = propensity_model.predict_proba(features[test])[:, 1]
        treated_model = LinearRegression().fit(
            features[train][train_treatment == 1],
            train_outcome[train_treatment == 1],
        )
        control_model = LinearRegression().fit(
            features[train][train_treatment == 0],
            train_outcome[train_treatment == 0],
        )
        predicted_treated[test] = treated_model.predict(features[test])
        predicted_control[test] = control_model.predict(features[test])
    return propensity, predicted_treated, predicted_control


def _doubly_robust_scores(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    predicted_treated: np.ndarray,
    predicted_control: np.ndarray,
) -> np.ndarray:
    return (
        predicted_treated
        - predicted_control
        + treatment * (outcome - predicted_treated) / propensity
        - (1.0 - treatment) * (outcome - predicted_control) / (1.0 - propensity)
    )


def _cross_fitted_forest(
    scores: np.ndarray,
    modifiers: np.ndarray,
    request: AdvancedEstimationRequest,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty(len(scores), dtype=float)
    importance = np.zeros(modifiers.shape[1], dtype=float)
    splitter = KFold(
        n_splits=request.n_splits,
        shuffle=True,
        random_state=request.random_state,
    )
    for fold, (train, test) in enumerate(splitter.split(modifiers)):
        model = RandomForestRegressor(
            n_estimators=request.n_estimators,
            min_samples_leaf=request.min_samples_leaf,
            random_state=request.random_state + fold,
            n_jobs=1,
        ).fit(modifiers[train], scores[train])
        predictions[test] = model.predict(modifiers[test])
        importance += model.feature_importances_ / request.n_splits
    return predictions, importance


def _confidence_interval(
    estimate: float, standard_error: float, confidence_level: float
) -> tuple[float, float]:
    quantile = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    margin = quantile * standard_error
    return float(estimate - margin), float(estimate + margin)


def _blocked(
    status: AdvancedEstimationStatus,
    estimand: str,
    method: str,
    n_observations: int,
    adjustment_set: tuple[str, ...],
    conditioning_variables: tuple[str, ...],
    reasons: tuple[str, ...],
    diagnostics: dict[str, float | int | str] | None = None,
) -> AdvancedEffectEstimate:
    return AdvancedEffectEstimate(
        status=status,
        estimand=estimand,
        method=method,
        estimate=None,
        standard_error=None,
        confidence_interval=None,
        n_observations=n_observations,
        adjustment_set=adjustment_set,
        conditioning_variables=conditioning_variables,
        unit_effects=None,
        diagnostics=diagnostics or {},
        reasons=reasons,
    )
