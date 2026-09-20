from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class DiagnosticStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class DiagnosticThresholds:
    overlap_bounds: tuple[float, float] = (0.05, 0.95)
    max_abs_smd: float = 0.20
    min_overlap_fraction: float = 0.80
    min_ess_fraction: float = 0.50
    max_weight: float = 20.0
    propensity_clip: float = 1e-6
    max_clipped_fraction: float = 0.0


@dataclass(frozen=True)
class DiagnosticReport:
    max_abs_smd: float
    unweighted_max_abs_smd: float
    overlap_fraction: float
    effective_sample_size: float
    effective_sample_fraction: float
    max_weight: float
    extreme_weight_fraction: float
    clipped_propensity_fraction: float
    covariate_smds: dict[str, float]
    weighted_covariate_smds: dict[str, float]
    estimand: str
    n_observations: int
    status: DiagnosticStatus
    reasons: tuple[str, ...]
    requires_human_review: bool = True

    @property
    def passed(self) -> bool:
        return self.status is DiagnosticStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["reasons"] = list(self.reasons)
        return result


def standardized_mean_difference(
    values: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Return the signed treated-minus-control standardized mean difference."""
    x = np.asarray(values, dtype=float)
    t = np.asarray(treatment, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    _validate_smd_inputs(x, t, w)

    treated = t == 1.0
    control = ~treated
    if w is None:
        treated_mean = float(x[treated].mean())
        control_mean = float(x[control].mean())
        treated_variance = float(x[treated].var(ddof=1))
        control_variance = float(x[control].var(ddof=1))
    else:
        treated_mean, treated_variance = _weighted_mean_variance(
            x[treated], w[treated]
        )
        control_mean, control_variance = _weighted_mean_variance(
            x[control], w[control]
        )

    difference = treated_mean - control_mean
    pooled = float(np.sqrt((treated_variance + control_variance) / 2.0))
    if np.isclose(pooled, 0.0):
        if np.isclose(difference, 0.0):
            return 0.0
        return float(np.copysign(np.inf, difference))
    return float(difference / pooled)


def diagnose(
    data: pd.DataFrame,
    treatment: str,
    covariates: Sequence[str],
    propensity: np.ndarray,
    overlap_bounds: tuple[float, float] | None = None,
    *,
    estimand: str = "ATE",
    thresholds: DiagnosticThresholds | None = None,
) -> DiagnosticReport:
    """Audit empirical estimability using balance, overlap, and weight checks.

    A PASS is not evidence that exchangeability or the supplied causal graph is
    true. REVIEW is a deterministic stop-and-investigate signal; it does not
    authorize silent trimming or a changed target population.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if thresholds is None:
        thresholds = DiagnosticThresholds()
    if not isinstance(thresholds, DiagnosticThresholds):
        raise TypeError("thresholds must be DiagnosticThresholds")
    if overlap_bounds is not None:
        thresholds = replace(thresholds, overlap_bounds=overlap_bounds)
    _validate_thresholds(thresholds)

    if not isinstance(treatment, str) or not treatment.strip():
        raise ValueError("treatment must be a non-empty string")
    if not isinstance(covariates, (list, tuple)) or not all(
        isinstance(name, str) and name.strip() for name in covariates
    ):
        raise ValueError("covariates must be a sequence of non-empty strings")
    names = tuple(covariates)
    if len(set(names)) != len(names):
        raise ValueError("covariates must not contain duplicates")
    if treatment in names:
        raise ValueError("treatment cannot be included as a covariate")

    required = (treatment, *names)
    missing = tuple(name for name in required if name not in data.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    non_numeric = tuple(name for name in required if not is_numeric_dtype(data[name]))
    if non_numeric:
        raise ValueError(f"non-numeric columns: {', '.join(non_numeric)}")
    observed = data.loc[:, list(required)].to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError("diagnostic columns must contain only finite values")

    t = data[treatment].to_numpy(dtype=float)
    levels = set(np.unique(t))
    if levels != {0.0, 1.0}:
        raise ValueError("treatment must contain exactly binary values 0 and 1")
    counts = np.bincount(t.astype(int), minlength=2)
    if np.any(counts < 2):
        raise ValueError("each treatment group requires at least two observations")

    p = np.asarray(propensity, dtype=float)
    if p.ndim != 1 or len(p) != len(data):
        raise ValueError("propensity must be one-dimensional and match row count")
    if not np.isfinite(p).all():
        raise ValueError("propensity must contain only finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("propensity values must lie in the unit interval")

    target = estimand.upper() if isinstance(estimand, str) else ""
    if target not in {"ATE", "ATT"}:
        raise ValueError("estimand must be ATE or ATT")

    lower, upper = thresholds.overlap_bounds
    clipped_mask = (p < thresholds.propensity_clip) | (
        p > 1.0 - thresholds.propensity_clip
    )
    clipped_fraction = float(clipped_mask.mean())
    clipped = np.clip(
        p, thresholds.propensity_clip, 1.0 - thresholds.propensity_clip
    )
    weights = _causal_weights(t, clipped, target)

    raw_smds = {
        name: standardized_mean_difference(
            data[name].to_numpy(dtype=float), t
        )
        for name in names
    }
    weighted_smds = {
        name: standardized_mean_difference(
            data[name].to_numpy(dtype=float), t, weights
        )
        for name in names
    }
    raw_max = _max_absolute(raw_smds.values())
    weighted_max = _max_absolute(weighted_smds.values())
    overlap_fraction = float(np.mean((p >= lower) & (p <= upper)))
    effective_sample_size = float(weights.sum() ** 2 / np.square(weights).sum())
    ess_fraction = float(effective_sample_size / len(data))
    maximum_weight = float(weights.max())
    extreme_fraction = float(np.mean(weights > thresholds.max_weight))

    reasons: list[str] = []
    if weighted_max > thresholds.max_abs_smd:
        reasons.append("weighted_balance_exceeds_threshold")
    if overlap_fraction < thresholds.min_overlap_fraction:
        reasons.append("insufficient_propensity_overlap")
    if ess_fraction < thresholds.min_ess_fraction:
        reasons.append("low_effective_sample_size")
    if maximum_weight > thresholds.max_weight:
        reasons.append("extreme_weights_present")
    if clipped_fraction > thresholds.max_clipped_fraction:
        reasons.append("propensity_clipping_exceeds_threshold")

    status = DiagnosticStatus.PASS if not reasons else DiagnosticStatus.REVIEW
    return DiagnosticReport(
        max_abs_smd=weighted_max,
        unweighted_max_abs_smd=raw_max,
        overlap_fraction=overlap_fraction,
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=ess_fraction,
        max_weight=maximum_weight,
        extreme_weight_fraction=extreme_fraction,
        clipped_propensity_fraction=clipped_fraction,
        covariate_smds=raw_smds,
        weighted_covariate_smds=weighted_smds,
        estimand=target,
        n_observations=len(data),
        status=status,
        reasons=tuple(reasons),
    )


def _validate_smd_inputs(
    values: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None,
) -> None:
    if values.ndim != 1 or treatment.ndim != 1 or len(values) != len(treatment):
        raise ValueError("values and treatment must be equal-length vectors")
    if not np.isfinite(values).all() or not np.isfinite(treatment).all():
        raise ValueError("values and treatment must be finite")
    if set(np.unique(treatment)) != {0.0, 1.0}:
        raise ValueError("treatment must contain exactly binary values 0 and 1")
    counts = np.bincount(treatment.astype(int), minlength=2)
    if np.any(counts < 2):
        raise ValueError("each treatment group requires at least two observations")
    if weights is not None:
        if weights.ndim != 1 or len(weights) != len(values):
            raise ValueError("weights must be an equal-length vector")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("weights must be finite and strictly positive")


def _weighted_mean_variance(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    weight_sum = float(weights.sum())
    mean = float(np.dot(weights, values) / weight_sum)
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = weight_sum - float(np.square(weights).sum() / weight_sum)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("weights do not support a finite variance estimate")
    variance = float(np.dot(weights, np.square(values - mean)) / denominator)
    return mean, variance


def _causal_weights(
    treatment: np.ndarray, propensity: np.ndarray, estimand: str
) -> np.ndarray:
    if estimand == "ATE":
        return treatment / propensity + (1.0 - treatment) / (1.0 - propensity)
    return treatment + (1.0 - treatment) * propensity / (1.0 - propensity)


def _max_absolute(values: Any) -> float:
    items = tuple(abs(float(value)) for value in values)
    return max(items, default=0.0)


def _validate_thresholds(thresholds: DiagnosticThresholds) -> None:
    lower, upper = thresholds.overlap_bounds
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("overlap bounds must satisfy 0 < lower < upper < 1")
    if thresholds.max_abs_smd < 0.0:
        raise ValueError("max_abs_smd must be non-negative")
    if not 0.0 <= thresholds.min_overlap_fraction <= 1.0:
        raise ValueError("min_overlap_fraction must lie in the unit interval")
    if not 0.0 < thresholds.min_ess_fraction <= 1.0:
        raise ValueError("min_ess_fraction must lie in (0, 1]")
    if thresholds.max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    if not 0.0 < thresholds.propensity_clip < 0.5:
        raise ValueError("propensity_clip must lie in (0, 0.5)")
    if not 0.0 <= thresholds.max_clipped_fraction <= 1.0:
        raise ValueError("max_clipped_fraction must lie in the unit interval")
