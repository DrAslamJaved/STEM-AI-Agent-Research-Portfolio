from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class TruthLevel(str, Enum):
    INDIVIDUAL = "INDIVIDUAL"
    AVERAGE_REFERENCE = "AVERAGE_REFERENCE"
    NONE = "NONE"


class EvaluationStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_TRUTH = "INSUFFICIENT_TRUTH"
    INVALID_REQUEST = "INVALID_REQUEST"


@dataclass(frozen=True)
class BenchmarkProvenance:
    source_path: str
    source_sha256: str
    source_uri: str
    version: str
    replication: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    data: pd.DataFrame
    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    truth_level: TruthLevel
    true_ate: float | None
    individual_effect: np.ndarray | None
    provenance: BenchmarkProvenance
    reasons: tuple[str, ...]
    requires_human_review: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_observations": len(self.data),
            "treatment": self.treatment,
            "outcome": self.outcome,
            "covariates": list(self.covariates),
            "truth_level": self.truth_level.value,
            "true_ate": self.true_ate,
            "individual_effect_available": self.individual_effect is not None,
            "provenance": self.provenance.to_dict(),
            "reasons": list(self.reasons),
            "requires_human_review": self.requires_human_review,
        }


@dataclass(frozen=True)
class CausalMetrics:
    status: EvaluationStatus
    benchmark: str
    n_observations: int
    estimated_ate: float | None
    reference_ate: float | None
    ate_bias: float | None
    ate_error: float | None
    pehe: float | None
    confidence_interval: tuple[float, float] | None
    ci_covered: bool | None
    reasons: tuple[str, ...]
    requires_human_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class BenchmarkSummary:
    n_results: int
    ate_metric_count: int
    pehe_metric_count: int
    coverage_metric_count: int
    mean_ate_error: float | None
    ate_rmse: float | None
    mean_pehe: float | None
    confidence_interval_coverage: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def load_ihdp(
    path: str | Path,
    *,
    replication: int = 0,
    treatment_column: str = "t",
    outcome_column: str = "yf",
    mu0_column: str = "mu0",
    mu1_column: str = "mu1",
    covariates: Iterable[str] | None = None,
    source_uri: str = "",
    version: str = "unversioned",
) -> BenchmarkDataset:
    """Load an IHDP-style CSV or canonical NPZ replication from local storage."""
    source = _validated_path(path)
    if not isinstance(replication, int) or isinstance(replication, bool) or replication < 0:
        raise ValueError("replication must be a non-negative integer")
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        if replication != 0:
            raise ValueError("CSV input contains one replication; replication must be zero")
        frame = pd.read_csv(source)
        selected_covariates = _resolve_covariates(
            frame,
            covariates,
            excluded=(
                treatment_column,
                outcome_column,
                mu0_column,
                mu1_column,
                "ycf",
            ),
        )
    elif suffix == ".npz":
        if covariates is not None:
            raise ValueError("NPZ covariate names are generated and cannot be overridden")
        frame, selected_covariates = _load_ihdp_npz(
            source,
            replication,
            treatment_column,
            outcome_column,
            mu0_column,
            mu1_column,
        )
    else:
        raise ValueError("IHDP input must be a .csv or .npz file")

    _validate_columns(
        frame,
        treatment_column,
        outcome_column,
        selected_covariates,
        extra_numeric=(mu0_column, mu1_column),
    )
    truth_columns = {mu0_column, mu1_column, "ycf"}
    if any(name in truth_columns for name in selected_covariates):
        raise ValueError("counterfactual truth columns cannot be used as covariates")
    mu0 = frame[mu0_column].to_numpy(dtype=float)
    mu1 = frame[mu1_column].to_numpy(dtype=float)
    individual_effect = mu1 - mu0
    return BenchmarkDataset(
        name="IHDP",
        data=frame,
        treatment=treatment_column,
        outcome=outcome_column,
        covariates=selected_covariates,
        truth_level=TruthLevel.INDIVIDUAL,
        true_ate=float(np.mean(individual_effect)),
        individual_effect=individual_effect,
        provenance=_provenance(source, source_uri, version, replication),
        reasons=("counterfactual_means_available",),
    )


def load_lalonde(
    path: str | Path,
    *,
    treatment_column: str = "treat",
    outcome_column: str = "re78",
    covariates: Iterable[str] | None = None,
    reference_ate: float | None = None,
    source_uri: str = "",
    version: str = "unversioned",
) -> BenchmarkDataset:
    """Load a local Lalonde/NSW CSV without inventing counterfactual truth."""
    source = _validated_path(path)
    if source.suffix.casefold() != ".csv":
        raise ValueError("Lalonde input must be a .csv file")
    frame = pd.read_csv(source)
    selected_covariates = _resolve_covariates(
        frame,
        covariates,
        excluded=(treatment_column, outcome_column),
    )
    _validate_columns(frame, treatment_column, outcome_column, selected_covariates)
    if reference_ate is not None and not _finite_number(reference_ate):
        raise ValueError("reference_ate must be finite when supplied")
    truth_level = (
        TruthLevel.AVERAGE_REFERENCE if reference_ate is not None else TruthLevel.NONE
    )
    reasons = (
        ("experimental_average_effect_reference_supplied", "individual_truth_unavailable")
        if reference_ate is not None
        else ("counterfactual_truth_unavailable",)
    )
    return BenchmarkDataset(
        name="LALONDE",
        data=frame,
        treatment=treatment_column,
        outcome=outcome_column,
        covariates=selected_covariates,
        truth_level=truth_level,
        true_ate=float(reference_ate) if reference_ate is not None else None,
        individual_effect=None,
        provenance=_provenance(source, source_uri, version, None),
        reasons=reasons,
    )


def evaluate_predictions(
    benchmark: BenchmarkDataset,
    *,
    estimated_ate: float | None = None,
    estimated_ite: Iterable[float] | None = None,
    confidence_interval: tuple[float, float] | None = None,
) -> CausalMetrics:
    if not isinstance(benchmark, BenchmarkDataset):
        raise TypeError("benchmark must be a BenchmarkDataset")
    errors: list[str] = []
    ite = None
    if estimated_ite is not None:
        try:
            ite = np.asarray(tuple(estimated_ite), dtype=float)
        except (TypeError, ValueError):
            errors.append("estimated_ite_must_be_numeric")
        else:
            if ite.ndim != 1 or len(ite) != len(benchmark.data):
                errors.append("estimated_ite_length_mismatch")
            elif not np.all(np.isfinite(ite)):
                errors.append("estimated_ite_must_be_finite")
    if estimated_ate is not None and not _finite_number(estimated_ate):
        errors.append("estimated_ate_must_be_finite")
    if estimated_ate is None and ite is None:
        errors.append("estimated_ate_or_ite_is_required")
    derived_ate = float(np.mean(ite)) if ite is not None and not errors else None
    if estimated_ate is not None and derived_ate is not None and not np.isclose(
        float(estimated_ate), derived_ate, rtol=1e-9, atol=1e-12
    ):
        errors.append("estimated_ate_inconsistent_with_ite")
    interval = _validate_interval(confidence_interval, errors)
    chosen_ate = float(estimated_ate) if _finite_number(estimated_ate) else derived_ate
    if errors:
        return _invalid_metrics(benchmark, chosen_ate, interval, tuple(errors))

    if benchmark.true_ate is None:
        return CausalMetrics(
            status=EvaluationStatus.INSUFFICIENT_TRUTH,
            benchmark=benchmark.name,
            n_observations=len(benchmark.data),
            estimated_ate=chosen_ate,
            reference_ate=None,
            ate_bias=None,
            ate_error=None,
            pehe=None,
            confidence_interval=interval,
            ci_covered=None,
            reasons=("average_effect_reference_unavailable",),
        )

    reference = float(benchmark.true_ate)
    bias = float(chosen_ate - reference)
    pehe = None
    reasons: list[str] = []
    if benchmark.individual_effect is None:
        reasons.append("pehe_unavailable:individual_counterfactual_truth_absent")
    elif ite is None:
        reasons.append("pehe_unavailable:estimated_ite_not_supplied")
    else:
        pehe = float(np.sqrt(np.mean(np.square(ite - benchmark.individual_effect))))
    covered = None if interval is None else interval[0] <= reference <= interval[1]
    if interval is None:
        reasons.append("ci_coverage_not_requested")
    status = (
        EvaluationStatus.COMPLETE
        if pehe is not None and covered is not None
        else EvaluationStatus.PARTIAL
    )
    return CausalMetrics(
        status=status,
        benchmark=benchmark.name,
        n_observations=len(benchmark.data),
        estimated_ate=chosen_ate,
        reference_ate=reference,
        ate_bias=bias,
        ate_error=abs(bias),
        pehe=pehe,
        confidence_interval=interval,
        ci_covered=covered,
        reasons=tuple(reasons),
    )


def summarize_replications(results: Iterable[CausalMetrics]) -> BenchmarkSummary:
    try:
        items = tuple(results)
    except TypeError as exc:
        raise TypeError("results must be an iterable of CausalMetrics") from exc
    if not items:
        raise ValueError("at least one result is required")
    if any(not isinstance(item, CausalMetrics) for item in items):
        raise TypeError("every result must be a CausalMetrics instance")
    if any(item.status is EvaluationStatus.INVALID_REQUEST for item in items):
        raise ValueError("invalid evaluation results cannot be summarized")
    if len({item.benchmark for item in items}) != 1:
        raise ValueError("replication summaries cannot mix benchmarks")
    ate_errors = [item.ate_error for item in items if item.ate_error is not None]
    ate_biases = [item.ate_bias for item in items if item.ate_bias is not None]
    pehes = [item.pehe for item in items if item.pehe is not None]
    coverage = [item.ci_covered for item in items if item.ci_covered is not None]
    reasons: list[str] = []
    if not ate_errors:
        reasons.append("ate_metrics_unavailable")
    if not pehes:
        reasons.append("pehe_metrics_unavailable")
    if not coverage:
        reasons.append("coverage_metrics_unavailable")
    return BenchmarkSummary(
        n_results=len(items),
        ate_metric_count=len(ate_errors),
        pehe_metric_count=len(pehes),
        coverage_metric_count=len(coverage),
        mean_ate_error=_mean_or_none(ate_errors),
        ate_rmse=(
            float(np.sqrt(np.mean(np.square(ate_biases)))) if ate_biases else None
        ),
        mean_pehe=_mean_or_none(pehes),
        confidence_interval_coverage=_mean_or_none(coverage),
        reasons=tuple(reasons),
    )


def _load_ihdp_npz(
    source: Path,
    replication: int,
    treatment_column: str,
    outcome_column: str,
    mu0_column: str,
    mu1_column: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    with np.load(source, allow_pickle=False) as archive:
        required = {"x", "t", "yf", "mu0", "mu1"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"IHDP NPZ is missing arrays: {sorted(missing)}")
        x = np.asarray(archive["x"], dtype=float)
        if x.ndim == 2:
            if replication != 0:
                raise ValueError("single-replication NPZ requires replication zero")
            selected_x = x
        elif x.ndim == 3:
            if replication >= x.shape[2]:
                raise ValueError("replication index is outside the IHDP NPZ range")
            selected_x = x[:, :, replication]
        else:
            raise ValueError("IHDP x array must have two or three dimensions")
        vectors = {
            treatment_column: _select_replication(archive["t"], replication, len(selected_x)),
            outcome_column: _select_replication(archive["yf"], replication, len(selected_x)),
            mu0_column: _select_replication(archive["mu0"], replication, len(selected_x)),
            mu1_column: _select_replication(archive["mu1"], replication, len(selected_x)),
        }
    covariate_names = tuple(f"x{index + 1}" for index in range(selected_x.shape[1]))
    frame = pd.DataFrame(selected_x, columns=covariate_names)
    for name, values in vectors.items():
        frame[name] = values
    return frame, covariate_names


def _select_replication(values: np.ndarray, replication: int, n_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if replication != 0:
            raise ValueError("single-replication array requires replication zero")
        selected = array
    elif array.ndim == 2:
        if array.shape[0] != n_rows and array.shape[1] == n_rows:
            array = array.T
        if replication >= array.shape[1]:
            raise ValueError("replication index is outside an IHDP array range")
        selected = array[:, replication]
    else:
        raise ValueError("IHDP vector arrays must have one or two dimensions")
    if len(selected) != n_rows:
        raise ValueError("IHDP arrays have inconsistent observation counts")
    return selected


def _resolve_covariates(
    frame: pd.DataFrame,
    covariates: Iterable[str] | None,
    *,
    excluded: tuple[str, ...],
) -> tuple[str, ...]:
    if covariates is None:
        selected = tuple(name for name in frame.columns if name not in excluded)
    else:
        if isinstance(covariates, (str, bytes)):
            raise TypeError("covariates must be an iterable of column names")
        selected = tuple(covariates)
    if not selected:
        raise ValueError("at least one covariate is required")
    if any(not isinstance(name, str) or not name for name in selected):
        raise ValueError("covariate names must be non-empty strings")
    if len(set(selected)) != len(selected):
        raise ValueError("covariate names must be unique")
    return selected


def _validate_columns(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    *,
    extra_numeric: tuple[str, ...] = (),
) -> None:
    requested = (treatment, outcome, *covariates, *extra_numeric)
    missing = tuple(name for name in requested if name not in frame.columns)
    if missing:
        raise ValueError(f"missing benchmark columns: {missing}")
    if treatment == outcome or treatment in covariates or outcome in covariates:
        raise ValueError("treatment, outcome, and covariates must be distinct")
    for name in requested:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"benchmark column must be numeric:{name}")
        if not np.all(np.isfinite(frame[name].to_numpy(dtype=float))):
            raise ValueError(f"benchmark column must be finite:{name}")
    levels = set(frame[treatment].to_numpy(dtype=float))
    if levels != {0.0, 1.0}:
        raise ValueError("treatment must contain both binary levels zero and one")


def _validate_interval(
    interval: tuple[float, float] | None, errors: list[str]
) -> tuple[float, float] | None:
    if interval is None:
        return None
    if (
        not isinstance(interval, tuple)
        or len(interval) != 2
        or any(not _finite_number(value) for value in interval)
        or interval[0] > interval[1]
    ):
        errors.append("confidence_interval_is_invalid")
        return None
    return float(interval[0]), float(interval[1])


def _invalid_metrics(
    benchmark: BenchmarkDataset,
    estimated_ate: float | None,
    interval: tuple[float, float] | None,
    reasons: tuple[str, ...],
) -> CausalMetrics:
    return CausalMetrics(
        status=EvaluationStatus.INVALID_REQUEST,
        benchmark=benchmark.name,
        n_observations=len(benchmark.data),
        estimated_ate=estimated_ate,
        reference_ate=benchmark.true_ate,
        ate_bias=None,
        ate_error=None,
        pehe=None,
        confidence_interval=interval,
        ci_covered=None,
        reasons=reasons,
    )


def _validated_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"benchmark file was not found: {source}")
    return source


def _provenance(
    source: Path, source_uri: str, version: str, replication: int | None
) -> BenchmarkProvenance:
    if not isinstance(source_uri, str) or not isinstance(version, str):
        raise TypeError("source_uri and version must be strings")
    digest = sha256(source.read_bytes()).hexdigest()
    return BenchmarkProvenance(
        source_path=str(source.resolve()),
        source_sha256=digest,
        source_uri=source_uri,
        version=version,
        replication=replication,
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _mean_or_none(values: list[float | bool]) -> float | None:
    return float(np.mean(values)) if values else None
