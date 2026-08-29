"""Reporting for synthetic PCA anomaly evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pathlib import Path

from .detector import AnomalyThresholdResult
from .evaluation import BinaryEvaluationResult
from .pca_workflow import PCAFitResult
import json

from matplotlib.backends.backend_agg import (
    FigureCanvasAgg,
)
from matplotlib.figure import Figure

@dataclass(frozen=True)
class SyntheticEvaluationArtifacts:
    """Paths for deterministic Phase 6 artifacts."""

    summary_json: Path
    predictions_csv: Path
    metrics_csv: Path
    scenario_metrics_csv: Path
    confusion_matrix_figure: Path
    reconstruction_errors_figure: Path
    scree_plot_figure: Path
    scenario_rates_figure: Path

def _to_builtin(value: object) -> object:
    """Convert NumPy and nested values to JSON-safe types."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): _to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_builtin(item)
            for item in value
        ]

    return value

def resolve_synthetic_evaluation_artifacts(
    output_root: str | Path = ".",
) -> SyntheticEvaluationArtifacts:
    """Resolve Phase 6 artifact paths without writing files."""

    if not isinstance(output_root, (str, Path)):
        raise TypeError(
            "output_root must be a string or Path."
        )

    root = Path(output_root)

    return SyntheticEvaluationArtifacts(
        summary_json=(
            root
            / "results"
            / "synthetic_evaluation.json"
        ),
        predictions_csv=(
            root
            / "results"
            / "synthetic_predictions.csv"
        ),
        metrics_csv=(
            root
            / "reports"
            / "tables"
            / "synthetic_metrics.csv"
        ),
        scenario_metrics_csv=(
            root
            / "reports"
            / "tables"
            / "synthetic_scenario_metrics.csv"
        ),
        confusion_matrix_figure=(
            root
            / "reports"
            / "figures"
            / "synthetic_confusion_matrix.png"
        ),
        reconstruction_errors_figure=(
            root
            / "reports"
            / "figures"
            / "synthetic_reconstruction_errors.png"
        ),
        scree_plot_figure=(
            root
            / "reports"
            / "figures"
            / "synthetic_scree_plot.png"
        ),
        scenario_rates_figure=(
            root
            / "reports"
            / "figures"
            / "synthetic_scenario_rates.png"
        ),
    )

def build_synthetic_evaluation_summary(
    fit_result: PCAFitResult,
    threshold_result: AnomalyThresholdResult,
    binary_result: BinaryEvaluationResult,
    scenario_result: pd.DataFrame,
) -> dict[str, object]:
    """Build the serializable Phase 6 summary."""

    if not isinstance(fit_result, PCAFitResult):
        raise TypeError(
            "fit_result must be a PCAFitResult."
        )

    if not isinstance(
        threshold_result,
        AnomalyThresholdResult,
    ):
        raise TypeError(
            "threshold_result must be an "
            "AnomalyThresholdResult."
        )

    if not isinstance(
        binary_result,
        BinaryEvaluationResult,
    ):
        raise TypeError(
            "binary_result must be a "
            "BinaryEvaluationResult."
        )

    if not isinstance(scenario_result, pd.DataFrame):
        raise TypeError(
            "scenario_result must be a pandas "
            "DataFrame."
        )

    if scenario_result.empty:
        raise ValueError(
            "scenario_result must not be empty."
        )

    expected_scenario_columns = (
        "scenario",
        "true_label",
        "observations",
        "predicted_normal",
        "predicted_anomaly",
        "predicted_anomaly_rate",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "maximum_reconstruction_error",
    )

    if (
        tuple(scenario_result.columns)
        != expected_scenario_columns
    ):
        raise ValueError(
            "scenario_result columns must exactly "
            f"match {expected_scenario_columns}."
        )

    scenario_observations = int(
        scenario_result["observations"].sum()
    )

    if scenario_observations != binary_result.total:
        raise ValueError(
            "Scenario observations do not match "
            "the binary evaluation total."
        )

    metrics = _to_builtin(
        asdict(binary_result)
    )

    scenarios = _to_builtin(
        scenario_result.to_dict(
            orient="records"
        )
    )

    summary: dict[str, object] = {
        "status": "passed",
        "data": {
            "test_observations": int(
                binary_result.total
            ),
            "normal_observations": int(
                binary_result.normal_support
            ),
            "anomaly_observations": int(
                binary_result.anomaly_support
            ),
        },
        "pca": {
            "selected_components": int(
                fit_result.n_components
            ),
            "explained_variance_target": float(
                fit_result.explained_variance_target
            ),
            "achieved_explained_variance": float(
                fit_result.achieved_explained_variance
            ),
            "full_explained_variance": (
                np.asarray(
                    fit_result.full_explained_variance,
                    dtype=np.float64,
                ).tolist()
            ),
            "full_explained_variance_ratios": (
                np.asarray(
                    fit_result
                    .full_explained_variance_ratio,
                    dtype=np.float64,
                ).tolist()
            ),
            "full_cumulative_explained_variance": (
                np.asarray(
                    fit_result
                    .full_cumulative_explained_variance,
                    dtype=np.float64,
                ).tolist()
            ),
        },
        "threshold": {
            "value": float(
                threshold_result.threshold
            ),
            "quantile": float(
                threshold_result.quantile
            ),
            "quantile_method": str(
                threshold_result.quantile_method
            ),
            "calibration_count": int(
                threshold_result.calibration_count
            ),
            "comparison": (
                "strictly_greater_than"
            ),
        },
        "metrics": metrics,
        "scenarios": scenarios,
        "limitations": (
            "Synthetic evaluation evidence does not "
            "represent real-world cybersecurity "
            "performance."
        ),
    }

    return summary

def _new_figure(
    *,
    width: float,
    height: float,
) -> Figure:
    """Create a headless Matplotlib figure."""

    figure = Figure(
        figsize=(width, height)
    )
    FigureCanvasAgg(figure)

    return figure


def _save_figure(
    figure: Figure,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Save a deterministic PNG and release its contents."""

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Software": "cyber-pca",
        },
    )

    figure.clear()

def write_synthetic_evaluation_artifacts(
    evaluation_data: pd.DataFrame,
    fit_result: PCAFitResult,
    threshold_result: AnomalyThresholdResult,
    binary_result: BinaryEvaluationResult,
    scenario_result: pd.DataFrame,
    *,
    output_root: str | Path = ".",
    dpi: int = 150,
) -> SyntheticEvaluationArtifacts:
    """Write tables, JSON evidence, and figures."""

    if not isinstance(evaluation_data, pd.DataFrame):
        raise TypeError(
            "evaluation_data must be a pandas "
            "DataFrame."
        )

    if evaluation_data.empty:
        raise ValueError(
            "evaluation_data must not be empty."
        )

    expected_evaluation_columns = (
        "true_anomaly",
        "predicted_anomaly",
        "scenario",
        "reconstruction_error",
    )

    if (
        tuple(evaluation_data.columns)
        != expected_evaluation_columns
    ):
        raise ValueError(
            "evaluation_data columns must exactly "
            f"match {expected_evaluation_columns}."
        )

    if evaluation_data.index.name != "flow_id":
        raise ValueError(
            "evaluation_data index must be named "
            "flow_id."
        )

    if (
        isinstance(dpi, bool)
        or not isinstance(dpi, int)
    ):
        raise TypeError(
            "dpi must be an integer."
        )

    if dpi <= 0:
        raise ValueError(
            "dpi must be positive."
        )

    summary = build_synthetic_evaluation_summary(
        fit_result,
        threshold_result,
        binary_result,
        scenario_result,
    )

    artifacts = (
        resolve_synthetic_evaluation_artifacts(
            output_root
        )
    )

    artifact_paths = (
        artifacts.summary_json,
        artifacts.predictions_csv,
        artifacts.metrics_csv,
        artifacts.scenario_metrics_csv,
        artifacts.confusion_matrix_figure,
        artifacts.reconstruction_errors_figure,
        artifacts.scree_plot_figure,
        artifacts.scenario_rates_figure,
    )

    for artifact_path in artifact_paths:
        artifact_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    summary_text = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    artifacts.summary_json.write_text(
        summary_text + "\n",
        encoding="utf-8",
        newline="\n",
    )

    evaluation_data.to_csv(
        artifacts.predictions_csv,
        index=True,
        index_label="flow_id",
        float_format="%.17g",
        lineterminator="\n",
    )

    metrics_record = asdict(binary_result)

    metrics_record["confusion_matrix"] = (
        json.dumps(
            _to_builtin(
                binary_result.confusion_matrix
            ),
            separators=(",", ":"),
        )
    )

    metrics_frame = pd.DataFrame(
        [metrics_record]
    )

    metrics_frame.to_csv(
        artifacts.metrics_csv,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    scenario_result.to_csv(
        artifacts.scenario_metrics_csv,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    confusion_matrix = np.asarray(
        binary_result.confusion_matrix,
        dtype=np.int64,
    )

    confusion_figure = _new_figure(
        width=5.5,
        height=4.5,
    )

    confusion_axis = (
        confusion_figure.subplots()
    )

    confusion_image = (
        confusion_axis.imshow(
            confusion_matrix,
            cmap="Blues",
        )
    )

    for row in range(2):
        for column in range(2):
            confusion_axis.text(
                column,
                row,
                str(
                    confusion_matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

    confusion_axis.set_xticks([0, 1])
    confusion_axis.set_yticks([0, 1])

    confusion_axis.set_xticklabels(
        ["Normal", "Anomaly"]
    )
    confusion_axis.set_yticklabels(
        ["Normal", "Anomaly"]
    )

    confusion_axis.set_xlabel(
        "Predicted label"
    )
    confusion_axis.set_ylabel(
        "True label"
    )
    confusion_axis.set_title(
        "Synthetic test confusion matrix"
    )

    confusion_figure.colorbar(
        confusion_image,
        ax=confusion_axis,
        fraction=0.046,
        pad=0.04,
    )

    _save_figure(
        confusion_figure,
        artifacts.confusion_matrix_figure,
        dpi=dpi,
    )

    true_labels = evaluation_data[
        "true_anomaly"
    ].to_numpy(
        dtype=np.int8,
        copy=True,
    )

    reconstruction_errors = evaluation_data[
        "reconstruction_error"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    normal_errors = reconstruction_errors[
        true_labels == 0
    ]

    anomaly_errors = reconstruction_errors[
        true_labels == 1
    ]

    error_figure = _new_figure(
        width=7.0,
        height=4.5,
    )

    error_axis = error_figure.subplots()

    error_axis.hist(
        np.log1p(normal_errors),
        bins=30,
        alpha=0.70,
        label="Normal",
        color="#2563EB",
    )

    error_axis.hist(
        np.log1p(anomaly_errors),
        bins=30,
        alpha=0.65,
        label="Anomaly",
        color="#DC2626",
    )

    error_axis.set_xlabel(
        "log(1 + reconstruction MSE)"
    )
    error_axis.set_ylabel(
        "Observations"
    )
    error_axis.set_title(
        "Synthetic reconstruction-error "
        "distributions"
    )
    error_axis.legend()
    error_axis.grid(
        axis="y",
        alpha=0.25,
    )

    _save_figure(
        error_figure,
        artifacts.reconstruction_errors_figure,
        dpi=dpi,
    )

    variance_ratios = np.asarray(
        fit_result.full_explained_variance_ratio,
        dtype=np.float64,
    )

    cumulative_variance = np.asarray(
        fit_result
        .full_cumulative_explained_variance,
        dtype=np.float64,
    )

    component_numbers = np.arange(
        1,
        variance_ratios.size + 1,
        dtype=np.int64,
    )

    scree_figure = _new_figure(
        width=7.0,
        height=4.5,
    )

    scree_axis = scree_figure.subplots()

    scree_axis.bar(
        component_numbers,
        variance_ratios,
        alpha=0.75,
        color="#2563EB",
        label="Individual variance",
    )

    scree_axis.plot(
        component_numbers,
        cumulative_variance,
        color="#DC2626",
        marker="o",
        linewidth=2.0,
        label="Cumulative variance",
    )

    scree_axis.axhline(
        fit_result.explained_variance_target,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label="Variance target",
    )

    scree_axis.axvline(
        fit_result.n_components,
        color="#059669",
        linestyle=":",
        linewidth=1.5,
        label="Selected components",
    )

    scree_axis.set_xticks(
        component_numbers
    )
    scree_axis.set_ylim(0.0, 1.05)
    scree_axis.set_xlabel(
        "Principal component"
    )
    scree_axis.set_ylabel(
        "Explained variance ratio"
    )
    scree_axis.set_title(
        "PCA scree and cumulative variance"
    )
    scree_axis.legend()
    scree_axis.grid(
        axis="y",
        alpha=0.25,
    )

    _save_figure(
        scree_figure,
        artifacts.scree_plot_figure,
        dpi=dpi,
    )

    scenario_names = scenario_result[
        "scenario"
    ].astype(str).tolist()

    scenario_rates = (
        scenario_result[
            "predicted_anomaly_rate"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        * 100.0
    )

    scenario_figure = _new_figure(
        width=8.0,
        height=4.8,
    )

    scenario_axis = (
        scenario_figure.subplots()
    )

    scenario_colors = [
        (
            "#D97706"
            if scenario == "normal"
            else "#0F766E"
        )
        for scenario in scenario_names
    ]

    bars = scenario_axis.bar(
        [
            scenario.replace("_", "\n")
            for scenario in scenario_names
        ],
        scenario_rates,
        color=scenario_colors,
    )

    for bar, rate in zip(
        bars,
        scenario_rates,
        strict=True,
    ):
        scenario_axis.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            min(float(rate) + 2.0, 103.0),
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    scenario_axis.set_ylim(0.0, 105.0)
    scenario_axis.set_ylabel(
        "Predicted anomaly rate (%)"
    )
    scenario_axis.set_title(
        "Synthetic scenario detection rates"
    )
    scenario_axis.grid(
        axis="y",
        alpha=0.25,
    )

    _save_figure(
        scenario_figure,
        artifacts.scenario_rates_figure,
        dpi=dpi,
    )

    return artifacts
