"""Reporting for official UNSW-NB15 evaluation evidence."""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib.figure import Figure

from cyber_pca.evaluation import (
    BinaryEvaluationResult,
)
from cyber_pca.unsw_evaluation import (
    UNSW_ATTACK_CATEGORY_ORDER,
)
from cyber_pca.unsw_experiment import (
    UNSWDetectionResult,
)


@dataclass(frozen=True)
class UNSWEvaluationArtifacts:
    """Permanent Phase 8 evaluation artifact paths."""

    summary_json: Path
    predictions_csv: Path
    metrics_csv: Path
    attack_category_metrics_csv: Path
    confusion_matrix_figure: Path
    reconstruction_errors_figure: Path
    scree_plot_figure: Path
    attack_category_rates_figure: Path


def _to_builtin(
    value: object,
) -> object:
    """Convert nested NumPy values to built-in types."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [
            _to_builtin(item)
            for item in value.tolist()
        ]

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


def _series_sha256(
    series: pd.Series,
    *,
    dtype: object,
) -> str:
    """Hash one indexed numeric evidence series."""

    digest = sha256()

    digest.update(
        "\n".join(
            str(value)
            for value in series.index
        ).encode("utf-8")
    )

    values = np.ascontiguousarray(
        series.to_numpy(
            dtype=dtype,
            copy=True,
        )
    )

    digest.update(
        str(values.shape).encode("ascii")
    )
    digest.update(
        str(values.dtype).encode("ascii")
    )
    digest.update(values.tobytes())

    return digest.hexdigest()


def _model_state_sha256(
    detection_result: UNSWDetectionResult,
) -> str:
    """Hash the fitted model and frozen threshold state."""

    fit_result = detection_result.fit_result
    model = fit_result.model

    digest = sha256()

    scalar_values = (
        fit_result.n_components,
        fit_result.explained_variance_target,
        fit_result.achieved_explained_variance,
        detection_result
        .threshold_result.threshold,
        detection_result
        .threshold_result.quantile,
        detection_result
        .threshold_result.calibration_count,
    )

    digest.update(
        repr(scalar_values).encode("ascii")
    )

    arrays = (
        model.mean_,
        model.components_,
        fit_result.full_explained_variance,
        fit_result.full_explained_variance_ratio,
        fit_result
        .full_cumulative_explained_variance,
    )

    for values in arrays:
        array = np.ascontiguousarray(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )

        digest.update(
            str(array.shape).encode("ascii")
        )
        digest.update(array.tobytes())

    return digest.hexdigest()


def _new_figure(
    *,
    width: float,
    height: float,
) -> Figure:
    """Create one deterministic noninteractive figure."""

    return Figure(
        figsize=(width, height),
        facecolor="white",
        layout="constrained",
    )


def _save_figure(
    figure: Figure,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Save and close one deterministic PNG figure."""

    figure.savefig(
        path,
        format="png",
        dpi=dpi,
        facecolor="white",
        metadata={
            "Software": "cyber_pca",
        },
    )

    figure.clear()


def resolve_unsw_evaluation_artifacts(
    output_root: str | Path = ".",
) -> UNSWEvaluationArtifacts:
    """Resolve Phase 8 output paths without writing files."""

    if not isinstance(
        output_root,
        (str, Path),
    ):
        raise TypeError(
            "output_root must be a string "
            "or Path."
        )

    root = Path(output_root)

    return UNSWEvaluationArtifacts(
        summary_json=(
            root
            / "results"
            / "unsw_nb15_evaluation.json"
        ),
        predictions_csv=(
            root
            / "results"
            / "unsw_nb15_predictions.csv"
        ),
        metrics_csv=(
            root
            / "reports"
            / "tables"
            / "unsw_nb15_metrics.csv"
        ),
        attack_category_metrics_csv=(
            root
            / "reports"
            / "tables"
            / (
                "unsw_nb15_"
                "attack_category_metrics.csv"
            )
        ),
        confusion_matrix_figure=(
            root
            / "reports"
            / "figures"
            / (
                "unsw_nb15_"
                "confusion_matrix.png"
            )
        ),
        reconstruction_errors_figure=(
            root
            / "reports"
            / "figures"
            / (
                "unsw_nb15_"
                "reconstruction_errors.png"
            )
        ),
        scree_plot_figure=(
            root
            / "reports"
            / "figures"
            / "unsw_nb15_scree_plot.png"
        ),
        attack_category_rates_figure=(
            root
            / "reports"
            / "figures"
            / (
                "unsw_nb15_"
                "attack_category_rates.png"
            )
        ),
    )


def _validate_reporting_inputs(
    detection_result: object,
    binary_result: object,
    attack_category_result: object,
) -> None:
    """Validate shared reporting inputs."""

    if not isinstance(
        detection_result,
        UNSWDetectionResult,
    ):
        raise TypeError(
            "detection_result must be a "
            "UNSWDetectionResult."
        )

    if not isinstance(
        binary_result,
        BinaryEvaluationResult,
    ):
        raise TypeError(
            "binary_result must be a "
            "BinaryEvaluationResult."
        )

    if not isinstance(
        attack_category_result,
        pd.DataFrame,
    ):
        raise TypeError(
            "attack_category_result must be "
            "a pandas DataFrame."
        )

    if attack_category_result.empty:
        raise ValueError(
            "attack_category_result must not "
            "be empty."
        )

    expected_columns = (
        "attack_category",
        "true_label",
        "observations",
        "predicted_normal",
        "predicted_anomaly",
        "predicted_anomaly_rate",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "maximum_reconstruction_error",
    )

    if tuple(
        attack_category_result.columns
    ) != expected_columns:
        raise ValueError(
            "attack_category_result columns "
            f"must exactly match {expected_columns}."
        )

    if attack_category_result[
        "attack_category"
    ].tolist() != list(
        UNSW_ATTACK_CATEGORY_ORDER
    ):
        raise ValueError(
            "attack_category_result must use "
            "the official category order."
        )

    category_total = int(
        attack_category_result[
            "observations"
        ].sum()
    )

    if category_total != binary_result.total:
        raise ValueError(
            "Attack-category observations do "
            "not match the binary total."
        )

    if (
        detection_result.test_predictions.shape[0]
        != binary_result.total
    ):
        raise ValueError(
            "Frozen predictions do not match "
            "the binary evaluation total."
        )

    if (
        detection_result
        .reconstruction_errors.test.shape[0]
        != binary_result.total
    ):
        raise ValueError(
            "Test reconstruction errors do not "
            "match the binary evaluation total."
        )


def build_unsw_evaluation_summary(
    detection_result: UNSWDetectionResult,
    binary_result: BinaryEvaluationResult,
    attack_category_result: pd.DataFrame,
) -> dict[str, object]:
    """Build the serializable Phase 8 summary."""

    _validate_reporting_inputs(
        detection_result,
        binary_result,
        attack_category_result,
    )

    fit_result = detection_result.fit_result
    threshold_result = (
        detection_result.threshold_result
    )

    metrics = _to_builtin(
        asdict(binary_result)
    )

    attack_categories = _to_builtin(
        attack_category_result.to_dict(
            orient="records"
        )
    )

    summary: dict[str, object] = {
        "status": "passed",
        "phase": 8,
        "dataset": "UNSW-NB15",
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
            "fitting_split": (
                "normal_fit_only"
            ),
            "selected_components": int(
                fit_result.n_components
            ),
            "explained_variance_target": float(
                fit_result
                .explained_variance_target
            ),
            "achieved_explained_variance": float(
                fit_result
                .achieved_explained_variance
            ),
            "full_explained_variance": (
                np.asarray(
                    fit_result
                    .full_explained_variance,
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
            "calibration_split": (
                "normal_calibration_only"
            ),
            "value": float(
                threshold_result.threshold
            ),
            "quantile": float(
                threshold_result.quantile
            ),
            "quantile_method": str(
                threshold_result
                .quantile_method
            ),
            "calibration_count": int(
                threshold_result
                .calibration_count
            ),
            "comparison": (
                "strictly_greater_than"
            ),
        },
        "metrics": metrics,
        "attack_categories": (
            attack_categories
        ),
        "evidence": {
            "hash_algorithm": "sha256",
            "model_state_sha256": (
                _model_state_sha256(
                    detection_result
                )
            ),
            "test_reconstruction_errors_sha256": (
                _series_sha256(
                    detection_result
                    .reconstruction_errors.test,
                    dtype=np.float64,
                )
            ),
            "test_predictions_sha256": (
                _series_sha256(
                    detection_result
                    .test_predictions,
                    dtype=np.int8,
                )
            ),
        },
        "protocol": {
            "predictions_frozen_before_labels": (
                True
            ),
            "alignment_key": [
                "source_partition",
                "id",
            ],
            "label_access": (
                "after_predictions_frozen"
            ),
            "attack_category_access": (
                "after_predictions_frozen"
            ),
            "post_evaluation_tuning": False,
        },
        "limitations": (
            "This is an untuned PCA reconstruction-"
            "error baseline on the official "
            "UNSW-NB15 test partition. The observed "
            "performance must not be interpreted as "
            "an optimized operational detector."
        ),
    }

    return summary


def write_unsw_evaluation_artifacts(
    evaluation_data: pd.DataFrame,
    detection_result: UNSWDetectionResult,
    binary_result: BinaryEvaluationResult,
    attack_category_result: pd.DataFrame,
    *,
    output_root: str | Path = ".",
    dpi: int = 150,
) -> UNSWEvaluationArtifacts:
    """Write permanent Phase 8 evidence artifacts."""

    if not isinstance(
        evaluation_data,
        pd.DataFrame,
    ):
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

    if tuple(
        evaluation_data.columns
    ) != expected_evaluation_columns:
        raise ValueError(
            "evaluation_data columns must "
            "exactly match "
            f"{expected_evaluation_columns}."
        )

    if evaluation_data.index.name != "flow_id":
        raise ValueError(
            "evaluation_data index must be "
            "named flow_id."
        )

    if evaluation_data.index.hasnans:
        raise ValueError(
            "evaluation_data contains missing "
            "flow IDs."
        )

    if evaluation_data.index.duplicated().any():
        raise ValueError(
            "evaluation_data contains duplicate "
            "flow IDs."
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

    _validate_reporting_inputs(
        detection_result,
        binary_result,
        attack_category_result,
    )

    if (
        evaluation_data.shape[0]
        != binary_result.total
    ):
        raise ValueError(
            "evaluation_data rows do not match "
            "the binary total."
        )

    if not evaluation_data.index.equals(
        detection_result
        .test_predictions.index
    ):
        raise ValueError(
            "evaluation_data IDs do not match "
            "frozen prediction IDs."
        )

    if not evaluation_data.index.equals(
        detection_result
        .reconstruction_errors.test.index
    ):
        raise ValueError(
            "evaluation_data IDs do not match "
            "test reconstruction-error IDs."
        )

    summary = build_unsw_evaluation_summary(
        detection_result,
        binary_result,
        attack_category_result,
    )

    artifacts = (
        resolve_unsw_evaluation_artifacts(
            output_root
        )
    )

    artifact_paths = (
        artifacts.summary_json,
        artifacts.predictions_csv,
        artifacts.metrics_csv,
        artifacts.attack_category_metrics_csv,
        artifacts.confusion_matrix_figure,
        artifacts.reconstruction_errors_figure,
        artifacts.scree_plot_figure,
        artifacts.attack_category_rates_figure,
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

    metrics_record = asdict(
        binary_result
    )

    metrics_record["confusion_matrix"] = (
        json.dumps(
            _to_builtin(
                binary_result
                .confusion_matrix
            ),
            separators=(",", ":"),
        )
    )

    pd.DataFrame(
        [metrics_record]
    ).to_csv(
        artifacts.metrics_csv,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    attack_category_result.to_csv(
        artifacts
        .attack_category_metrics_csv,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    confusion_matrix = np.asarray(
        binary_result.confusion_matrix,
        dtype=np.int64,
    )

    confusion_figure = _new_figure(
        width=5.8,
        height=4.8,
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
        "Official UNSW-NB15 confusion matrix"
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

    reconstruction_errors = (
        evaluation_data[
            "reconstruction_error"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )

    error_figure = _new_figure(
        width=7.2,
        height=4.8,
    )

    error_axis = error_figure.subplots()

    error_axis.hist(
        np.log1p(
            reconstruction_errors[
                true_labels == 0
            ]
        ),
        bins=40,
        alpha=0.70,
        label="Normal",
        color="#2563EB",
    )

    error_axis.hist(
        np.log1p(
            reconstruction_errors[
                true_labels == 1
            ]
        ),
        bins=40,
        alpha=0.60,
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
        "UNSW-NB15 reconstruction-error "
        "distributions"
    )
    error_axis.legend()
    error_axis.grid(
        axis="y",
        alpha=0.25,
    )

    _save_figure(
        error_figure,
        artifacts
        .reconstruction_errors_figure,
        dpi=dpi,
    )

    fit_result = detection_result.fit_result

    variance_ratios = np.asarray(
        fit_result
        .full_explained_variance_ratio,
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
        width=8.0,
        height=4.8,
    )

    scree_axis = scree_figure.subplots()

    scree_axis.bar(
        component_numbers,
        variance_ratios,
        alpha=0.70,
        color="#2563EB",
        label="Individual variance",
    )

    scree_axis.plot(
        component_numbers,
        cumulative_variance,
        color="#DC2626",
        linewidth=2.0,
        label="Cumulative variance",
    )

    scree_axis.axhline(
        fit_result
        .explained_variance_target,
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

    scree_axis.set_ylim(0.0, 1.05)
    scree_axis.set_xlabel(
        "Principal component"
    )
    scree_axis.set_ylabel(
        "Explained variance ratio"
    )
    scree_axis.set_title(
        "UNSW-NB15 PCA scree plot"
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

    category_names = (
        attack_category_result[
            "attack_category"
        ].astype(str).tolist()
    )

    category_rates = (
        attack_category_result[
            "predicted_anomaly_rate"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        * 100.0
    )

    category_figure = _new_figure(
        width=10.0,
        height=5.2,
    )

    category_axis = (
        category_figure.subplots()
    )

    colors = [
        (
            "#D97706"
            if category == "Normal"
            else "#0F766E"
        )
        for category in category_names
    ]

    bars = category_axis.bar(
        category_names,
        category_rates,
        color=colors,
    )

    for bar, rate in zip(
        bars,
        category_rates,
        strict=True,
    ):
        category_axis.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            min(
                float(rate) + 1.0,
                102.0,
            ),
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    category_axis.set_ylim(
        0.0,
        max(
            10.0,
            min(
                105.0,
                float(
                    np.max(
                        category_rates
                    )
                )
                + 8.0,
            ),
        ),
    )
    category_axis.set_ylabel(
        "Predicted anomaly rate (%)"
    )
    category_axis.set_title(
        "UNSW-NB15 attack-category "
        "detection rates"
    )
    category_axis.tick_params(
        axis="x",
        rotation=35,
    )
    category_axis.grid(
        axis="y",
        alpha=0.25,
    )

    _save_figure(
        category_figure,
        artifacts
        .attack_category_rates_figure,
        dpi=dpi,
    )

    return artifacts
