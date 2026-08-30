"""Build a reproducible final evidence synthesis for the Davis DTI study.

This module does not train, tune, or select a model.  It validates and
summarises already committed evaluation artefacts produced with the frozen
cold-drug design.  It deliberately separates predictive results from
statistical evidence, biological interpretation, and causal claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PRIMARY_LABEL = "interaction_kd_le_1000_nM"
PRIMARY_VARIANT = "primary_kd_le_1000_nM"
SENSITIVITY_LABEL = "interaction_kd_le_100_nM"
SENSITIVITY_VARIANT = "sensitivity_kd_le_100_nM"
EXPECTED_MODEL_IDS = (
    "dummy_prior",
    "logistic_regression_balanced",
    "random_forest_balanced",
    "hist_gradient_boosting_balanced",
)
REPORTED_METRICS = (
    "average_precision",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
)
CONFUSION_KEYS = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)


class FinalSynthesisError(ValueError):
    """Raised when a required evidence artefact violates the frozen contract."""


@dataclass(frozen=True)
class FinalSynthesisRun:
    """Structured JSON evidence and its human-readable Markdown companion."""

    summary: dict[str, Any]
    markdown: str


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalSynthesisError(f"{name} must be a JSON object.")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalSynthesisError(f"{name} must be a JSON array.")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalSynthesisError(f"{name} must be a non-empty string.")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalSynthesisError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FinalSynthesisError(f"{name} must be finite.")
    return numeric


def _integer(value: Any, name: str) -> int:
    numeric = _number(value, name)
    if not numeric.is_integer():
        raise FinalSynthesisError(f"{name} must be an integer.")
    return int(numeric)


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise FinalSynthesisError(
            f"{name} must equal {expected!r}; received {actual!r}."
        )


def read_json_report(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON report and require an object at its top level."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FinalSynthesisError(f"Could not parse JSON report {path}: {error}") from error
    return _mapping(payload, f"JSON report {path}")


def sha256_file(path: Path) -> str:
    """Return a content hash for an input artefact recorded in the manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_requirements(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    requirements = [line.strip() for line in lines if line.strip()]
    if not requirements:
        raise FinalSynthesisError(f"Requirements file {path} is empty.")
    return requirements


def _dataset_commit_from_provenance(path: Path) -> str:
    """Extract the immutable upstream DeepDTA commit from the provenance record."""
    text = path.read_text(encoding="utf-8-sig")
    match = re.search(r"^Pinned commit:\s*([0-9a-f]{40})\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise FinalSynthesisError(
            f"Dataset provenance {path} does not contain a 40-character pinned commit."
        )
    return match.group(1)


def _cold_drug_split(split_audit: dict[str, Any]) -> dict[str, Any]:
    policies = _list(split_audit.get("split_policies"), "split_policies")
    matches = [
        _mapping(policy, "split policy")
        for policy in policies
        if _mapping(policy, "split policy").get("policy") == "cold_drug"
    ]
    if len(matches) != 1:
        raise FinalSynthesisError("Split audit must contain exactly one cold_drug policy.")

    split = matches[0]
    _require_equal(split.get("drug_overlap_count"), 0, "cold_drug drug_overlap_count")
    _require_equal(split.get("reference_label_column"), PRIMARY_LABEL, "cold_drug reference label")
    if _integer(split.get("train_pair_count"), "cold_drug train_pair_count") <= 0:
        raise FinalSynthesisError("cold_drug train_pair_count must be positive.")
    if _integer(split.get("test_pair_count"), "cold_drug test_pair_count") <= 0:
        raise FinalSynthesisError("cold_drug test_pair_count must be positive.")
    return split


def _model_results_by_id(report: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    results = _list(report.get("model_results"), f"{name}.model_results")
    indexed: dict[str, dict[str, Any]] = {}
    for raw_result in results:
        result = _mapping(raw_result, f"{name} model result")
        model_id = _string(result.get("model_id"), f"{name}.model_id")
        if model_id in indexed:
            raise FinalSynthesisError(f"{name} contains duplicate model id {model_id!r}.")
        indexed[model_id] = result

    if tuple(indexed) != EXPECTED_MODEL_IDS:
        raise FinalSynthesisError(
            f"{name} model IDs must be {EXPECTED_MODEL_IDS}; received {tuple(indexed)}."
        )
    return indexed


def _metric_summary(result: dict[str, Any], model_id: str) -> dict[str, dict[str, float]]:
    raw_summary = _mapping(
        result.get("fold_metric_summary"), f"{model_id}.fold_metric_summary"
    )
    summary: dict[str, dict[str, float]] = {}
    for metric in REPORTED_METRICS:
        raw_metric = _mapping(raw_summary.get(metric), f"{model_id}.{metric}")
        summary[metric] = {
            "mean": _number(raw_metric.get("mean"), f"{model_id}.{metric}.mean"),
            "standard_deviation": _number(
                raw_metric.get("standard_deviation"),
                f"{model_id}.{metric}.standard_deviation",
            ),
            "minimum": _number(raw_metric.get("minimum"), f"{model_id}.{metric}.minimum"),
            "maximum": _number(raw_metric.get("maximum"), f"{model_id}.{metric}.maximum"),
        }
    return summary


def _pooled_metrics(
    result: dict[str, Any], model_id: str, key: str
) -> dict[str, Any]:
    raw = _mapping(result.get(key), f"{model_id}.{key}")
    metrics: dict[str, Any] = {}
    for metric in REPORTED_METRICS:
        metrics[metric] = _number(raw.get(metric), f"{model_id}.{key}.{metric}")
    metrics["positive_rate"] = _number(
        raw.get("positive_rate"), f"{model_id}.{key}.positive_rate"
    )
    metrics["sample_count"] = _integer(
        raw.get("sample_count"), f"{model_id}.{key}.sample_count"
    )
    metrics["decision_threshold"] = _number(
        raw.get("decision_threshold"), f"{model_id}.{key}.decision_threshold"
    )
    metrics["confusion_matrix"] = {
        name: _integer(raw.get(name), f"{model_id}.{key}.{name}")
        for name in CONFUSION_KEYS
    }
    return metrics


def _summarise_model_result(
    result: dict[str, Any], pooled_key: str
) -> dict[str, Any]:
    model_id = _string(result.get("model_id"), "model_id")
    return {
        "model_id": model_id,
        "model_name": _string(result.get("model_name"), f"{model_id}.model_name"),
        "fold_metrics": _metric_summary(result, model_id),
        "pooled_oof_metrics_descriptive_only": _pooled_metrics(
            result, model_id, pooled_key
        ),
        "oof_prediction_count": _integer(
            result.get("oof_prediction_count"), f"{model_id}.oof_prediction_count"
        ),
        "model_parameters": _mapping(
            result.get("model_parameters"), f"{model_id}.model_parameters"
        ),
    }


def _validate_inner_cv(
    inner_cv: dict[str, Any], cold_drug_split: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _require_equal(inner_cv.get("outer_policy"), "cold_drug", "inner CV outer_policy")
    _require_equal(
        inner_cv.get("cv_scope"),
        "frozen_outer_training_partition_only",
        "inner CV scope",
    )
    _require_equal(
        inner_cv.get("outer_test_partition_used"),
        False,
        "inner CV outer_test_partition_used",
    )
    _require_equal(inner_cv.get("label_column"), PRIMARY_LABEL, "inner CV label")

    n_splits = _integer(inner_cv.get("n_splits"), "inner CV n_splits")
    if n_splits < 2:
        raise FinalSynthesisError("inner CV n_splits must be at least two.")
    input_pairs = _integer(inner_cv.get("input_pair_count"), "inner CV input_pair_count")
    input_drugs = _integer(inner_cv.get("input_drug_count"), "inner CV input_drug_count")
    _require_equal(
        input_pairs,
        _integer(cold_drug_split.get("train_pair_count"), "cold_drug train_pair_count"),
        "inner CV input_pair_count versus cold-drug train_pair_count",
    )
    _require_equal(
        input_drugs,
        _integer(cold_drug_split.get("train_drug_count"), "cold_drug train_drug_count"),
        "inner CV input_drug_count versus cold-drug train_drug_count",
    )

    results = _model_results_by_id(inner_cv, "inner CV")
    for model_id, result in results.items():
        _require_equal(
            _integer(result.get("oof_prediction_count"), f"{model_id}.oof_prediction_count"),
            input_pairs,
            f"{model_id} OOF coverage",
        )
        fold_results = _list(result.get("fold_results"), f"{model_id}.fold_results")
        _require_equal(len(fold_results), n_splits, f"{model_id} fold-result count")
        for raw_fold in fold_results:
            fold = _mapping(raw_fold, f"{model_id} fold result")
            _require_equal(
                fold.get("drug_overlap_count"), 0, f"{model_id} fold drug overlap"
            )

    return (
        {
            "outer_policy": "cold_drug",
            "cv_scope": "frozen_outer_training_partition_only",
            "outer_test_partition_used": False,
            "label_column": PRIMARY_LABEL,
            "n_splits": n_splits,
            "random_state": _integer(inner_cv.get("random_state"), "inner CV random_state"),
            "group_column": _string(inner_cv.get("group_column"), "inner CV group_column"),
            "input_pair_count": input_pairs,
            "input_drug_count": input_drugs,
            "input_feature_count": _integer(
                inner_cv.get("input_feature_count"), "inner CV input_feature_count"
            ),
            "primary_comparison_metric": _string(
                inner_cv.get("primary_comparison_metric"), "primary comparison metric"
            ),
        },
        results,
    )


def _validate_threshold_sensitivity(
    threshold_report: dict[str, Any], inner_contract: dict[str, Any]
) -> dict[str, Any]:
    _require_equal(
        threshold_report.get("outer_policy"), "cold_drug", "threshold outer_policy"
    )
    _require_equal(
        threshold_report.get("outer_test_partition_used"),
        False,
        "threshold outer_test_partition_used",
    )
    _require_equal(
        threshold_report.get("outer_test_outcomes_selected"),
        False,
        "threshold outer_test_outcomes_selected",
    )
    _require_equal(
        _integer(threshold_report.get("n_splits"), "threshold n_splits"),
        inner_contract["n_splits"],
        "threshold n_splits versus inner CV n_splits",
    )
    _require_equal(
        _integer(threshold_report.get("input_pair_count"), "threshold input_pair_count"),
        inner_contract["input_pair_count"],
        "threshold input_pair_count versus inner CV",
    )
    _require_equal(
        _integer(threshold_report.get("input_drug_count"), "threshold input_drug_count"),
        inner_contract["input_drug_count"],
        "threshold input_drug_count versus inner CV",
    )

    model_selection = _mapping(
        threshold_report.get("model_selection"), "threshold model_selection"
    )
    _require_equal(
        model_selection.get("selection_reopened"), False, "threshold selection_reopened"
    )
    _require_equal(
        model_selection.get("hyperparameter_tuning_performed"),
        False,
        "threshold hyperparameter_tuning_performed",
    )
    _require_equal(
        model_selection.get("selection_metric"),
        "average_precision",
        "threshold selection metric",
    )

    variants = _list(threshold_report.get("variants"), "threshold variants")
    indexed: dict[str, dict[str, Any]] = {}
    for raw_variant in variants:
        variant = _mapping(raw_variant, "threshold variant")
        variant_id = _string(variant.get("variant_id"), "threshold variant_id")
        if variant_id in indexed:
            raise FinalSynthesisError(f"Duplicate threshold variant {variant_id!r}.")
        indexed[variant_id] = variant
    if tuple(indexed) != (PRIMARY_VARIANT, SENSITIVITY_VARIANT):
        raise FinalSynthesisError(
            "Threshold variants must be primary 1,000 nM then sensitivity 100 nM."
        )
    _require_equal(
        indexed[PRIMARY_VARIANT].get("label_column"),
        PRIMARY_LABEL,
        "primary threshold label",
    )
    _require_equal(
        indexed[SENSITIVITY_VARIANT].get("label_column"),
        SENSITIVITY_LABEL,
        "sensitivity threshold label",
    )
    _require_equal(
        _number(indexed[PRIMARY_VARIANT].get("kd_threshold_nM"), "primary Kd threshold"),
        1000.0,
        "primary Kd threshold",
    )
    _require_equal(
        _number(indexed[SENSITIVITY_VARIANT].get("kd_threshold_nM"), "sensitivity Kd threshold"),
        100.0,
        "sensitivity Kd threshold",
    )

    model_results: dict[str, list[dict[str, Any]]] = {}
    for variant_id, variant in indexed.items():
        results = _model_results_by_id(variant, f"threshold {variant_id}")
        model_results[variant_id] = [
            _summarise_model_result(
                results[model_id], "pooled_oof_metrics_descriptive_only"
            )
            for model_id in EXPECTED_MODEL_IDS
        ]

    return {
        "analysis_scope": _string(
            threshold_report.get("analysis_scope"), "threshold analysis_scope"
        ),
        "outer_policy": "cold_drug",
        "outer_test_partition_used": False,
        "outer_test_outcomes_selected": False,
        "n_splits": inner_contract["n_splits"],
        "random_state": _integer(
            threshold_report.get("random_state"), "threshold random_state"
        ),
        "model_selection": model_selection,
        "variants": [
            {
                "variant_id": variant_id,
                "label_column": _string(
                    indexed[variant_id].get("label_column"),
                    f"{variant_id} label_column",
                ),
                "kd_threshold_nM": _number(
                    indexed[variant_id].get("kd_threshold_nM"),
                    f"{variant_id} Kd threshold",
                ),
                "pKd_threshold": _number(
                    indexed[variant_id].get("pKd_threshold"),
                    f"{variant_id} pKd threshold",
                ),
                "positive_count": _integer(
                    indexed[variant_id].get("positive_count"),
                    f"{variant_id} positive_count",
                ),
                "negative_count": _integer(
                    indexed[variant_id].get("negative_count"),
                    f"{variant_id} negative_count",
                ),
                "positive_rate": _number(
                    indexed[variant_id].get("positive_rate"),
                    f"{variant_id} positive_rate",
                ),
                "model_results": model_results[variant_id],
            }
            for variant_id in (PRIMARY_VARIANT, SENSITIVITY_VARIANT)
        ],
    }


def _collision_summary(collision_report: dict[str, Any]) -> dict[str, Any]:
    _require_equal(
        collision_report.get("model_predictions_used"),
        False,
        "collision-audit model_predictions_used",
    )
    _require_equal(
        collision_report.get("outcome_values_used"),
        False,
        "collision-audit outcome_values_used",
    )

    def entity_summary(name: str) -> dict[str, int]:
        entity = _mapping(collision_report.get(name), name)
        return {
            "entity_count": _integer(entity.get("entity_count"), f"{name}.entity_count"),
            "raw_duplicate_group_count": _integer(
                entity.get("raw_duplicate_group_count"),
                f"{name}.raw_duplicate_group_count",
            ),
            "exact_feature_collision_group_count": _integer(
                entity.get("exact_feature_collision_group_count"),
                f"{name}.exact_feature_collision_group_count",
            ),
            "distinct_raw_feature_collision_pair_count": _integer(
                entity.get("distinct_raw_feature_collision_pair_count"),
                f"{name}.distinct_raw_feature_collision_pair_count",
            ),
        }

    return {
        "audit_scope": _string(collision_report.get("audit_scope"), "collision audit scope"),
        "model_predictions_used": False,
        "outcome_values_used": False,
        "drug": entity_summary("drug_audit"),
        "target": entity_summary("target_audit"),
        "interpretation_limits": _list(
            collision_report.get("interpretation_limits"),
            "collision interpretation_limits",
        ),
    }


def _artifact_manifest(paths: dict[str, Path]) -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "path": path.as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    ]


def _rank_primary_models(model_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        model_results,
        key=lambda result: result["fold_metrics"]["average_precision"]["mean"],
        reverse=True,
    )


def build_final_synthesis(
    *,
    inner_cv_report: dict[str, Any],
    threshold_report: dict[str, Any],
    split_audit: dict[str, Any],
    collision_report: dict[str, Any],
    input_paths: dict[str, Path],
    source_git_commit: str,
) -> FinalSynthesisRun:
    """Validate frozen artefacts and return a final JSON/Markdown synthesis."""
    source_commit = _string(source_git_commit, "source_git_commit")
    cold_drug_split = _cold_drug_split(split_audit)
    inner_contract, inner_results = _validate_inner_cv(inner_cv_report, cold_drug_split)
    primary_models = [
        _summarise_model_result(inner_results[model_id], "pooled_oof_metrics")
        for model_id in EXPECTED_MODEL_IDS
    ]
    threshold = _validate_threshold_sensitivity(threshold_report, inner_contract)
    selected_model_id = _string(
        threshold["model_selection"].get("primary_selected_model_id"),
        "primary_selected_model_id",
    )
    ranked_models = _rank_primary_models(primary_models)
    if ranked_models[0]["model_id"] != selected_model_id:
        raise FinalSynthesisError(
            "Recorded primary model selection does not match the primary mean "
            "average-precision ranking."
        )
    runner_up = ranked_models[1]
    selection_margin = (
        ranked_models[0]["fold_metrics"]["average_precision"]["mean"]
        - runner_up["fold_metrics"]["average_precision"]["mean"]
    )

    manifest_paths = dict(input_paths)
    requirements = _read_requirements(manifest_paths["requirements_file"])
    dataset_source_commit = _dataset_commit_from_provenance(
        manifest_paths["dataset_provenance"]
    )
    collision = _collision_summary(collision_report)
    split_summary = {
        "policy": "cold_drug",
        "splitter_name": _string(cold_drug_split.get("splitter_name"), "cold_drug splitter"),
        "fold_index": _integer(cold_drug_split.get("fold_index"), "cold_drug fold index"),
        "random_state": _integer(cold_drug_split.get("random_state"), "cold_drug random_state"),
        "train_pair_count": _integer(cold_drug_split.get("train_pair_count"), "cold_drug train pairs"),
        "test_pair_count": _integer(cold_drug_split.get("test_pair_count"), "cold_drug test pairs"),
        "train_drug_count": _integer(cold_drug_split.get("train_drug_count"), "cold_drug train drugs"),
        "test_drug_count": _integer(cold_drug_split.get("test_drug_count"), "cold_drug test drugs"),
        "drug_overlap_count": 0,
        "target_overlap_count": _integer(cold_drug_split.get("target_overlap_count"), "cold_drug target overlap"),
        "train_positive_rate": _number(cold_drug_split.get("train_positive_rate"), "cold-drug train positive rate"),
        "test_positive_rate": _number(cold_drug_split.get("test_positive_rate"), "cold-drug test positive rate"),
    }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": {
            "name": "Davis binary drug-target interaction prediction",
            "dataset_representation": "DeepDTA-format Davis benchmark",
            "dataset_source_commit": dataset_source_commit,
            "source_git_commit": source_commit,
        },
        "reproducibility": {
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "requirements": requirements,
            "input_artifacts": _artifact_manifest(manifest_paths),
        },
        "leakage_aware_design": {
            "random_pair_warning": "Random pair splitting reuses related drugs and targets in both partitions and can overestimate unseen-entity generalisation.",
            "cold_drug_holdout": split_summary,
            "inner_cross_validation": inner_contract,
            "outer_holdout_disclosure": (
                "The final synthesis reads only inner-CV and sensitivity artefacts that record outer_test_partition_used=false. "
                "Earlier model-specific outer-holdout results remain part of the development history; therefore, the outer holdout must not be presented as a newly blind confirmatory test for any decision made after it was inspected."
            ),
        },
        "primary_model_comparison": {
            "label_column": PRIMARY_LABEL,
            "positive_definition": "Kd less than or equal to 1,000 nM (pKd greater than or equal to 6)",
            "principal_metric": "unweighted mean inner-fold average precision",
            "models": primary_models,
            "pre_specified_selection": {
                "selected_model_id": selected_model_id,
                "runner_up_model_id": runner_up["model_id"],
                "mean_average_precision_margin": selection_margin,
                "interpretation": (
                    "This is a descriptive ranking under five frozen inner cold-drug folds, not a formal statistical superiority result."
                ),
            },
        },
        "threshold_sensitivity": threshold,
        "feature_representation_audit": collision,
        "scientific_claim_boundaries": {
            "predictive_performance": [
                "The reported values estimate ranking and thresholded classification performance for this Davis benchmark under the stated cold-drug inner-CV design.",
                "Average precision is the principal imbalance-aware comparison metric; ROC-AUC is secondary, while accuracy, precision, recall, F1, and confusion matrices describe the fixed 0.5 operating point.",
            ],
            "statistical_evidence": [
                "Fold means, standard deviations, and pooled OOF summaries are descriptive. Five grouped folds do not by themselves establish statistical superiority or provide a p-value.",
                "The small primary average-precision difference between the selected random forest and the histogram gradient booster must be interpreted with fold variability in view.",
            ],
            "biological_interpretation": [
                "The benchmark labels are measured affinities, but computational predictions do not validate binding experimentally.",
                "Target representation collisions and coarse transparent descriptors limit entity-level biological interpretation; identical benchmark sequences do not establish biological equivalence.",
            ],
            "causal_claims": [
                "No association, feature importance, prediction, or threshold-sensitivity result establishes a biological mechanism, therapeutic effect, clinical utility, or causal drug-target relationship.",
            ],
        },
    }
    return FinalSynthesisRun(summary=summary, markdown=render_markdown(summary))


def _format(value: float) -> str:
    return f"{value:.4f}"


def _mean_sd(metrics: dict[str, Any], metric: str) -> str:
    value = metrics[metric]
    return f"{_format(value['mean'])} +/- {_format(value['standard_deviation'])}"


def _markdown_model_rows(models: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for model in models:
        metrics = model["fold_metrics"]
        rows.append(
            "| "
            + " | ".join(
                [
                    model["model_id"],
                    _mean_sd(metrics, "average_precision"),
                    _mean_sd(metrics, "roc_auc"),
                    _mean_sd(metrics, "accuracy"),
                    _mean_sd(metrics, "precision"),
                    _mean_sd(metrics, "recall"),
                    _mean_sd(metrics, "f1"),
                ]
            )
            + " |"
        )
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a concise, non-causal Markdown evidence report from the JSON."""
    study = _mapping(summary["study"], "study")
    reproducibility = _mapping(summary["reproducibility"], "reproducibility")
    design = _mapping(summary["leakage_aware_design"], "leakage_aware_design")
    primary = _mapping(summary["primary_model_comparison"], "primary_model_comparison")
    selection = _mapping(primary["pre_specified_selection"], "pre_specified_selection")
    primary_models = _list(primary["models"], "primary models")
    threshold = _mapping(summary["threshold_sensitivity"], "threshold_sensitivity")
    collision = _mapping(summary["feature_representation_audit"], "feature representation audit")
    claims = _mapping(summary["scientific_claim_boundaries"], "scientific claim boundaries")
    cold_split = _mapping(design["cold_drug_holdout"], "cold drug holdout")

    lines = [
        "# Davis Binary DTI: Final Evidence Synthesis",
        "",
        "## Scope",
        "",
        "This document consolidates already fixed, versioned evidence. It does not train a new model, tune hyperparameters, choose a probability threshold, or reopen model selection.",
        "",
        f"Dataset representation: {study['dataset_representation']}. Upstream DeepDTA commit: `{study['dataset_source_commit']}`.",
        "",
        "## Leakage-aware evaluation design",
        "",
        f"The primary evaluation is a cold-drug design: {cold_split['train_drug_count']} training drugs and {cold_split['test_drug_count']} held-out drugs, with zero drug overlap. Targets overlap by design ({cold_split['target_overlap_count']} targets), so the claim is generalisation to unseen drugs rather than unseen drug-target pairs or unseen targets.",
        "",
        f"Inner validation uses {design['inner_cross_validation']['n_splits']} drug-grouped folds on {design['inner_cross_validation']['input_pair_count']} outer-training pairs. These artefacts explicitly record `outer_test_partition_used=false`.",
        "",
        f"{design['random_pair_warning']}",
        "",
        f"**Holdout disclosure:** {design['outer_holdout_disclosure']}",
        "",
        "## Primary 1,000 nM task",
        "",
        f"Positive label: `{primary['label_column']}` — {primary['positive_definition']}.",
        "",
        "Values are unweighted mean +/- standard deviation across frozen inner cold-drug folds. Average precision is the principal metric.",
        "",
        "| Model | AP | ROC-AUC | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_markdown_model_rows(primary_models),
        "",
        f"Pre-specified selection: `{selection['selected_model_id']}` by mean inner-fold AP. The next-ranked fixed candidate was `{selection['runner_up_model_id']}`; the AP margin was {_format(selection['mean_average_precision_margin'])}. {selection['interpretation']}",
        "",
        "### Selected-model pooled OOF confusion matrix",
        "",
        "The following fixed-threshold (0.5) confusion matrix is pooled across separately fitted inner folds and is descriptive, not a new independent test result.",
        "",
        "| True negatives | False positives | False negatives | True positives |",
        "| ---: | ---: | ---: | ---: |",
    ]
    selected = next(
        model for model in primary_models if model["model_id"] == selection["selected_model_id"]
    )
    matrix = selected["pooled_oof_metrics_descriptive_only"]["confusion_matrix"]
    lines.extend(
        [
            f"| {matrix['true_negative']} | {matrix['false_positive']} | {matrix['false_negative']} | {matrix['true_positive']} |",
            "",
            "## Affinity-threshold sensitivity",
            "",
            "The 100 nM task reuses the frozen 1,000 nM inner folds and fixed candidate configurations. It is descriptive only and does not replace the primary model-selection decision.",
            "",
            "| Variant | Positive rate | Model | AP | ROC-AUC |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for variant in _list(threshold["variants"], "threshold variants"):
        variant_map = _mapping(variant, "threshold variant")
        for model in _list(variant_map["model_results"], "threshold model results"):
            model_map = _mapping(model, "threshold model")
            metrics = _mapping(model_map["fold_metrics"], "threshold fold metrics")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{variant_map['kd_threshold_nM']:.0f} nM ({variant_map['label_column']})",
                        _format(variant_map["positive_rate"]),
                        model_map["model_id"],
                        _mean_sd(metrics, "average_precision"),
                        _mean_sd(metrics, "roc_auc"),
                    ]
                )
                + " |"
            )

    drug_audit = _mapping(collision["drug"], "drug collision audit")
    target_audit = _mapping(collision["target"], "target collision audit")
    lines.extend(
        [
            "",
            "Raw AP values should not be compared across the two label definitions because the prediction task and class prevalence differ.",
            "",
            "## Feature-representation limits",
            "",
            f"The unsupervised audit found {drug_audit['exact_feature_collision_group_count']} exact drug feature-collision groups and {target_audit['exact_feature_collision_group_count']} exact target feature-collision groups. The latter are associated with duplicated benchmark sequences; this limits distinguishability for the current representation and does not establish biological equivalence.",
            "",
            "## Scientific interpretation boundaries",
            "",
            "### Predictive performance",
            "",
            *[f"- {item}" for item in _list(claims["predictive_performance"], "predictive claims")],
            "",
            "### Statistical evidence",
            "",
            *[f"- {item}" for item in _list(claims["statistical_evidence"], "statistical claims")],
            "",
            "### Biological interpretation",
            "",
            *[f"- {item}" for item in _list(claims["biological_interpretation"], "biological claims")],
            "",
            "### Causal claims",
            "",
            *[f"- {item}" for item in _list(claims["causal_claims"], "causal claims")],
            "",
            "## Reproducibility record",
            "",
            f"Evidence source commit: `{study['source_git_commit']}`.",
            "",
            f"Execution environment: {reproducibility['python_implementation']} {reproducibility['python_version']} on {reproducibility['platform']}.",
            "",
            "The companion JSON records SHA-256 hashes for every input artefact and the complete requirements lock. Re-run the named upstream commands before this synthesis if any source artefact changes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_final_synthesis_summary(run: FinalSynthesisRun, output_path: Path) -> Path:
    """Write deterministic, sorted JSON evidence to a version-controlled path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def write_final_synthesis_markdown(run: FinalSynthesisRun, output_path: Path) -> Path:
    """Write the human-readable companion report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(run.markdown, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Validate all inputs and write the final Davis evidence package."""
    parser = argparse.ArgumentParser(
        description="Create a leakage-aware final evidence synthesis for Davis binary DTI."
    )
    parser.add_argument(
        "--inner-cv-report",
        type=Path,
        default=Path("reports/davis_inner_cold_drug_cv.json"),
    )
    parser.add_argument(
        "--threshold-sensitivity-report",
        type=Path,
        default=Path("reports/davis_threshold_sensitivity.json"),
    )
    parser.add_argument(
        "--split-audit-report",
        type=Path,
        default=Path("reports/davis_split_audit.json"),
    )
    parser.add_argument(
        "--collision-audit-report",
        type=Path,
        default=Path("reports/davis_feature_collision_audit.json"),
    )
    parser.add_argument(
        "--dataset-provenance",
        type=Path,
        default=Path("validation/dataset_provenance.md"),
    )
    parser.add_argument(
        "--requirements-file",
        type=Path,
        default=Path("requirements.txt"),
    )
    parser.add_argument(
        "--source-git-commit",
        required=True,
        help="Exact commit that contains the evaluated input artefacts.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_final_evidence_summary.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/davis_final_evidence_summary.md"),
    )
    args = parser.parse_args(argv)

    paths = {
        "inner_cv_report": args.inner_cv_report,
        "threshold_sensitivity_report": args.threshold_sensitivity_report,
        "split_audit_report": args.split_audit_report,
        "collision_audit_report": args.collision_audit_report,
        "dataset_provenance": args.dataset_provenance,
        "requirements_file": args.requirements_file,
    }
    try:
        run = build_final_synthesis(
            inner_cv_report=read_json_report(args.inner_cv_report),
            threshold_report=read_json_report(args.threshold_sensitivity_report),
            split_audit=read_json_report(args.split_audit_report),
            collision_report=read_json_report(args.collision_audit_report),
            input_paths=paths,
            source_git_commit=args.source_git_commit,
        )
        json_path = write_final_synthesis_summary(run, args.summary_output)
        markdown_path = write_final_synthesis_markdown(run, args.markdown_output)
    except (FinalSynthesisError, OSError, UnicodeError) as error:
        print(f"Final synthesis failed: {error}", file=sys.stderr)
        return 2

    selection = run.summary["primary_model_comparison"]["pre_specified_selection"]
    print(
        json.dumps(
            {
                "primary_selected_model_id": selection["selected_model_id"],
                "summary_output": json_path.as_posix(),
                "markdown_output": markdown_path.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
