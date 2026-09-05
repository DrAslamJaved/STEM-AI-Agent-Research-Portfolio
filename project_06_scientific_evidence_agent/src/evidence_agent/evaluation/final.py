"""Phase 7 final evaluation: compare frozen Phase 05/06 policies on one trace.

This module owns every piece of behaviour behind ``evaluate --config``: safe
config loading, artifact hash validation, trace freezing, applying both fixed
policies to the identical frozen trace, loading gold labels only after that
freeze, paired-bootstrap confidence intervals, and writing the result JSON.
``evidence_agent.cli`` only calls :func:`run_evaluate_command` and prints what
it returns -- it contains no evaluation logic of its own.

This is a held-out *development* evaluation, not an independent test: it
reuses ``claims_dev.jsonl``, the same split already used to report the Phase
06 citation-audit numbers. Nothing here rebuilds the BM25 index, retrains the
verifier, or recalibrates the citation-audit policy; every artifact is loaded
read-only and its SHA-256 is checked against ``configs/final.yaml`` first.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from evidence_agent.audit.bootstrap import build_claim_outcomes, paired_bootstrap_confidence_intervals
from evidence_agent.audit.calibration import (
    CALIBRATION_REPORT_SCHEMA,
    load_calibration_report,
    load_selected_policy,
)
from evidence_agent.audit.policy import PHASE_05_POLICY, apply_citation_audit_to_traces
from evidence_agent.data.acquisition import sha256_file
from evidence_agent.evaluation.final_config import (
    EVALUATION_LABEL,
    FinalEvaluationConfig,
    load_final_evaluation_config,
)
from evidence_agent.evaluation.verification import evaluate_verification_traces, write_verification_report
from evidence_agent.retrieval.bm25 import load_bm25_index
from evidence_agent.retrieval.scifact import load_runtime_claims, load_scifact_corpus
from evidence_agent.verification.agent import run_verification_agent
from evidence_agent.verification.models import load_verifier_bundle
from evidence_agent.verification.scifact import load_gold_claim_annotations


RESULT_SCHEMA_VERSION = "evidence_agent_final_evaluation_v1"
RUNTIME_TRACE_SCHEMA = "evidence_agent_final_evaluation_runtime_trace_v1"


class FinalEvaluationError(ValueError):
    """Raised when frozen Phase 7 artifacts are missing, mismatched, or unsafe."""


def _relative_posix_path(path: Path, project_root: Path) -> str:
    """Render ``path`` relative to ``project_root`` with forward slashes.

    Report fields must stay reproducible across machines and working
    directories, so absolute, platform-specific paths (e.g. ``C:\\...``) are
    never written to the result JSON. A path outside ``project_root`` falls
    back to its resolved absolute POSIX form rather than raising, since that
    can only happen for artifacts deliberately declared outside the project.
    """
    resolved = path if path.is_absolute() else path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validated_artifact_sha256(path: Path, expected_sha256: str, name: str) -> str:
    try:
        actual = sha256_file(path)
    except OSError as error:
        raise FinalEvaluationError(f"Unable to read declared artifact {name} at {path}: {error}") from error
    if actual != expected_sha256:
        raise FinalEvaluationError(
            f"{name}: SHA-256 mismatch for {path} "
            f"(declared {expected_sha256}, actual {actual})."
        )
    return actual


def _citation_audit_deltas(
    selected_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object],
) -> dict[str, float]:
    """Return selected-minus-baseline changes; negative unsupported rate is better."""

    def nested_metric(summary: Mapping[str, object], group: str, metric: str) -> float:
        values = summary[group]
        if not isinstance(values, Mapping):  # pragma: no cover - internal invariant
            raise FinalEvaluationError(f"{group} must be a metric object.")
        return float(values[metric])

    return {
        "citation_correctness_f1": nested_metric(selected_summary, "citation_correctness", "f1")
        - nested_metric(baseline_summary, "citation_correctness", "f1"),
        "claim_macro_f1": nested_metric(selected_summary, "claim_classification", "macro_f1")
        - nested_metric(baseline_summary, "claim_classification", "macro_f1"),
        "coverage": float(selected_summary["coverage"]) - float(baseline_summary["coverage"]),
        "evidence_sentence_f1": nested_metric(selected_summary, "evidence_sentence", "f1")
        - nested_metric(baseline_summary, "evidence_sentence", "f1"),
        "faithfulness": float(selected_summary["faithfulness"]) - float(baseline_summary["faithfulness"]),
        "unsupported_assertion_rate": float(selected_summary["unsupported_assertion_rate"])
        - float(baseline_summary["unsupported_assertion_rate"]),
    }


def run_final_evaluation(config: FinalEvaluationConfig) -> dict[str, object]:
    """Run the complete Phase 7 pipeline and write its result JSON.

    Ordering is load-bearing: the raw runtime trace is written to disk, and
    its SHA-256 recorded, before ``load_gold_claim_annotations`` is called.
    Both policies are then applied to that one already-frozen trace object.
    """
    artifacts = config.artifacts

    corpus_sha256 = _validated_artifact_sha256(artifacts.corpus.path, artifacts.corpus.sha256, "corpus")
    claims_dev_sha256 = _validated_artifact_sha256(
        artifacts.claims_dev.path, artifacts.claims_dev.sha256, "claims_dev"
    )
    bm25_index_sha256 = _validated_artifact_sha256(
        artifacts.bm25_index.path, artifacts.bm25_index.sha256, "bm25_index"
    )
    verifier_model_sha256 = _validated_artifact_sha256(
        artifacts.verifier_model.path, artifacts.verifier_model.sha256, "verifier_model"
    )
    calibration_report_sha256 = _validated_artifact_sha256(
        artifacts.calibration_report.path,
        artifacts.calibration_report.sha256,
        "calibration_report",
    )
    train_claims_sha256 = (
        _validated_artifact_sha256(
            artifacts.train_claims.path, artifacts.train_claims.sha256, "train_claims"
        )
        if artifacts.train_claims is not None
        else None
    )

    corpus = load_scifact_corpus(artifacts.corpus.path)
    bm25_index = load_bm25_index(artifacts.bm25_index.path)
    bundle = load_verifier_bundle(artifacts.verifier_model.path)
    if bm25_index.corpus_sha256 != corpus_sha256:
        raise FinalEvaluationError("BM25 index does not match the declared corpus SHA-256.")
    if bundle.corpus_sha256 != corpus_sha256:
        raise FinalEvaluationError("Verifier bundle does not match the declared corpus SHA-256.")
    if train_claims_sha256 is not None and bundle.training_claims_sha256 != train_claims_sha256:
        raise FinalEvaluationError(
            "Verifier bundle was not trained on the declared train_claims artifact."
        )

    calibration_report = load_calibration_report(artifacts.calibration_report.path)
    selected_policy = load_selected_policy(artifacts.calibration_report.path)
    if train_claims_sha256 is not None:
        calibration_data = calibration_report.get("data")
        if not isinstance(calibration_data, Mapping):
            raise FinalEvaluationError("Calibration report is missing its data provenance block.")
        calibration_training = calibration_data.get("main_training_claims")
        if not isinstance(calibration_training, Mapping) or calibration_training.get("sha256") != train_claims_sha256:
            raise FinalEvaluationError(
                "Calibration report and declared train_claims do not identify the same training split."
            )

    runtime_claims = load_runtime_claims(artifacts.claims_dev.path)
    started_at = time.perf_counter()
    raw_runtime_traces = run_verification_agent(
        bundle,
        bm25_index,
        corpus,
        runtime_claims,
        retrieval_k=config.runtime.retrieval_k,
        assertion_threshold=0.0,
        sentence_threshold=0.0,
        max_sentences_per_citation=max(
            PHASE_05_POLICY.max_sentences_per_citation,
            selected_policy.max_sentences_per_citation,
        ),
    )
    runtime_elapsed_seconds = time.perf_counter() - started_at

    # The raw, gold-free trace is frozen to disk -- and its SHA-256 recorded --
    # before any development gold field (evidence, cited_doc_ids) is read.
    trace_payload = {
        "schema_version": RUNTIME_TRACE_SCHEMA,
        "traces": [trace.as_dict() for trace in raw_runtime_traces],
    }
    write_verification_report(trace_payload, config.output.trace_path)
    trace_sha256 = sha256_file(config.output.trace_path)
    trace_artifact = {
        "path": _relative_posix_path(config.output.trace_path, config.project_root),
        "schema_version": RUNTIME_TRACE_SCHEMA,
        "sha256": trace_sha256,
        "trace_count": len(raw_runtime_traces),
    }

    selected_traces = apply_citation_audit_to_traces(raw_runtime_traces, selected_policy)
    phase_05_traces = apply_citation_audit_to_traces(raw_runtime_traces, PHASE_05_POLICY)
    raw_claim_order = [trace.decision.claim_id for trace in raw_runtime_traces]
    if (
        [trace.decision.claim_id for trace in selected_traces] != raw_claim_order
        or [trace.decision.claim_id for trace in phase_05_traces] != raw_claim_order
    ):
        raise FinalEvaluationError(  # pragma: no cover - internal invariant
            "Selected and Phase 05 policies must be applied to the identical frozen claim order."
        )

    gold_annotations = load_gold_claim_annotations(artifacts.claims_dev.path, artifacts.corpus.path)
    selected_evaluation = evaluate_verification_traces(selected_traces, gold_annotations)
    phase_05_evaluation = evaluate_verification_traces(phase_05_traces, gold_annotations)
    selected_summary = selected_evaluation.summary_dict()
    phase_05_summary = phase_05_evaluation.summary_dict()
    comparison = _citation_audit_deltas(selected_summary, phase_05_summary)

    bootstrap_result: dict[str, object] | None = None
    if config.bootstrap.enabled:
        selected_outcomes = build_claim_outcomes(selected_traces, gold_annotations)
        phase_05_outcomes = build_claim_outcomes(phase_05_traces, gold_annotations)
        bootstrap_result = paired_bootstrap_confidence_intervals(
            selected_outcomes,
            phase_05_outcomes,
            resamples=config.bootstrap.resamples,
            seed=config.bootstrap.seed,
            confidence_level=config.bootstrap.confidence_level,
        )

    report: dict[str, object] = {
        "algorithm": "bm25_lexical_verifier_cross_validated_citation_audit",
        "artifacts": {
            "bm25_index": {
                "path": _relative_posix_path(artifacts.bm25_index.path, config.project_root),
                "sha256": bm25_index_sha256,
            },
            "calibration_report": {
                "path": _relative_posix_path(artifacts.calibration_report.path, config.project_root),
                "sha256": calibration_report_sha256,
                "schema_version": CALIBRATION_REPORT_SCHEMA,
            },
            "claims_dev": {
                "path": _relative_posix_path(artifacts.claims_dev.path, config.project_root),
                "sha256": claims_dev_sha256,
            },
            "corpus": {
                "path": _relative_posix_path(artifacts.corpus.path, config.project_root),
                "sha256": corpus_sha256,
            },
            "verifier_model": {
                "path": _relative_posix_path(artifacts.verifier_model.path, config.project_root),
                "sha256": verifier_model_sha256,
            },
            **(
                {
                    "train_claims": {
                        "path": _relative_posix_path(artifacts.train_claims.path, config.project_root),
                        "sha256": train_claims_sha256,
                    }
                }
                if artifacts.train_claims is not None
                else {}
            ),
        },
        "baseline_phase_05_policy": {"policy": PHASE_05_POLICY.as_dict(), "summary": phase_05_summary},
        "bootstrap_confidence_intervals": bootstrap_result,
        "comparison_to_phase_05_policy": comparison,
        "config_path": _relative_posix_path(config.config_path, config.project_root),
        "corpus_sha256": corpus_sha256,
        "evaluation_label": EVALUATION_LABEL,
        "is_independent_test": False,
        "output": {
            "agent_trace_path": _relative_posix_path(config.output.agent_trace_path, config.project_root),
            "report_path": _relative_posix_path(config.output.report_path, config.project_root),
            "result_path": _relative_posix_path(config.output.result_path, config.project_root),
            "trace_path": _relative_posix_path(config.output.trace_path, config.project_root),
        },
        "phase_05_decisions": [trace.decision_dict() for trace in phase_05_traces],
        "runtime_settings": {
            "retrieval_k": config.runtime.retrieval_k,
            "selected_policy": selected_policy.as_dict(),
        },
        "runtime_timing": {
            "claim_count": len(runtime_claims),
            "per_claim_milliseconds": 1_000 * runtime_elapsed_seconds / len(runtime_claims),
            "total_seconds": runtime_elapsed_seconds,
        },
        "schema_version": RESULT_SCHEMA_VERSION,
        "selected_decisions": [trace.decision_dict() for trace in selected_traces],
        "selected_policy": selected_policy.as_dict(),
        "summary": selected_summary,
        "trace_artifact": trace_artifact,
    }
    write_verification_report(report, config.output.result_path)
    return report


def run_evaluate_command(config_path: Path) -> dict[str, object]:
    """Load ``configs/final.yaml``, run the pipeline, and summarise for the CLI.

    ``result_path``/``trace_path`` here are the real, directly-openable
    filesystem locations (from ``config.output``), unlike the project-relative
    provenance strings recorded inside the result JSON itself.
    """
    config = load_final_evaluation_config(config_path)
    report = run_final_evaluation(config)
    return {
        "bootstrap_confidence_intervals": report["bootstrap_confidence_intervals"],
        "comparison_to_phase_05_policy": report["comparison_to_phase_05_policy"],
        "evaluation_label": report["evaluation_label"],
        "result_path": str(config.output.result_path),
        "selected_policy": report["selected_policy"],
        "summary": report["summary"],
        "trace_path": str(config.output.trace_path),
    }
