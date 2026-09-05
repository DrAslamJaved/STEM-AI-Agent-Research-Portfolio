"""Phase 8 controlled SciFact experiments.

The module compares two deterministic arms on the *same* frozen, gold-free
runtime trace:

* ``direct_rag``: cite the first BM25-retrieved document with its highest
  scoring rationale sentences and no abstention policy;
* ``audited_agent``: apply the fixed Phase 6 cross-validated citation-audit
  policy to that exact trace.

All dev-set gold fields are loaded only after both arms' predictions and the
raw trace are written to disk.  The report contains both the project's audit
metrics and SciFact-compatible abstract/sentence scoring, plus paired
bootstrap intervals for both comparisons.  It is deliberately labelled a
held-out *development* evaluation rather than an independent test.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from evidence_agent.audit.bootstrap import build_claim_outcomes, paired_bootstrap_confidence_intervals
from evidence_agent.audit.calibration import CALIBRATION_REPORT_SCHEMA, load_calibration_report, load_selected_policy
from evidence_agent.audit.policy import CitationAuditPolicy, apply_citation_audit_to_traces
from evidence_agent.data.acquisition import sha256_file
from evidence_agent.data.schemas import AuditDecision, Citation, Verdict
from evidence_agent.evaluation.controlled_config import (
    EVALUATION_LABEL,
    ControlledExperimentsConfig,
    load_controlled_experiments_config,
)
from evidence_agent.evaluation.verification import evaluate_verification_traces, write_verification_report
from evidence_agent.retrieval.bm25 import load_bm25_index
from evidence_agent.retrieval.scifact import load_runtime_claims, load_scifact_corpus
from evidence_agent.verification.agent import CandidateTrace, VerificationTrace, run_verification_agent
from evidence_agent.verification.models import load_verifier_bundle
from evidence_agent.verification.scifact import GoldClaimAnnotation, load_gold_claim_annotations


RESULT_SCHEMA_VERSION = "evidence_agent_controlled_experiments_v1"
RUNTIME_TRACE_SCHEMA = "evidence_agent_controlled_experiments_runtime_trace_v1"
OFFICIAL_BOOTSTRAP_METRICS = ("abstract_level_f1", "sentence_level_f1")


class ControlledExperimentsError(ValueError):
    """Raised when a controlled experiment would be invalid or unsafe."""


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_ratio(2 * precision * recall, precision + recall)


def _relative_posix_path(path: Path, project_root: Path) -> str:
    """Record project-contained paths portably in result artifacts."""
    resolved = path if path.is_absolute() else path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validated_artifact_sha256(path: Path, expected_sha256: str, name: str) -> str:
    try:
        actual = sha256_file(path)
    except OSError as error:
        raise ControlledExperimentsError(
            f"Unable to read declared artifact {name} at {path}: {error}"
        ) from error
    if actual != expected_sha256:
        raise ControlledExperimentsError(
            f"{name}: SHA-256 mismatch for {path} "
            f"(declared {expected_sha256}, actual {actual})."
        )
    return actual


def _select_top_sentence_ids(
    trace: VerificationTrace, maximum: int
) -> tuple[tuple[int, ...], float, CandidateTrace] | None:
    """Choose direct-RAG evidence from the first retrieved candidate only."""
    if not trace.candidates:
        return None
    candidate = min(
        trace.candidates,
        key=lambda item: (item.retrieval_rank, item.retrieval_hit.doc_id),
    )
    selected_scores = sorted(
        candidate.sentence_scores,
        key=lambda score: (-score.probability, score.input.sentence_id),
    )[:maximum]
    if not selected_scores:
        return None
    sentence_ids = tuple(sorted(score.input.sentence_id for score in selected_scores))
    return sentence_ids, selected_scores[0].probability, candidate


def apply_direct_rag_to_trace(trace: VerificationTrace, *, max_sentences_per_citation: int) -> VerificationTrace:
    """Produce a direct retrieval-to-citation baseline without audit abstention.

    The arm intentionally chooses the first BM25 candidate.  It may abstain
    only if retrieval produced no candidate or that candidate has no abstract
    sentences; it never uses an assertion or sentence acceptance threshold.
    """
    if (
        isinstance(max_sentences_per_citation, bool)
        or not isinstance(max_sentences_per_citation, int)
        or not 1 <= max_sentences_per_citation <= 3
    ):
        raise ControlledExperimentsError(
            "max_sentences_per_citation must be an integer in [1, 3]."
        )
    selected = _select_top_sentence_ids(trace, max_sentences_per_citation)
    if selected is None:
        decision = AuditDecision(
            claim_id=trace.decision.claim_id,
            verdict=Verdict.NO_EVIDENCE,
            confidence=1.0,
        )
        return VerificationTrace(decision=decision, candidates=trace.candidates)

    sentence_ids, sentence_probability, candidate = selected
    stance = candidate.selected_stance
    stance_probability = candidate.stance.probability(stance)
    confidence = math.sqrt(stance_probability * sentence_probability)
    decision = AuditDecision(
        claim_id=trace.decision.claim_id,
        verdict=stance,
        confidence=confidence,
        citations=(
            Citation(
                doc_id=candidate.retrieval_hit.doc_id,
                sentence_ids=sentence_ids,
                stance=stance,
            ),
        ),
    )
    return VerificationTrace(decision=decision, candidates=trace.candidates)


def apply_direct_rag_to_traces(
    traces: Sequence[VerificationTrace], *, max_sentences_per_citation: int
) -> tuple[VerificationTrace, ...]:
    """Apply the fixed direct-RAG baseline to one frozen trace collection."""
    claim_ids = [trace.decision.claim_id for trace in traces]
    if len(claim_ids) != len(set(claim_ids)):
        raise ControlledExperimentsError("Frozen traces must have unique claim IDs.")
    return tuple(
        apply_direct_rag_to_trace(
            trace, max_sentences_per_citation=max_sentences_per_citation
        )
        for trace in traces
    )


def _prediction_documents(decision: AuditDecision) -> dict[int, tuple[Verdict, tuple[int, ...]]]:
    """Normalize a decision into the official SciFact document-keyed schema."""
    documents: dict[int, tuple[Verdict, set[int]]] = {}
    for citation in decision.citations:
        existing = documents.get(citation.doc_id)
        if existing is None:
            documents[citation.doc_id] = (citation.stance, set(citation.sentence_ids))
            continue
        existing_stance, existing_sentences = existing
        if existing_stance is not citation.stance:
            raise ControlledExperimentsError(
                f"Claim {decision.claim_id} predicts conflicting stances for document {citation.doc_id}."
            )
        existing_sentences.update(citation.sentence_ids)
    return {
        doc_id: (stance, tuple(sorted(sentence_ids)))
        for doc_id, (stance, sentence_ids) in documents.items()
    }


def official_predictions_from_traces(traces: Sequence[VerificationTrace]) -> tuple[dict[str, object], ...]:
    """Serialize frozen decisions in the official SciFact submission shape."""
    seen_claim_ids: set[int] = set()
    predictions: list[dict[str, object]] = []
    for trace in traces:
        decision = trace.decision
        if decision.claim_id in seen_claim_ids:
            raise ControlledExperimentsError(
                f"Duplicate runtime decision for claim {decision.claim_id}."
            )
        seen_claim_ids.add(decision.claim_id)
        evidence = {
            str(doc_id): {"label": str(stance), "sentences": list(sentence_ids)}
            for doc_id, (stance, sentence_ids) in sorted(_prediction_documents(decision).items())
        }
        predictions.append({"id": decision.claim_id, "evidence": evidence})
    return tuple(predictions)


def write_official_predictions(predictions: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write one deterministic official-format prediction object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(prediction, sort_keys=True) + "\n" for prediction in predictions),
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class OfficialSciFactEvaluation:
    """SciFact-compatible abstract and rationale sentence precision/recall/F1."""

    claim_count: int
    abstract_true_positive: int
    abstract_predicted: int
    abstract_gold: int
    sentence_true_positive: int
    sentence_predicted: int
    sentence_gold: int

    def summary_dict(self) -> dict[str, object]:
        abstract_precision = _safe_ratio(self.abstract_true_positive, self.abstract_predicted)
        abstract_recall = _safe_ratio(self.abstract_true_positive, self.abstract_gold)
        sentence_precision = _safe_ratio(self.sentence_true_positive, self.sentence_predicted)
        sentence_recall = _safe_ratio(self.sentence_true_positive, self.sentence_gold)
        return {
            "abstract_level": {
                "f1": _f1(abstract_precision, abstract_recall),
                "precision": abstract_precision,
                "recall": abstract_recall,
            },
            "claim_count": self.claim_count,
            "sentence_level": {
                "f1": _f1(sentence_precision, sentence_recall),
                "precision": sentence_precision,
                "recall": sentence_recall,
            },
        }


def _gold_citations_by_document(annotation: GoldClaimAnnotation) -> dict[int, tuple[Citation, ...]]:
    grouped: dict[int, list[Citation]] = {}
    for citation in annotation.citations:
        grouped.setdefault(citation.doc_id, []).append(citation)
    return {doc_id: tuple(citations) for doc_id, citations in grouped.items()}


def evaluate_official_scifact_traces(
    traces: Sequence[VerificationTrace],
    gold_annotations: Mapping[int, GoldClaimAnnotation],
) -> OfficialSciFactEvaluation:
    """Score frozen traces with SciFact's document/stance/rationale semantics.

    Abstract credit requires a relevant document, correct stance, and a complete
    gold rationale set among the first three predicted sentences.  Sentence
    credit requires a correct document/stance and the whole corresponding gold
    rationale set among all predicted sentences.
    """
    trace_by_claim: dict[int, VerificationTrace] = {}
    for trace in traces:
        claim_id = trace.decision.claim_id
        if claim_id in trace_by_claim:
            raise ControlledExperimentsError(f"Duplicate runtime decision for claim {claim_id}.")
        trace_by_claim[claim_id] = trace
    missing = sorted(set(gold_annotations) - set(trace_by_claim))
    unexpected = sorted(set(trace_by_claim) - set(gold_annotations))
    if missing or unexpected:
        raise ControlledExperimentsError(
            f"Trace/gold claim populations differ; missing={missing}, unexpected={unexpected}."
        )

    abstract_true_positive = 0
    abstract_predicted = 0
    abstract_gold = 0
    sentence_true_positive = 0
    sentence_predicted = 0
    sentence_gold = 0

    for claim_id in sorted(gold_annotations):
        annotation = gold_annotations[claim_id]
        predicted_documents = _prediction_documents(trace_by_claim[claim_id].decision)
        gold_by_document = _gold_citations_by_document(annotation)
        abstract_gold += len(gold_by_document)
        gold_sentence_keys = {
            (doc_id, sentence_id)
            for doc_id, citations in gold_by_document.items()
            for citation in citations
            for sentence_id in citation.sentence_ids
        }
        sentence_gold += len(gold_sentence_keys)

        for doc_id, (predicted_stance, predicted_sentence_ids) in predicted_documents.items():
            abstract_predicted += 1
            sentence_predicted += len(predicted_sentence_ids)
            gold_citations = gold_by_document.get(doc_id, ())
            if not gold_citations:
                continue
            if any(citation.stance is not predicted_stance for citation in gold_citations):
                continue

            first_three = set(predicted_sentence_ids[:3])
            all_predicted = set(predicted_sentence_ids)
            abstract_correct = any(
                set(citation.sentence_ids).issubset(first_three)
                for citation in gold_citations
            )
            abstract_true_positive += int(abstract_correct)

            fully_covered_sentence_ids = {
                sentence_id
                for citation in gold_citations
                if set(citation.sentence_ids).issubset(all_predicted)
                for sentence_id in citation.sentence_ids
            }
            sentence_true_positive += sum(
                sentence_id in fully_covered_sentence_ids
                for sentence_id in predicted_sentence_ids
            )

    return OfficialSciFactEvaluation(
        claim_count=len(gold_annotations),
        abstract_true_positive=abstract_true_positive,
        abstract_predicted=abstract_predicted,
        abstract_gold=abstract_gold,
        sentence_true_positive=sentence_true_positive,
        sentence_predicted=sentence_predicted,
        sentence_gold=sentence_gold,
    )


def _resampled_traces_and_gold(
    claim_ids: Sequence[int],
    traces_by_claim: Mapping[int, VerificationTrace],
    gold_by_claim: Mapping[int, GoldClaimAnnotation],
) -> tuple[tuple[VerificationTrace, ...], dict[int, GoldClaimAnnotation]]:
    """Namespace duplicate bootstrap draws by occurrence index."""
    resampled_traces: list[VerificationTrace] = []
    resampled_gold: dict[int, GoldClaimAnnotation] = {}
    for occurrence, claim_id in enumerate(claim_ids):
        trace = traces_by_claim[claim_id]
        annotation = gold_by_claim[claim_id]
        resampled_traces.append(
            VerificationTrace(
                decision=replace(trace.decision, claim_id=occurrence),
                candidates=(),
            )
        )
        resampled_gold[occurrence] = replace(annotation, claim_id=occurrence)
    return tuple(resampled_traces), resampled_gold


def paired_bootstrap_official_confidence_intervals(
    audited_traces: Sequence[VerificationTrace],
    direct_rag_traces: Sequence[VerificationTrace],
    gold_annotations: Mapping[int, GoldClaimAnnotation],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Paired bootstrap CIs for official SciFact F1 deltas.

    The same claim-id draw is used in both arms.  Repeated draws receive an
    occurrence-specific id before scoring so their evidence keys cannot collapse.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ControlledExperimentsError("resamples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ControlledExperimentsError("seed must be a non-negative integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ControlledExperimentsError("confidence_level must lie in (0, 1).")

    audited_by_claim = {trace.decision.claim_id: trace for trace in audited_traces}
    direct_by_claim = {trace.decision.claim_id: trace for trace in direct_rag_traces}
    claim_ids = tuple(sorted(gold_annotations))
    if not claim_ids:
        raise ControlledExperimentsError("At least one labelled claim is required for bootstrap.")
    if set(audited_by_claim) != set(gold_annotations) or set(direct_by_claim) != set(gold_annotations):
        raise ControlledExperimentsError("Both experiment arms must cover the identical gold claim IDs.")

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in OFFICIAL_BOOTSTRAP_METRICS}
    for _ in range(resamples):
        draw_indices = rng.integers(0, len(claim_ids), size=len(claim_ids))
        drawn_claim_ids = [claim_ids[int(index)] for index in draw_indices]
        audited_draw, audited_gold = _resampled_traces_and_gold(
            drawn_claim_ids, audited_by_claim, gold_annotations
        )
        direct_draw, direct_gold = _resampled_traces_and_gold(
            drawn_claim_ids, direct_by_claim, gold_annotations
        )
        audited_summary = evaluate_official_scifact_traces(audited_draw, audited_gold).summary_dict()
        direct_summary = evaluate_official_scifact_traces(direct_draw, direct_gold).summary_dict()
        values["abstract_level_f1"].append(
            float(audited_summary["abstract_level"]["f1"])
            - float(direct_summary["abstract_level"]["f1"])
        )
        values["sentence_level_f1"].append(
            float(audited_summary["sentence_level"]["f1"])
            - float(direct_summary["sentence_level"]["f1"])
        )

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    return {
        "claim_count": len(claim_ids),
        "confidence_level": confidence_level,
        "metric": "audited_agent_minus_direct_rag_delta",
        "metrics": {
            metric: {
                "lower": float(np.percentile(metric_values, lower_percentile)),
                "mean": float(np.mean(metric_values)),
                "upper": float(np.percentile(metric_values, upper_percentile)),
            }
            for metric, metric_values in values.items()
        },
        "resamples": resamples,
        "seed": seed,
    }


def _audit_metric_deltas(
    audited_summary: Mapping[str, object], direct_summary: Mapping[str, object]
) -> dict[str, float]:
    """Return audited-agent minus direct-RAG changes for existing audit metrics."""

    def nested(summary: Mapping[str, object], group: str, metric: str) -> float:
        value = summary[group]
        if not isinstance(value, Mapping):
            raise ControlledExperimentsError(f"{group} must be a metric mapping.")
        return float(value[metric])

    return {
        "citation_correctness_f1": nested(audited_summary, "citation_correctness", "f1")
        - nested(direct_summary, "citation_correctness", "f1"),
        "claim_macro_f1": nested(audited_summary, "claim_classification", "macro_f1")
        - nested(direct_summary, "claim_classification", "macro_f1"),
        "coverage": float(audited_summary["coverage"]) - float(direct_summary["coverage"]),
        "evidence_sentence_f1": nested(audited_summary, "evidence_sentence", "f1")
        - nested(direct_summary, "evidence_sentence", "f1"),
        "faithfulness": float(audited_summary["faithfulness"]) - float(direct_summary["faithfulness"]),
        "unsupported_assertion_rate": float(audited_summary["unsupported_assertion_rate"])
        - float(direct_summary["unsupported_assertion_rate"]),
    }


def _official_deltas(audited: Mapping[str, object], direct: Mapping[str, object]) -> dict[str, float]:
    return {
        "abstract_level_f1": float(audited["abstract_level"]["f1"])
        - float(direct["abstract_level"]["f1"]),
        "sentence_level_f1": float(audited["sentence_level"]["f1"])
        - float(direct["sentence_level"]["f1"]),
    }


def run_adversarial_evaluator_suite() -> dict[str, object]:
    """Exercise official scoring against document/stance/rationale attacks.

    These are evaluator regression cases, not data used to tune either arm.
    They model the three ways a superficially plausible citation must fail:
    wrong document, wrong stance, and an incomplete multi-sentence rationale.
    """
    gold_citation = Citation(doc_id=10, sentence_ids=(0, 1), stance=Verdict.SUPPORT)
    gold = {1: GoldClaimAnnotation(1, Verdict.SUPPORT, (gold_citation,))}
    cases = {
        "wrong_document": Citation(doc_id=11, sentence_ids=(0, 1), stance=Verdict.SUPPORT),
        "wrong_stance": Citation(doc_id=10, sentence_ids=(0, 1), stance=Verdict.CONTRADICT),
        "incomplete_rationale": Citation(doc_id=10, sentence_ids=(0,), stance=Verdict.SUPPORT),
    }
    results: dict[str, dict[str, float | bool]] = {}
    for name, citation in cases.items():
        trace = VerificationTrace(
            decision=AuditDecision(1, citation.stance, 0.9, (citation,)), candidates=()
        )
        summary = evaluate_official_scifact_traces((trace,), gold).summary_dict()
        abstract_f1 = float(summary["abstract_level"]["f1"])
        sentence_f1 = float(summary["sentence_level"]["f1"])
        results[name] = {
            "abstract_level_f1": abstract_f1,
            "passed": abstract_f1 == 0.0 and sentence_f1 == 0.0,
            "sentence_level_f1": sentence_f1,
        }
    return {"all_passed": all(item["passed"] for item in results.values()), "cases": results}


def _write_markdown_report(report: Mapping[str, object], path: Path, result_sha256: str) -> None:
    """Write a concise narrative report after the machine-readable result exists."""
    direct_official = report["direct_rag"]["official_scifact"]
    audited_official = report["audited_agent"]["official_scifact"]
    official_deltas = report["comparison_to_direct_rag"]["official_scifact"]
    official_bootstrap = report["official_bootstrap_confidence_intervals"]
    lines = [
        "# Phase 08 — Controlled direct-RAG vs audited-agent experiment",
        "",
        "**This is a held-out development evaluation, not an independent test.**",
        "",
        "Both arms consumed the same frozen BM25/verifier trace. The direct-RAG arm used",
        "the first retrieved document with up to three top-scoring rationale sentences and",
        "no audit thresholds; the audited arm applied the frozen Phase 06 policy.",
        "",
        f"Result JSON SHA-256: `{result_sha256}`",
        f"Raw trace SHA-256: `{report['trace_artifact']['sha256']}`",
        "",
        "## Official SciFact-compatible scoring",
        "",
        "| Metric | Direct RAG | Audited agent | Delta | 95% paired-bootstrap CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for metric, label in (
        ("abstract_level_f1", "Abstract-level F1"),
        ("sentence_level_f1", "Sentence-level F1"),
    ):
        group = metric.removesuffix("_f1")
        direct_value = float(direct_official[group]["f1"])
        audited_value = float(audited_official[group]["f1"])
        delta = float(official_deltas[metric])
        interval = official_bootstrap["metrics"][metric]
        lines.append(
            f"| {label} | {direct_value:.4f} | {audited_value:.4f} | {delta:+.4f} | "
            f"[{float(interval['lower']):+.4f}, {float(interval['upper']):+.4f}] |"
        )
    adversarial = report["adversarial_evaluator_suite"]
    lines.extend(
        [
            "",
            "## Adversarial scoring checks",
            "",
            "The official evaluator regression suite rejects a wrong document, wrong stance,",
            "and incomplete multi-sentence rationale.",
            "",
            f"All adversarial scoring checks passed: `{adversarial['all_passed']}`.",
            "",
            "Quality claims must be made only when the corresponding paired confidence interval",
            "excludes zero. The development result must not be represented as an independent test.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_agent_trace(report: Mapping[str, object], path: Path, result_sha256: str) -> None:
    lines = [
        "# Phase 08 controlled-experiments execution trace",
        "",
        "1. Validated every fixed input artifact against its declared SHA-256.",
        "2. Loaded only corpus text, claims, index, verifier bundle, and frozen calibration policy.",
        "3. Wrote and hashed one raw gold-free runtime trace.",
        "4. Produced direct-RAG and audited-agent official-format predictions from that same trace.",
        "5. Only then loaded development gold annotations for scoring and paired bootstrap analysis.",
        "6. Ran deterministic adversarial evaluator checks for document, stance, and rationale failures.",
        "",
        f"Result JSON SHA-256: `{result_sha256}`",
        f"Raw trace SHA-256: `{report['trace_artifact']['sha256']}`",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_controlled_experiments(config: ControlledExperimentsConfig) -> dict[str, object]:
    """Run the complete Phase 8 controlled experiment and write all artifacts."""
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
        artifacts.calibration_report.path, artifacts.calibration_report.sha256, "calibration_report"
    )
    train_claims_sha256 = (
        _validated_artifact_sha256(artifacts.train_claims.path, artifacts.train_claims.sha256, "train_claims")
        if artifacts.train_claims is not None
        else None
    )

    corpus = load_scifact_corpus(artifacts.corpus.path)
    bm25_index = load_bm25_index(artifacts.bm25_index.path)
    bundle = load_verifier_bundle(artifacts.verifier_model.path)
    if bm25_index.corpus_sha256 != corpus_sha256:
        raise ControlledExperimentsError("BM25 index does not match the declared corpus SHA-256.")
    if bundle.corpus_sha256 != corpus_sha256:
        raise ControlledExperimentsError("Verifier bundle does not match the declared corpus SHA-256.")
    if train_claims_sha256 is not None and bundle.training_claims_sha256 != train_claims_sha256:
        raise ControlledExperimentsError("Verifier bundle was not trained on declared train_claims.")

    calibration_report = load_calibration_report(artifacts.calibration_report.path)
    selected_policy: CitationAuditPolicy = load_selected_policy(artifacts.calibration_report.path)
    if train_claims_sha256 is not None:
        calibration_data = calibration_report.get("data")
        if not isinstance(calibration_data, Mapping):
            raise ControlledExperimentsError("Calibration report is missing data provenance.")
        calibration_training = calibration_data.get("main_training_claims")
        if not isinstance(calibration_training, Mapping) or calibration_training.get("sha256") != train_claims_sha256:
            raise ControlledExperimentsError(
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
            config.direct_rag.max_sentences_per_citation,
            selected_policy.max_sentences_per_citation,
        ),
    )
    runtime_elapsed_seconds = time.perf_counter() - started_at

    # Gold fields are not read until every runtime output is frozen to disk.
    trace_payload = {
        "schema_version": RUNTIME_TRACE_SCHEMA,
        "traces": [trace.as_dict() for trace in raw_runtime_traces],
    }
    write_verification_report(trace_payload, config.output.trace_path)
    trace_sha256 = sha256_file(config.output.trace_path)
    direct_rag_traces = apply_direct_rag_to_traces(
        raw_runtime_traces,
        max_sentences_per_citation=config.direct_rag.max_sentences_per_citation,
    )
    audited_traces = apply_citation_audit_to_traces(raw_runtime_traces, selected_policy)
    raw_claim_order = [trace.decision.claim_id for trace in raw_runtime_traces]
    if (
        [trace.decision.claim_id for trace in direct_rag_traces] != raw_claim_order
        or [trace.decision.claim_id for trace in audited_traces] != raw_claim_order
    ):
        raise ControlledExperimentsError("Both experiment arms must preserve the frozen claim order.")
    direct_predictions = official_predictions_from_traces(direct_rag_traces)
    audited_predictions = official_predictions_from_traces(audited_traces)
    write_official_predictions(direct_predictions, config.output.direct_predictions_path)
    write_official_predictions(audited_predictions, config.output.audited_predictions_path)
    direct_predictions_sha256 = sha256_file(config.output.direct_predictions_path)
    audited_predictions_sha256 = sha256_file(config.output.audited_predictions_path)

    gold_annotations = load_gold_claim_annotations(artifacts.claims_dev.path, artifacts.corpus.path)
    direct_audit_summary = evaluate_verification_traces(direct_rag_traces, gold_annotations).summary_dict()
    audited_summary = evaluate_verification_traces(audited_traces, gold_annotations).summary_dict()
    direct_official_summary = evaluate_official_scifact_traces(
        direct_rag_traces, gold_annotations
    ).summary_dict()
    audited_official_summary = evaluate_official_scifact_traces(
        audited_traces, gold_annotations
    ).summary_dict()

    audit_bootstrap: dict[str, object] | None = None
    official_bootstrap: dict[str, object] | None = None
    if config.bootstrap.enabled:
        audit_bootstrap = paired_bootstrap_confidence_intervals(
            build_claim_outcomes(audited_traces, gold_annotations),
            build_claim_outcomes(direct_rag_traces, gold_annotations),
            resamples=config.bootstrap.resamples,
            seed=config.bootstrap.seed,
            confidence_level=config.bootstrap.confidence_level,
        )
        audit_bootstrap["metric"] = "audited_agent_minus_direct_rag_delta"
        official_bootstrap = paired_bootstrap_official_confidence_intervals(
            audited_traces,
            direct_rag_traces,
            gold_annotations,
            resamples=config.bootstrap.resamples,
            seed=config.bootstrap.seed,
            confidence_level=config.bootstrap.confidence_level,
        )

    report: dict[str, object] = {
        "algorithm": "frozen_bm25_lexical_verifier_direct_rag_vs_citation_audit",
        "adversarial_evaluator_suite": run_adversarial_evaluator_suite(),
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
        "audit_bootstrap_confidence_intervals": audit_bootstrap,
        "audited_agent": {
            "audit_metrics": audited_summary,
            "official_scifact": audited_official_summary,
            "policy": selected_policy.as_dict(),
            "predictions_artifact": {
                "path": _relative_posix_path(config.output.audited_predictions_path, config.project_root),
                "sha256": audited_predictions_sha256,
            },
        },
        "comparison_to_direct_rag": {
            "audit_metrics": _audit_metric_deltas(audited_summary, direct_audit_summary),
            "official_scifact": _official_deltas(audited_official_summary, direct_official_summary),
        },
        "config_path": _relative_posix_path(config.config_path, config.project_root),
        "direct_rag": {
            "audit_metrics": direct_audit_summary,
            "official_scifact": direct_official_summary,
            "policy": {
                "max_sentences_per_citation": config.direct_rag.max_sentences_per_citation,
                "retrieval_candidate": "bm25_rank_1",
                "uses_abstention_thresholds": False,
            },
            "predictions_artifact": {
                "path": _relative_posix_path(config.output.direct_predictions_path, config.project_root),
                "sha256": direct_predictions_sha256,
            },
        },
        "evaluation_label": EVALUATION_LABEL,
        "is_independent_test": False,
        "official_bootstrap_confidence_intervals": official_bootstrap,
        "output": {
            "agent_trace_path": _relative_posix_path(config.output.agent_trace_path, config.project_root),
            "audited_predictions_path": _relative_posix_path(
                config.output.audited_predictions_path, config.project_root
            ),
            "direct_predictions_path": _relative_posix_path(
                config.output.direct_predictions_path, config.project_root
            ),
            "report_path": _relative_posix_path(config.output.report_path, config.project_root),
            "result_path": _relative_posix_path(config.output.result_path, config.project_root),
            "trace_path": _relative_posix_path(config.output.trace_path, config.project_root),
        },
        "runtime_settings": {
            "direct_rag_max_sentences_per_citation": config.direct_rag.max_sentences_per_citation,
            "retrieval_k": config.runtime.retrieval_k,
            "selected_policy": selected_policy.as_dict(),
        },
        "runtime_timing": {
            "claim_count": len(runtime_claims),
            "per_claim_milliseconds": 1_000 * runtime_elapsed_seconds / len(runtime_claims),
            "total_seconds": runtime_elapsed_seconds,
        },
        "schema_version": RESULT_SCHEMA_VERSION,
        "trace_artifact": {
            "path": _relative_posix_path(config.output.trace_path, config.project_root),
            "schema_version": RUNTIME_TRACE_SCHEMA,
            "sha256": trace_sha256,
            "trace_count": len(raw_runtime_traces),
        },
    }
    write_verification_report(report, config.output.result_path)
    result_sha256 = sha256_file(config.output.result_path)
    _write_markdown_report(report, config.output.report_path, result_sha256)
    _write_agent_trace(report, config.output.agent_trace_path, result_sha256)
    return report


def run_controlled_experiments_command(config_path: Path) -> dict[str, object]:
    """Thin CLI entry point for the Phase 8 module."""
    config = load_controlled_experiments_config(config_path)
    report = run_controlled_experiments(config)
    return {
        "audit_bootstrap_confidence_intervals": report["audit_bootstrap_confidence_intervals"],
        "comparison_to_direct_rag": report["comparison_to_direct_rag"],
        "evaluation_label": report["evaluation_label"],
        "official_bootstrap_confidence_intervals": report["official_bootstrap_confidence_intervals"],
        "result_path": str(config.output.result_path),
        "trace_path": str(config.output.trace_path),
    }
