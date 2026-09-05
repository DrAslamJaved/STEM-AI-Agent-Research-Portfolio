"""Leakage-safe cross-validation selection for citation-audit policies.

SciFact's supplied folds partition the union of its ordinary training and
development claims.  To preserve ``claims_dev.jsonl`` as a final evaluation
split, this module uses only the supplied fold *assignment* of IDs that already
belong to ``claims_train.jsonl``.  The supplied fold training files are not
used: they contain ordinary-development claims and would contaminate policy
selection.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_agent.audit.policy import CitationAuditPolicy, apply_citation_audit_to_traces
from evidence_agent.data.acquisition import sha256_file
from evidence_agent.data.schemas import Claim
from evidence_agent.evaluation.verification import evaluate_verification_traces
from evidence_agent.retrieval.bm25 import BM25Index
from evidence_agent.retrieval.scifact import load_runtime_claims
from evidence_agent.verification.agent import VerificationTrace, run_verification_agent
from evidence_agent.verification.models import (
    DEFAULT_MAX_FEATURES,
    DEFAULT_RANDOM_SEED,
    fit_verifier_bundle,
    write_verifier_bundle,
)
from evidence_agent.verification.scifact import load_gold_claim_annotations, load_verification_training_data


CALIBRATION_REPORT_SCHEMA = "evidence_agent_citation_audit_calibration_v1"
RUNTIME_TRACE_SCHEMA = "evidence_agent_citation_audit_runtime_trace_v1"
DEFAULT_ASSERTION_THRESHOLDS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_SENTENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
DEFAULT_MAX_SENTENCES_PER_CITATION = (1, 2)
DEFAULT_MINIMUM_COVERAGE = 0.20


class CitationAuditCalibrationError(ValueError):
    """Raised when a calibration split, policy grid, or report is invalid."""


@dataclass(frozen=True, slots=True)
class TrainOnlyFoldPartition:
    """One validation partition derived from a supplied SciFact fold assignment."""

    fold_number: int
    fold_development_path: Path
    validation_claim_ids: tuple[int, ...]
    training_claim_ids: tuple[int, ...]
    ordinary_development_claim_count_excluded: int

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_number": self.fold_number,
            "fold_development_path": str(self.fold_development_path),
            "fold_development_sha256": sha256_file(self.fold_development_path),
            "ordinary_development_claim_count_excluded": self.ordinary_development_claim_count_excluded,
            "training_claim_count": len(self.training_claim_ids),
            "validation_claim_count": len(self.validation_claim_ids),
        }


@dataclass(frozen=True, slots=True)
class FoldRuntimeArtifact:
    """A fold model and gold-free raw trace frozen before fold labels load."""

    partition: TrainOnlyFoldPartition
    model_path: Path
    model_summary: Mapping[str, object]
    trace_path: Path
    traces: tuple[VerificationTrace, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            **self.partition.as_dict(),
            "model": {"path": str(self.model_path), **self.model_summary},
            "trace_artifact": {
                "path": str(self.trace_path),
                "schema_version": RUNTIME_TRACE_SCHEMA,
                "sha256": sha256_file(self.trace_path),
                "trace_count": len(self.traces),
            },
        }


@dataclass(frozen=True, slots=True)
class PolicyScore:
    """Pooled out-of-fold audit metrics for a single pre-specified policy."""

    policy: CitationAuditPolicy
    summary: Mapping[str, object]

    @property
    def coverage(self) -> float:
        return float(self.summary["coverage"])

    @property
    def unsupported_assertion_rate(self) -> float:
        return float(self.summary["unsupported_assertion_rate"])

    @property
    def faithfulness(self) -> float:
        return float(self.summary["faithfulness"])

    @property
    def citation_f1(self) -> float:
        citation = self.summary["citation_correctness"]
        if not isinstance(citation, Mapping):  # pragma: no cover - internal invariant
            raise CitationAuditCalibrationError("citation_correctness must be an object.")
        return float(citation["f1"])

    @property
    def evidence_f1(self) -> float:
        evidence = self.summary["evidence_sentence"]
        if not isinstance(evidence, Mapping):  # pragma: no cover - internal invariant
            raise CitationAuditCalibrationError("evidence_sentence must be an object.")
        return float(evidence["f1"])

    @property
    def claim_macro_f1(self) -> float:
        classification = self.summary["claim_classification"]
        if not isinstance(classification, Mapping):  # pragma: no cover - internal invariant
            raise CitationAuditCalibrationError("claim_classification must be an object.")
        return float(classification["macro_f1"])

    def compact_metrics_dict(self) -> dict[str, float | int]:
        return {
            "assertive_decision_count": int(self.summary["assertive_decision_count"]),
            "citation_correctness_f1": self.citation_f1,
            "claim_macro_f1": self.claim_macro_f1,
            "coverage": self.coverage,
            "evidence_sentence_f1": self.evidence_f1,
            "faithfulness": self.faithfulness,
            "unsupported_assertion_rate": self.unsupported_assertion_rate,
        }

    def as_dict(self) -> dict[str, object]:
        return {"metrics": self.compact_metrics_dict(), "policy": self.policy.as_dict()}


@dataclass(frozen=True, slots=True)
class CitationAuditCalibration:
    """Complete selection result, ready for a final development-only evaluation."""

    partitions: tuple[TrainOnlyFoldPartition, ...]
    fold_artifacts: tuple[FoldRuntimeArtifact, ...]
    policy_scores: tuple[PolicyScore, ...]
    selected_policy: CitationAuditPolicy
    minimum_coverage: float
    development_claims_excluded: bool

    def selected_score(self) -> PolicyScore:
        return next(score for score in self.policy_scores if score.policy == self.selected_policy)

    def as_dict(
        self,
        *,
        corpus_sha256: str,
        main_training_claims_path: Path,
        ordinary_development_claims_path: Path,
        cross_validation_dir: Path,
        index: BM25Index,
        policy_grid: Sequence[CitationAuditPolicy],
        retrieval_k: int,
        random_seed: int,
        max_features: int,
    ) -> dict[str, object]:
        return {
            "cross_validation": {
                "development_claims_excluded_from_selection": self.development_claims_excluded,
                "fold_assignment_rule": "supplied_fold_dev_ids_intersected_with_main_training_ids",
                "fold_count": len(self.partitions),
                "source_directory": str(cross_validation_dir),
            },
            "data": {
                "corpus_sha256": corpus_sha256,
                "main_training_claims": {
                    "path": str(main_training_claims_path),
                    "sha256": sha256_file(main_training_claims_path),
                },
                "ordinary_development_claims": {
                    "path": str(ordinary_development_claims_path),
                    "sha256": sha256_file(ordinary_development_claims_path),
                },
            },
            "folds": [artifact.as_dict() for artifact in self.fold_artifacts],
            "index": {
                "document_count": index.document_count,
                "parameters": {"b": index.b, "k1": index.k1},
                "vocabulary_size": index.vocabulary_size,
            },
            "policy_grid": [policy.as_dict() for policy in policy_grid],
            "policy_scores": [score.as_dict() for score in self.policy_scores],
            "runtime_settings": {
                "max_features": max_features,
                "raw_trace_assertion_threshold": 0.0,
                "raw_trace_sentence_threshold": 0.0,
                "retrieval_k": retrieval_k,
                "random_seed": random_seed,
            },
            "schema_version": CALIBRATION_REPORT_SCHEMA,
            "selection_objective": {
                "constraint": {"minimum_coverage": self.minimum_coverage},
                "primary": "minimise_pooled_unsupported_assertion_rate",
                "tie_breakers": [
                    "maximise_coverage",
                    "maximise_faithfulness",
                    "maximise_strict_citation_f1",
                    "prefer_higher_assertion_threshold",
                    "prefer_higher_sentence_threshold",
                    "prefer_fewer_sentences_per_citation",
                ],
            },
            "selected_policy": {
                "metrics": self.selected_score().compact_metrics_dict(),
                "policy": self.selected_policy.as_dict(),
            },
        }


def _normalise_probability_grid(values: Sequence[float], name: str) -> tuple[float, ...]:
    normalised = tuple(sorted({_validate_probability(value, name) for value in values}))
    if not normalised:
        raise CitationAuditCalibrationError(f"{name} must contain at least one value.")
    return normalised


def _normalise_positive_int_grid(values: Sequence[int], name: str) -> tuple[int, ...]:
    normalised = tuple(sorted({_validate_positive_int(value, name) for value in values}))
    if not normalised:
        raise CitationAuditCalibrationError(f"{name} must contain at least one value.")
    return normalised


def _validate_probability(value: float, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise CitationAuditCalibrationError(f"{name} must lie in [0, 1].")
    return float(value)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CitationAuditCalibrationError(f"{name} must be a positive integer.")
    return value


def build_policy_grid(
    *,
    assertion_thresholds: Sequence[float] = DEFAULT_ASSERTION_THRESHOLDS,
    sentence_thresholds: Sequence[float] = DEFAULT_SENTENCE_THRESHOLDS,
    max_sentences_per_citation: Sequence[int] = DEFAULT_MAX_SENTENCES_PER_CITATION,
) -> tuple[CitationAuditPolicy, ...]:
    """Build a deterministic, duplicate-free grid of candidate audit policies."""
    return tuple(
        CitationAuditPolicy(assertion_threshold, sentence_threshold, maximum)
        for assertion_threshold in _normalise_probability_grid(
            assertion_thresholds, "assertion_thresholds"
        )
        for sentence_threshold in _normalise_probability_grid(
            sentence_thresholds, "sentence_thresholds"
        )
        for maximum in _normalise_positive_int_grid(
            max_sentences_per_citation, "max_sentences_per_citation"
        )
    )


def derive_train_only_fold_partitions(
    main_training_claims_path: Path,
    ordinary_development_claims_path: Path,
    cross_validation_dir: Path,
) -> tuple[TrainOnlyFoldPartition, ...]:
    """Filter supplied fold assignments so no ordinary-development claim is used.

    The supplied ``claims_train_i.jsonl`` files are intentionally ignored
    because they include the ordinary development split.  Each retained
    validation partition is the supplied ``claims_dev_i`` IDs intersected with
    the ordinary training IDs; its training complement also comes only from
    the ordinary training file.
    """
    main_training_ids = {claim.claim_id for claim in load_runtime_claims(main_training_claims_path)}
    ordinary_development_ids = {
        claim.claim_id for claim in load_runtime_claims(ordinary_development_claims_path)
    }
    if main_training_ids.intersection(ordinary_development_ids):
        raise CitationAuditCalibrationError(
            "Main training and ordinary development claim IDs must be disjoint."
        )
    if not main_training_ids:
        raise CitationAuditCalibrationError("Main training claims are empty.")

    partitions: list[TrainOnlyFoldPartition] = []
    assigned_validation_ids: set[int] = set()
    for fold_number in range(1, 6):
        fold_development_path = (
            Path(cross_validation_dir)
            / f"fold_{fold_number}"
            / f"claims_dev_{fold_number}.jsonl"
        )
        if not fold_development_path.is_file():
            raise CitationAuditCalibrationError(
                f"Missing supplied cross-validation file: {fold_development_path}"
            )
        supplied_ids = {claim.claim_id for claim in load_runtime_claims(fold_development_path)}
        validation_ids = supplied_ids.intersection(main_training_ids)
        ordinary_development_ids_in_fold = supplied_ids.intersection(ordinary_development_ids)
        unknown_ids = supplied_ids - main_training_ids - ordinary_development_ids
        if unknown_ids:
            raise CitationAuditCalibrationError(
                f"Fold {fold_number} contains IDs outside main train/dev: {sorted(unknown_ids)}"
            )
        if not validation_ids:
            raise CitationAuditCalibrationError(
                f"Fold {fold_number} contains no main-training validation IDs."
            )
        overlap = assigned_validation_ids.intersection(validation_ids)
        if overlap:
            raise CitationAuditCalibrationError(
                f"Supplied folds assign main-training IDs more than once: {sorted(overlap)}"
            )
        assigned_validation_ids.update(validation_ids)
        partitions.append(
            TrainOnlyFoldPartition(
                fold_number=fold_number,
                fold_development_path=fold_development_path,
                validation_claim_ids=tuple(sorted(validation_ids)),
                training_claim_ids=tuple(sorted(main_training_ids - validation_ids)),
                ordinary_development_claim_count_excluded=len(ordinary_development_ids_in_fold),
            )
        )
    missing = main_training_ids - assigned_validation_ids
    if missing:
        raise CitationAuditCalibrationError(
            f"Supplied folds omit main-training IDs: {sorted(missing)}"
        )
    return tuple(partitions)


def _subset_sha256(source_sha256: str, claim_ids: Sequence[int]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(source_sha256.encode("ascii"))
    digest.update(b"\n")
    digest.update(",".join(str(claim_id) for claim_id in sorted(claim_ids)).encode("ascii"))
    return digest.hexdigest()


def _write_runtime_trace(traces: Sequence[VerificationTrace], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": RUNTIME_TRACE_SCHEMA,
                "traces": [trace.as_dict() for trace in traces],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _policy_score_sort_key(score: PolicyScore) -> tuple[float, ...]:
    policy = score.policy
    return (
        score.unsupported_assertion_rate,
        -score.coverage,
        -score.faithfulness,
        -score.citation_f1,
        -policy.assertion_threshold,
        -policy.sentence_threshold,
        policy.max_sentences_per_citation,
    )


def select_policy(
    policy_scores: Sequence[PolicyScore],
    *,
    minimum_coverage: float,
) -> CitationAuditPolicy:
    """Select one policy by the pre-registered utility-constrained objective."""
    minimum_coverage = _validate_probability(minimum_coverage, "minimum_coverage")
    eligible = [score for score in policy_scores if score.coverage >= minimum_coverage]
    if not eligible:
        raise CitationAuditCalibrationError(
            "No candidate policy satisfies the minimum coverage constraint."
        )
    return min(eligible, key=_policy_score_sort_key).policy


def calibrate_citation_audit(
    *,
    corpus_path: Path,
    main_training_claims_path: Path,
    ordinary_development_claims_path: Path,
    cross_validation_dir: Path,
    index: BM25Index,
    artifact_dir: Path,
    assertion_thresholds: Sequence[float] = DEFAULT_ASSERTION_THRESHOLDS,
    sentence_thresholds: Sequence[float] = DEFAULT_SENTENCE_THRESHOLDS,
    max_sentences_per_citation: Sequence[int] = DEFAULT_MAX_SENTENCES_PER_CITATION,
    minimum_coverage: float = DEFAULT_MINIMUM_COVERAGE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_features: int = DEFAULT_MAX_FEATURES,
    retrieval_k: int = 10,
) -> tuple[CitationAuditCalibration, tuple[CitationAuditPolicy, ...]]:
    """Fit five train-only fold models, freeze traces, then select an audit policy.

    All five model traces are persisted before any fold gold annotations are
    loaded.  The returned policy is therefore chosen only from evaluator-side
    measurements of already-frozen predictions.
    """
    _validate_positive_int(retrieval_k, "retrieval_k")
    _validate_positive_int(random_seed, "random_seed")
    _validate_positive_int(max_features, "max_features")
    minimum_coverage = _validate_probability(minimum_coverage, "minimum_coverage")
    policy_grid = build_policy_grid(
        assertion_thresholds=assertion_thresholds,
        sentence_thresholds=sentence_thresholds,
        max_sentences_per_citation=max_sentences_per_citation,
    )
    corpus_sha256 = sha256_file(corpus_path)
    if index.corpus_sha256 != corpus_sha256:
        raise CitationAuditCalibrationError("BM25 index does not match the supplied corpus SHA-256.")
    partitions = derive_train_only_fold_partitions(
        main_training_claims_path,
        ordinary_development_claims_path,
        cross_validation_dir,
    )

    from evidence_agent.retrieval.scifact import load_scifact_corpus

    corpus = load_scifact_corpus(corpus_path)
    runtime_claims_by_id = {
        claim.claim_id: claim for claim in load_runtime_claims(main_training_claims_path)
    }
    max_sentences = max(policy.max_sentences_per_citation for policy in policy_grid)
    source_training_sha256 = sha256_file(main_training_claims_path)
    fold_artifacts: list[FoldRuntimeArtifact] = []
    artifact_dir = Path(artifact_dir)
    for partition in partitions:
        training_data = load_verification_training_data(
            main_training_claims_path,
            corpus_path,
            claim_ids=partition.training_claim_ids,
        )
        bundle = fit_verifier_bundle(
            training_data.stance_examples,
            training_data.sentence_examples,
            training_claims_sha256=_subset_sha256(
                source_training_sha256,
                partition.training_claim_ids,
            ),
            corpus_sha256=corpus_sha256,
            random_seed=random_seed,
            max_features=max_features,
        )
        model_path = artifact_dir / f"fold_{partition.fold_number}_verifier.joblib"
        write_verifier_bundle(bundle, model_path)
        validation_claims: tuple[Claim, ...] = tuple(
            runtime_claims_by_id[claim_id] for claim_id in partition.validation_claim_ids
        )
        raw_traces = run_verification_agent(
            bundle,
            index,
            corpus,
            validation_claims,
            retrieval_k=retrieval_k,
            assertion_threshold=0.0,
            sentence_threshold=0.0,
            max_sentences_per_citation=max_sentences,
        )
        trace_path = artifact_dir / f"fold_{partition.fold_number}_runtime_trace.json"
        _write_runtime_trace(raw_traces, trace_path)
        fold_artifacts.append(
            FoldRuntimeArtifact(
                partition=partition,
                model_path=model_path,
                model_summary=bundle.summary_dict(),
                trace_path=trace_path,
                traces=raw_traces,
            )
        )

    # Gold annotations are deliberately loaded only after every fold's raw
    # model output has been written to a local trace artifact.
    gold_annotations = load_gold_claim_annotations(
        main_training_claims_path,
        corpus_path,
        claim_ids=tuple(
            claim_id
            for partition in partitions
            for claim_id in partition.validation_claim_ids
        ),
    )
    all_raw_traces = tuple(
        trace for artifact in fold_artifacts for trace in artifact.traces
    )
    scores: list[PolicyScore] = []
    for policy in policy_grid:
        audited_traces = apply_citation_audit_to_traces(all_raw_traces, policy)
        scores.append(
            PolicyScore(
                policy=policy,
                summary=evaluate_verification_traces(
                    audited_traces,
                    gold_annotations,
                ).summary_dict(),
            )
        )
    selected_policy = select_policy(scores, minimum_coverage=minimum_coverage)
    return (
        CitationAuditCalibration(
            partitions=partitions,
            fold_artifacts=tuple(fold_artifacts),
            policy_scores=tuple(scores),
            selected_policy=selected_policy,
            minimum_coverage=minimum_coverage,
            development_claims_excluded=True,
        ),
        policy_grid,
    )


def write_calibration_report(payload: Mapping[str, object], path: Path) -> None:
    """Write the compact, version-controlled policy-selection report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_calibration_report(path: Path) -> Mapping[str, object]:
    """Load and validate the compact policy-selection report."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CitationAuditCalibrationError(f"Unable to read calibration report {path}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CALIBRATION_REPORT_SCHEMA:
        raise CitationAuditCalibrationError("Calibration report has an unsupported schema version.")
    return payload


def load_selected_policy(path: Path) -> CitationAuditPolicy:
    """Load and validate the frozen selected policy from a calibration report."""
    payload = load_calibration_report(path)
    selected = payload.get("selected_policy")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("policy"), Mapping):
        raise CitationAuditCalibrationError("Calibration report has no selected policy.")
    policy = selected["policy"]
    try:
        return CitationAuditPolicy(
            assertion_threshold=float(policy["assertion_threshold"]),
            sentence_threshold=float(policy["sentence_threshold"]),
            max_sentences_per_citation=int(policy["max_sentences_per_citation"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CitationAuditCalibrationError("Calibration report has an invalid selected policy.") from error
