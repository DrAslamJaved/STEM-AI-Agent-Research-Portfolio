"""Small, explicit command-line interface for the staged project workflow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from evidence_agent import __version__
from evidence_agent.audit.calibration import (
    CALIBRATION_REPORT_SCHEMA,
    DEFAULT_ASSERTION_THRESHOLDS,
    DEFAULT_MAX_SENTENCES_PER_CITATION as DEFAULT_AUDIT_MAX_SENTENCES_PER_CITATION,
    DEFAULT_MINIMUM_COVERAGE,
    DEFAULT_SENTENCE_THRESHOLDS,
    RUNTIME_TRACE_SCHEMA as AUDIT_RUNTIME_TRACE_SCHEMA,
    calibrate_citation_audit,
    load_calibration_report,
    load_selected_policy,
    write_calibration_report,
)
from evidence_agent.audit.policy import (
    PHASE_05_POLICY,
    apply_citation_audit_to_traces,
)
from evidence_agent.data.acquisition import (
    DEFAULT_SCIFACT_URL,
    acquire_scifact,
    sha256_file,
    write_acquisition_manifest,
)
from evidence_agent.data.scifact import (
    validate_scifact_dataset,
    write_validation_report,
)
from evidence_agent.evaluation.retrieval import (
    evaluate_retrieval_predictions,
    retrieve_claims,
    write_retrieval_report,
)
from evidence_agent.evaluation.verification import (
    evaluate_stance_benchmark,
    evaluate_verification_traces,
    write_verification_report,
)
from evidence_agent.retrieval.bm25 import (
    build_bm25_index,
    load_bm25_index,
    write_bm25_index,
)
from evidence_agent.retrieval.hybrid import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RRF_RANK_CONSTANT,
    HybridRetriever,
    as_evaluation_predictions,
    retrieve_hybrid_claims,
)
from evidence_agent.retrieval.scifact import (
    load_gold_evidence_documents,
    load_runtime_claims,
    load_scifact_corpus,
)
from evidence_agent.retrieval.semantic import (
    build_lsa_index,
    load_lsa_index,
    write_lsa_index,
)
from evidence_agent.verification.agent import (
    DEFAULT_ASSERTION_THRESHOLD,
    DEFAULT_MAX_SENTENCES_PER_CITATION,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SENTENCE_THRESHOLD,
    run_verification_agent,
)
from evidence_agent.verification.models import (
    DEFAULT_MAX_FEATURES,
    DEFAULT_RANDOM_SEED,
    fit_verifier_bundle,
    load_verifier_bundle,
    write_verifier_bundle,
)
from evidence_agent.verification.scifact import (
    load_gold_claim_annotations,
    load_stance_benchmark_inputs,
    load_stance_benchmark_labels,
    load_verification_training_data,
)


UNIMPLEMENTED_COMMANDS = {
    "evaluate": "Complete the relevant evaluation pipeline before running this command.",
}

DEFAULT_CORPUS_PATH = Path("data/raw/scifact/data/corpus.jsonl")
DEFAULT_DEV_CLAIMS_PATH = Path("data/raw/scifact/data/claims_dev.jsonl")
DEFAULT_INDEX_PATH = Path("artifacts/scifact_bm25_index.json")
DEFAULT_RETRIEVAL_REPORT_PATH = Path("results/retrieval_baseline_dev.json")
DEFAULT_SEMANTIC_INDEX_PATH = Path("artifacts/scifact_lsa_index.joblib")
DEFAULT_HYBRID_REPORT_PATH = Path("results/hybrid_retrieval_dev.json")
DEFAULT_TRAIN_CLAIMS_PATH = Path("data/raw/scifact/data/claims_train.jsonl")
DEFAULT_VERIFIER_MODEL_PATH = Path("artifacts/scifact_lexical_verifier.joblib")
DEFAULT_VERIFICATION_REPORT_PATH = Path("results/verification_dev.json")
DEFAULT_VERIFICATION_TRACE_PATH = Path("artifacts/verification_dev_trace.json")
DEFAULT_CROSS_VALIDATION_DIR = Path("data/raw/scifact/data/cross_validation")
DEFAULT_CITATION_AUDIT_ARTIFACT_DIR = Path("artifacts/citation_audit_cv")
DEFAULT_CITATION_AUDIT_CALIBRATION_REPORT_PATH = Path(
    "results/citation_audit_cross_validation.json"
)
DEFAULT_CITATION_AUDIT_REPORT_PATH = Path("results/citation_audit_dev.json")
DEFAULT_CITATION_AUDIT_TRACE_PATH = Path("artifacts/citation_audit_dev_trace.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser without performing I/O or model loading."""
    parser = argparse.ArgumentParser(
        prog="evidence-agent",
        description="Scientific Evidence Verification and Citation-Audit Agent",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("contract", help="Print the active project contract.")

    acquire = subparsers.add_parser(
        "acquire-data", help="Acquire the official SciFact data release."
    )
    acquire.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/scifact"),
        help="Directory for the immutable source archive and extracted release.",
    )
    acquire.add_argument(
        "--url",
        default=DEFAULT_SCIFACT_URL,
        help="Official SciFact archive URL.",
    )
    acquire.add_argument(
        "--provenance-path",
        type=Path,
        default=Path("validation/scifact_acquisition.json"),
        help="Path for the acquisition provenance manifest.",
    )

    validate = subparsers.add_parser(
        "validate-data", help="Validate an acquired SciFact release."
    )
    validate.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/scifact"),
        help="SciFact extraction directory or a parent containing it.",
    )
    validate.add_argument(
        "--report-path",
        type=Path,
        default=Path("validation/scifact_validation.json"),
        help="Path for the validation report.",
    )
    validate.add_argument(
        "--skip-cross-validation",
        action="store_true",
        help="Allow an intentionally reduced fixture without five-fold splits.",
    )
    index = subparsers.add_parser(
        "build-index", help="Build the deterministic SciFact BM25 baseline index."
    )
    index.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Validated SciFact corpus JSONL file.",
    )
    index.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Output path for the ignored BM25 index artifact.",
    )
    index.add_argument("--k1", type=float, default=1.2, help="BM25 term-saturation value.")
    index.add_argument("--b", type=float, default=0.75, help="BM25 length-normalisation value.")

    semantic = subparsers.add_parser(
        "build-semantic-index",
        help="Build a fixed corpus-only TF-IDF + LSA semantic index.",
    )
    semantic.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Validated SciFact corpus JSONL file.",
    )
    semantic.add_argument(
        "--semantic-index-path",
        type=Path,
        default=DEFAULT_SEMANTIC_INDEX_PATH,
        help="Ignored local path for the fitted LSA artifact.",
    )
    semantic.add_argument(
        "--n-components",
        type=int,
        default=128,
        help="Fixed number of latent semantic dimensions.",
    )
    semantic.add_argument(
        "--random-seed",
        type=int,
        default=20260904,
        help="Fixed randomized-SVD seed.",
    )
    semantic.add_argument(
        "--min-document-frequency",
        type=int,
        default=2,
        help="Minimum corpus document frequency retained by TF-IDF.",
    )

    retrieval = subparsers.add_parser(
        "evaluate-retrieval", help="Evaluate the frozen BM25 baseline with Recall@k."
    )
    retrieval.add_argument(
        "--claims-path",
        type=Path,
        default=DEFAULT_DEV_CLAIMS_PATH,
        help="SciFact claim JSONL file; development claims are the default.",
    )
    retrieval.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Previously built BM25 index artifact.",
    )
    retrieval.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_RETRIEVAL_REPORT_PATH,
        help="Machine-readable retrieval report to retain in Git.",
    )
    retrieval.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Positive retrieval cutoffs used for Recall@k.",
    )

    hybrid = subparsers.add_parser(
        "evaluate-hybrid-retrieval",
        help="Fuse BM25 and LSA retrieval, rerank candidates, and evaluate Recall@k.",
    )
    hybrid.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Validated SciFact corpus JSONL file used for title-aware reranking.",
    )
    hybrid.add_argument(
        "--claims-path",
        type=Path,
        default=DEFAULT_DEV_CLAIMS_PATH,
        help="SciFact development claim JSONL file.",
    )
    hybrid.add_argument(
        "--bm25-index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Previously built lexical BM25 index artifact.",
    )
    hybrid.add_argument(
        "--semantic-index-path",
        type=Path,
        default=DEFAULT_SEMANTIC_INDEX_PATH,
        help="Previously built corpus-only LSA index artifact.",
    )
    hybrid.add_argument(
        "--baseline-report-path",
        type=Path,
        default=DEFAULT_RETRIEVAL_REPORT_PATH,
        help="Committed BM25 development report used only after hybrid predictions freeze.",
    )
    hybrid.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_HYBRID_REPORT_PATH,
        help="Machine-readable hybrid retrieval report to retain in Git.",
    )
    hybrid.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Top candidates requested independently from each first-stage retriever.",
    )
    hybrid.add_argument(
        "--rrf-rank-constant",
        type=int,
        default=DEFAULT_RRF_RANK_CONSTANT,
        help="Fixed reciprocal-rank-fusion rank constant.",
    )
    hybrid.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Positive retrieval cutoffs used for Recall@k.",
    )

    train_verifier = subparsers.add_parser(
        "train-verifier",
        help="Fit train-split lexical stance and sentence-evidence models.",
    )
    train_verifier.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Validated SciFact public corpus JSONL file.",
    )
    train_verifier.add_argument(
        "--train-claims-path",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS_PATH,
        help="SciFact training claim JSONL file; development labels are forbidden here.",
    )
    train_verifier.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_VERIFIER_MODEL_PATH,
        help="Ignored local path for the trusted verifier bundle artifact.",
    )
    train_verifier.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Fixed logistic-regression random seed.",
    )
    train_verifier.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="Maximum TF-IDF feature count for each lexical model.",
    )

    verifier = subparsers.add_parser(
        "evaluate-verifier",
        help="Run BM25 -> evidence selection -> stance verification on development claims.",
    )
    verifier.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Validated SciFact public corpus JSONL file.",
    )
    verifier.add_argument(
        "--claims-path",
        type=Path,
        default=DEFAULT_DEV_CLAIMS_PATH,
        help="SciFact development claims used only for frozen runtime decisions and evaluation.",
    )
    verifier.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Previously built lexical BM25 index artifact.",
    )
    verifier.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_VERIFIER_MODEL_PATH,
        help="Locally trained verifier bundle from train-verifier.",
    )
    verifier.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_VERIFICATION_REPORT_PATH,
        help="Compact machine-readable verifier result report to retain in Git.",
    )
    verifier.add_argument(
        "--trace-path",
        type=Path,
        default=DEFAULT_VERIFICATION_TRACE_PATH,
        help="Ignored local path for the complete candidate and sentence diagnostic trace.",
    )
    verifier.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help="Fixed number of BM25 documents supplied to the runtime verifier.",
    )
    verifier.add_argument(
        "--assertion-threshold",
        type=float,
        default=DEFAULT_ASSERTION_THRESHOLD,
        help="Minimum combined stance/evidence confidence for an assertive verdict.",
    )
    verifier.add_argument(
        "--sentence-threshold",
        type=float,
        default=DEFAULT_SENTENCE_THRESHOLD,
        help="Minimum probability for a sentence to be cited as evidence.",
    )
    verifier.add_argument(
        "--max-sentences-per-citation",
        type=int,
        default=DEFAULT_MAX_SENTENCES_PER_CITATION,
        help="Maximum selected sentences retained in the single emitted citation.",
    )

    calibrate_audit = subparsers.add_parser(
        "calibrate-citation-audit",
        help="Select a fixed citation-audit policy using train-only SciFact fold assignments.",
    )
    calibrate_audit.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Validated SciFact public corpus JSONL file.",
    )
    calibrate_audit.add_argument(
        "--train-claims-path",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS_PATH,
        help="Ordinary SciFact training claims; the only labels eligible for selection.",
    )
    calibrate_audit.add_argument(
        "--development-claims-path",
        type=Path,
        default=DEFAULT_DEV_CLAIMS_PATH,
        help="Ordinary development claims retained completely outside policy selection.",
    )
    calibrate_audit.add_argument(
        "--cross-validation-dir",
        type=Path,
        default=DEFAULT_CROSS_VALIDATION_DIR,
        help="SciFact supplied cross-validation directory used only for fold assignments.",
    )
    calibrate_audit.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Previously built lexical BM25 index artifact.",
    )
    calibrate_audit.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_CITATION_AUDIT_ARTIFACT_DIR,
        help="Ignored local directory for fold models and complete diagnostic traces.",
    )
    calibrate_audit.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_CITATION_AUDIT_CALIBRATION_REPORT_PATH,
        help="Compact cross-validation policy-selection report to retain in Git.",
    )
    calibrate_audit.add_argument(
        "--assertion-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_ASSERTION_THRESHOLDS),
        help="Candidate combined-confidence thresholds for assertive decisions.",
    )
    calibrate_audit.add_argument(
        "--sentence-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_SENTENCE_THRESHOLDS),
        help="Candidate sentence-evidence thresholds for accepted citations.",
    )
    calibrate_audit.add_argument(
        "--max-sentences-per-citation",
        type=int,
        nargs="+",
        default=list(DEFAULT_AUDIT_MAX_SENTENCES_PER_CITATION),
        help="Candidate limits for sentences retained in one citation.",
    )
    calibrate_audit.add_argument(
        "--minimum-coverage",
        type=float,
        default=DEFAULT_MINIMUM_COVERAGE,
        help="Minimum pooled assertion coverage required during policy selection.",
    )
    calibrate_audit.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help="Fixed BM25 documents supplied to each fold runtime.",
    )
    calibrate_audit.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Fixed logistic-regression random seed for every fold model.",
    )
    calibrate_audit.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="Maximum TF-IDF feature count for each fold model.",
    )

    evaluate_audit = subparsers.add_parser(
        "evaluate-citation-audit",
        help="Evaluate the frozen selected citation-audit policy on held-out development claims.",
    )
    evaluate_audit.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Validated SciFact public corpus JSONL file.",
    )
    evaluate_audit.add_argument(
        "--claims-path",
        type=Path,
        default=DEFAULT_DEV_CLAIMS_PATH,
        help="Held-out claim file; ordinary development claims are the default.",
    )
    evaluate_audit.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Previously built lexical BM25 index artifact.",
    )
    evaluate_audit.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_VERIFIER_MODEL_PATH,
        help="Verifier bundle trained on the complete ordinary training split.",
    )
    evaluate_audit.add_argument(
        "--train-claims-path",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS_PATH,
        help="Ordinary training claims that must match the final verifier bundle.",
    )
    evaluate_audit.add_argument(
        "--calibration-report-path",
        type=Path,
        default=DEFAULT_CITATION_AUDIT_CALIBRATION_REPORT_PATH,
        help="Committed cross-validation report containing the selected policy.",
    )
    evaluate_audit.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_CITATION_AUDIT_REPORT_PATH,
        help="Compact held-out citation-audit evaluation report to retain in Git.",
    )
    evaluate_audit.add_argument(
        "--trace-path",
        type=Path,
        default=DEFAULT_CITATION_AUDIT_TRACE_PATH,
        help="Ignored local path for the complete diagnostic runtime trace.",
    )
    evaluate_audit.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help="Fixed BM25 documents supplied to the held-out runtime.",
    )
    evaluate = subparsers.add_parser("evaluate", help="Run a configured evaluation.")
    evaluate.add_argument("--config", type=str, help="Path to a YAML experiment config.")
    return parser


def _citation_audit_deltas(
    selected_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object],
) -> dict[str, float]:
    """Return selected-minus-baseline changes; negative unsupported rate is better."""

    def nested_metric(summary: Mapping[str, object], group: str, metric: str) -> float:
        values = summary[group]
        if not isinstance(values, Mapping):  # pragma: no cover - internal invariant
            raise ValueError(f"{group} must be a metric object.")
        return float(values[metric])

    return {
        "citation_correctness_f1": nested_metric(
            selected_summary, "citation_correctness", "f1"
        )
        - nested_metric(baseline_summary, "citation_correctness", "f1"),
        "claim_macro_f1": nested_metric(
            selected_summary, "claim_classification", "macro_f1"
        )
        - nested_metric(baseline_summary, "claim_classification", "macro_f1"),
        "coverage": float(selected_summary["coverage"]) - float(baseline_summary["coverage"]),
        "evidence_sentence_f1": nested_metric(
            selected_summary, "evidence_sentence", "f1"
        )
        - nested_metric(baseline_summary, "evidence_sentence", "f1"),
        "faithfulness": float(selected_summary["faithfulness"])
        - float(baseline_summary["faithfulness"]),
        "unsupported_assertion_rate": float(selected_summary["unsupported_assertion_rate"])
        - float(baseline_summary["unsupported_assertion_rate"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "contract":
        print(
            json.dumps(
                {
                    "project": "scientific_evidence_agent",
                    "version": __version__,
                    "runtime_gold_fields_forbidden": ["evidence", "cited_doc_ids"],
                    "current_phase": "cross_validated_citation_audit_policy_selection",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "acquire-data":
        manifest = acquire_scifact(output_dir=args.output_dir, url=args.url)
        write_acquisition_manifest(manifest, args.provenance_path)
        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "validate-data":
        summary = validate_scifact_dataset(
            data_dir=args.data_dir,
            require_cross_validation=not args.skip_cross_validation,
        )
        write_validation_report(summary, args.report_path)
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "build-index":
        corpus = load_scifact_corpus(args.corpus_path)
        index = build_bm25_index(
            {doc_id: document.searchable_text for doc_id, document in corpus.items()},
            k1=args.k1,
            b=args.b,
            corpus_sha256=sha256_file(args.corpus_path),
        )
        write_bm25_index(index, args.index_path)
        print(
            json.dumps(
                {
                    "algorithm": "BM25",
                    "average_document_length": index.average_document_length,
                    "corpus_path": str(args.corpus_path),
                    "corpus_sha256": index.corpus_sha256,
                    "document_count": index.document_count,
                    "index_path": str(args.index_path),
                    "parameters": {"b": index.b, "k1": index.k1},
                    "vocabulary_size": index.vocabulary_size,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "build-semantic-index":
        corpus = load_scifact_corpus(args.corpus_path)
        index = build_lsa_index(
            {doc_id: document.searchable_text for doc_id, document in corpus.items()},
            corpus_sha256=sha256_file(args.corpus_path),
            n_components=args.n_components,
            random_seed=args.random_seed,
            min_document_frequency=args.min_document_frequency,
        )
        write_lsa_index(index, args.semantic_index_path)
        print(
            json.dumps(
                {"index_path": str(args.semantic_index_path), **index.summary_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "evaluate-retrieval":
        index = load_bm25_index(args.index_path)
        runtime_claims = load_runtime_claims(args.claims_path)
        predictions = retrieve_claims(
            index,
            runtime_claims,
            top_k=max(args.cutoffs),
        )
        # Gold evidence is loaded only after runtime retrieval predictions freeze.
        gold_evidence_documents = load_gold_evidence_documents(args.claims_path)
        evaluation = evaluate_retrieval_predictions(
            predictions,
            gold_evidence_documents,
            cutoffs=args.cutoffs,
        )
        report = {
            "algorithm": "BM25",
            "claims": {
                "path": str(args.claims_path),
                "sha256": sha256_file(args.claims_path),
            },
            "index": {
                "corpus_sha256": index.corpus_sha256,
                "document_count": index.document_count,
                "parameters": {"b": index.b, "k1": index.k1},
                "path": str(args.index_path),
                "vocabulary_size": index.vocabulary_size,
            },
            "schema_version": "evidence_agent_retrieval_report_v1",
            "summary": evaluation.summary_dict(),
            "predictions": [prediction.as_dict() for prediction in evaluation.predictions],
        }
        write_retrieval_report(report, args.report_path)
        print(
            json.dumps(
                {"report_path": str(args.report_path), **evaluation.summary_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "evaluate-hybrid-retrieval":
        corpus_sha256 = sha256_file(args.corpus_path)
        corpus = load_scifact_corpus(args.corpus_path)
        bm25_index = load_bm25_index(args.bm25_index_path)
        semantic_index = load_lsa_index(args.semantic_index_path)
        if bm25_index.corpus_sha256 != corpus_sha256:
            raise ValueError("BM25 index does not match the supplied corpus SHA-256.")
        if semantic_index.corpus_sha256 != corpus_sha256:
            raise ValueError("Semantic index does not match the supplied corpus SHA-256.")
        retriever = HybridRetriever(
            bm25_index=bm25_index,
            semantic_index=semantic_index,
            corpus=corpus,
            candidate_k=args.candidate_k,
            rrf_rank_constant=args.rrf_rank_constant,
        )
        runtime_claims = load_runtime_claims(args.claims_path)
        hybrid_predictions = retrieve_hybrid_claims(
            retriever,
            runtime_claims,
            top_k=max(args.cutoffs),
        )
        # Gold evidence and baseline metrics load only after rankings are frozen.
        gold_evidence_documents = load_gold_evidence_documents(args.claims_path)
        evaluation = evaluate_retrieval_predictions(
            as_evaluation_predictions(hybrid_predictions),
            gold_evidence_documents,
            cutoffs=args.cutoffs,
        )
        try:
            baseline_report = json.loads(args.baseline_report_path.read_text(encoding="utf-8"))
            baseline_summary = baseline_report["summary"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"Unable to read the committed BM25 baseline report: {error}"
            ) from error

        hybrid_summary = evaluation.summary_dict()
        metric_deltas = {
            metric: {
                cutoff: hybrid_summary[metric][cutoff] - baseline_summary[metric][cutoff]
                for cutoff in hybrid_summary[metric]
            }
            for metric in ("claim_recall_at_k", "evidence_document_recall_at_k")
        }
        report = {
            "algorithm": "bm25_lsa_rrf_transparent_reranker",
            "baseline": {
                "report_path": str(args.baseline_report_path),
                "report_sha256": sha256_file(args.baseline_report_path),
                "summary": baseline_summary,
            },
            "claims": {
                "path": str(args.claims_path),
                "sha256": sha256_file(args.claims_path),
            },
            "comparison_to_bm25": {
                "claim_recall_at_k_delta": metric_deltas["claim_recall_at_k"],
                "evidence_document_recall_at_k_delta": metric_deltas[
                    "evidence_document_recall_at_k"
                ],
                "mean_reciprocal_rank_delta": hybrid_summary["mean_reciprocal_rank"]
                - baseline_summary["mean_reciprocal_rank"],
            },
            "corpus_sha256": corpus_sha256,
            "indexes": {
                "bm25": {
                    "document_count": bm25_index.document_count,
                    "path": str(args.bm25_index_path),
                    "parameters": {"b": bm25_index.b, "k1": bm25_index.k1},
                    "vocabulary_size": bm25_index.vocabulary_size,
                },
                "semantic": {
                    "path": str(args.semantic_index_path),
                    **semantic_index.summary_dict(),
                },
            },
            "predictions": [prediction.as_dict() for prediction in hybrid_predictions],
            "retriever_settings": retriever.settings_dict(),
            "schema_version": "evidence_agent_hybrid_retrieval_report_v1",
            "summary": hybrid_summary,
        }
        write_retrieval_report(report, args.report_path)
        print(
            json.dumps(
                {
                    "comparison_to_bm25": report["comparison_to_bm25"],
                    "report_path": str(args.report_path),
                    "summary": hybrid_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "train-verifier":
        training_data = load_verification_training_data(
            args.train_claims_path,
            args.corpus_path,
        )
        bundle = fit_verifier_bundle(
            training_data.stance_examples,
            training_data.sentence_examples,
            training_claims_sha256=sha256_file(args.train_claims_path),
            corpus_sha256=sha256_file(args.corpus_path),
            random_seed=args.random_seed,
            max_features=args.max_features,
        )
        write_verifier_bundle(bundle, args.model_path)
        print(
            json.dumps(
                {
                    "model": bundle.summary_dict(),
                    "model_path": str(args.model_path),
                    "training_data": training_data.summary_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "evaluate-verifier":
        if args.trace_path == args.report_path:
            raise ValueError("--trace-path and --report-path must be different files.")
        corpus_sha256 = sha256_file(args.corpus_path)
        corpus = load_scifact_corpus(args.corpus_path)
        bm25_index = load_bm25_index(args.index_path)
        bundle = load_verifier_bundle(args.model_path)
        if bm25_index.corpus_sha256 != corpus_sha256:
            raise ValueError("BM25 index does not match the supplied corpus SHA-256.")
        if bundle.corpus_sha256 != corpus_sha256:
            raise ValueError("Verifier bundle does not match the supplied corpus SHA-256.")

        runtime_claims = load_runtime_claims(args.claims_path)
        started_at = time.perf_counter()
        runtime_traces = run_verification_agent(
            bundle,
            bm25_index,
            corpus,
            runtime_claims,
            retrieval_k=args.retrieval_k,
            assertion_threshold=args.assertion_threshold,
            sentence_threshold=args.sentence_threshold,
            max_sentences_per_citation=args.max_sentences_per_citation,
        )
        runtime_elapsed_seconds = time.perf_counter() - started_at

        # Persist the diagnostic trace before any development gold field is
        # loaded. This trace is intentionally ignored by Git because it keeps
        # every candidate and sentence score; the committed report below is a
        # compact audit record with one decision per claim.
        trace_payload = {
            "schema_version": "evidence_agent_verification_trace_v1",
            "traces": [trace.as_dict() for trace in runtime_traces],
        }
        write_verification_report(trace_payload, args.trace_path)
        trace_artifact = {
            "path": str(args.trace_path),
            "schema_version": trace_payload["schema_version"],
            "sha256": sha256_file(args.trace_path),
            "trace_count": len(runtime_traces),
        }

        # Evaluation-only cited-document labels are read only after the full
        # BM25 -> sentence-selection -> claim-decision runtime trace is frozen.
        stance_benchmark_inputs = load_stance_benchmark_inputs(
            args.claims_path,
            args.corpus_path,
        )
        stance_benchmark_predictions = bundle.predict_stances(stance_benchmark_inputs)
        stance_benchmark_labels = load_stance_benchmark_labels(
            args.claims_path,
            args.corpus_path,
        )
        stance_benchmark = evaluate_stance_benchmark(
            stance_benchmark_predictions,
            stance_benchmark_labels,
        )
        gold_annotations = load_gold_claim_annotations(args.claims_path, args.corpus_path)
        agent_evaluation = evaluate_verification_traces(runtime_traces, gold_annotations)
        model_summary = bundle.summary_dict()
        report = {
            "algorithm": "bm25_lexical_stance_and_sentence_verifier",
            "claims": {
                "path": str(args.claims_path),
                "sha256": sha256_file(args.claims_path),
            },
            "corpus_sha256": corpus_sha256,
            "index": {
                "document_count": bm25_index.document_count,
                "parameters": {"b": bm25_index.b, "k1": bm25_index.k1},
                "path": str(args.index_path),
                "vocabulary_size": bm25_index.vocabulary_size,
            },
            "model": {"path": str(args.model_path), **model_summary},
            "runtime_settings": {
                "assertion_threshold": args.assertion_threshold,
                "max_sentences_per_citation": args.max_sentences_per_citation,
                "retrieval_k": args.retrieval_k,
                "sentence_threshold": args.sentence_threshold,
            },
            "runtime_timing": {
                "claim_count": len(runtime_claims),
                "per_claim_milliseconds": 1_000 * runtime_elapsed_seconds / len(runtime_claims),
                "total_seconds": runtime_elapsed_seconds,
            },
            "schema_version": "evidence_agent_verification_report_v1",
            "stance_benchmark": stance_benchmark.summary_dict(),
            "summary": agent_evaluation.summary_dict(),
            "decisions": [trace.decision_dict() for trace in runtime_traces],
            "trace_artifact": trace_artifact,
        }
        write_verification_report(report, args.report_path)
        print(
            json.dumps(
                {
                    "report_path": str(args.report_path),
                    "stance_benchmark": stance_benchmark.summary_dict(),
                    "summary": agent_evaluation.summary_dict(),
                    "trace_path": str(args.trace_path),
                    "runtime_timing": report["runtime_timing"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "calibrate-citation-audit":
        corpus_sha256 = sha256_file(args.corpus_path)
        bm25_index = load_bm25_index(args.index_path)
        calibration, policy_grid = calibrate_citation_audit(
            corpus_path=args.corpus_path,
            main_training_claims_path=args.train_claims_path,
            ordinary_development_claims_path=args.development_claims_path,
            cross_validation_dir=args.cross_validation_dir,
            index=bm25_index,
            artifact_dir=args.artifact_dir,
            assertion_thresholds=args.assertion_thresholds,
            sentence_thresholds=args.sentence_thresholds,
            max_sentences_per_citation=args.max_sentences_per_citation,
            minimum_coverage=args.minimum_coverage,
            random_seed=args.random_seed,
            max_features=args.max_features,
            retrieval_k=args.retrieval_k,
        )
        report = calibration.as_dict(
            corpus_sha256=corpus_sha256,
            main_training_claims_path=args.train_claims_path,
            ordinary_development_claims_path=args.development_claims_path,
            cross_validation_dir=args.cross_validation_dir,
            index=bm25_index,
            policy_grid=policy_grid,
            retrieval_k=args.retrieval_k,
            random_seed=args.random_seed,
            max_features=args.max_features,
        )
        write_calibration_report(report, args.report_path)
        print(
            json.dumps(
                {
                    "fold_count": len(calibration.partitions),
                    "policy_grid_count": len(policy_grid),
                    "report_path": str(args.report_path),
                    "selected_policy": report["selected_policy"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "evaluate-citation-audit":
        if args.trace_path == args.report_path:
            raise ValueError("--trace-path and --report-path must be different files.")
        corpus_sha256 = sha256_file(args.corpus_path)
        corpus = load_scifact_corpus(args.corpus_path)
        bm25_index = load_bm25_index(args.index_path)
        bundle = load_verifier_bundle(args.model_path)
        if bm25_index.corpus_sha256 != corpus_sha256:
            raise ValueError("BM25 index does not match the supplied corpus SHA-256.")
        if bundle.corpus_sha256 != corpus_sha256:
            raise ValueError("Verifier bundle does not match the supplied corpus SHA-256.")
        calibration_report = load_calibration_report(args.calibration_report_path)
        selected_policy = load_selected_policy(args.calibration_report_path)
        calibration_data = calibration_report.get("data")
        if not isinstance(calibration_data, Mapping):
            raise ValueError("Calibration report is missing its data provenance block.")
        calibration_training = calibration_data.get("main_training_claims")
        if not isinstance(calibration_training, Mapping):
            raise ValueError("Calibration report is missing main training provenance.")
        calibration_training_sha256 = calibration_training.get("sha256")
        supplied_training_sha256 = sha256_file(args.train_claims_path)
        if calibration_training_sha256 != supplied_training_sha256:
            raise ValueError(
                "The calibration report and --train-claims-path do not identify the same training split."
            )
        if bundle.training_claims_sha256 != supplied_training_sha256:
            raise ValueError(
                "Verifier bundle was not trained on the supplied ordinary training split."
            )

        runtime_claims = load_runtime_claims(args.claims_path)
        started_at = time.perf_counter()
        raw_runtime_traces = run_verification_agent(
            bundle,
            bm25_index,
            corpus,
            runtime_claims,
            retrieval_k=args.retrieval_k,
            assertion_threshold=0.0,
            sentence_threshold=0.0,
            max_sentences_per_citation=max(
                PHASE_05_POLICY.max_sentences_per_citation,
                selected_policy.max_sentences_per_citation,
            ),
        )
        runtime_elapsed_seconds = time.perf_counter() - started_at

        # The complete candidate trace is committed to an ignored local file
        # before the held-out labels are read.  Both policy decisions below are
        # pure functions of this same frozen trace.
        trace_payload = {
            "schema_version": AUDIT_RUNTIME_TRACE_SCHEMA,
            "traces": [trace.as_dict() for trace in raw_runtime_traces],
        }
        write_verification_report(trace_payload, args.trace_path)
        trace_artifact = {
            "path": str(args.trace_path),
            "schema_version": AUDIT_RUNTIME_TRACE_SCHEMA,
            "sha256": sha256_file(args.trace_path),
            "trace_count": len(raw_runtime_traces),
        }
        audited_traces = apply_citation_audit_to_traces(
            raw_runtime_traces,
            selected_policy,
        )
        phase_05_traces = apply_citation_audit_to_traces(
            raw_runtime_traces,
            PHASE_05_POLICY,
        )

        gold_annotations = load_gold_claim_annotations(args.claims_path, args.corpus_path)
        selected_evaluation = evaluate_verification_traces(
            audited_traces,
            gold_annotations,
        )
        phase_05_evaluation = evaluate_verification_traces(
            phase_05_traces,
            gold_annotations,
        )
        selected_summary = selected_evaluation.summary_dict()
        phase_05_summary = phase_05_evaluation.summary_dict()
        report = {
            "algorithm": "bm25_lexical_verifier_cross_validated_citation_audit",
            "baseline_phase_05_policy": {
                "policy": PHASE_05_POLICY.as_dict(),
                "summary": phase_05_summary,
            },
            "claims": {
                "path": str(args.claims_path),
                "sha256": sha256_file(args.claims_path),
            },
            "comparison_to_phase_05_policy": _citation_audit_deltas(
                selected_summary,
                phase_05_summary,
            ),
            "corpus_sha256": corpus_sha256,
            "index": {
                "document_count": bm25_index.document_count,
                "parameters": {"b": bm25_index.b, "k1": bm25_index.k1},
                "path": str(args.index_path),
                "vocabulary_size": bm25_index.vocabulary_size,
            },
            "model": {"path": str(args.model_path), **bundle.summary_dict()},
            "policy_selection": {
                "report_path": str(args.calibration_report_path),
                "report_sha256": sha256_file(args.calibration_report_path),
                "report_schema_version": CALIBRATION_REPORT_SCHEMA,
            },
            "runtime_settings": {
                "retrieval_k": args.retrieval_k,
                "selected_policy": selected_policy.as_dict(),
            },
            "runtime_timing": {
                "claim_count": len(runtime_claims),
                "per_claim_milliseconds": 1_000 * runtime_elapsed_seconds / len(runtime_claims),
                "total_seconds": runtime_elapsed_seconds,
            },
            "schema_version": "evidence_agent_citation_audit_evaluation_v1",
            "selected_decisions": [trace.decision_dict() for trace in audited_traces],
            "selected_policy": selected_policy.as_dict(),
            "summary": selected_summary,
            "trace_artifact": trace_artifact,
        }
        write_verification_report(report, args.report_path)
        print(
            json.dumps(
                {
                    "comparison_to_phase_05_policy": report["comparison_to_phase_05_policy"],
                    "report_path": str(args.report_path),
                    "selected_policy": selected_policy.as_dict(),
                    "summary": selected_summary,
                    "trace_path": str(args.trace_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        f"Command '{args.command}' is not available yet. "
        f"{UNIMPLEMENTED_COMMANDS[args.command]}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
