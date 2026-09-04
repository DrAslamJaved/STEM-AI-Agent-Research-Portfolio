"""Small, explicit command-line interface for the staged project workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from evidence_agent import __version__
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
from evidence_agent.retrieval.bm25 import (
    build_bm25_index,
    load_bm25_index,
    write_bm25_index,
)
from evidence_agent.retrieval.scifact import (
    load_gold_evidence_documents,
    load_runtime_claims,
    load_scifact_corpus,
)


UNIMPLEMENTED_COMMANDS = {
    "evaluate": "Complete the relevant evaluation pipeline before running this command.",
}

DEFAULT_CORPUS_PATH = Path("data/raw/scifact/data/corpus.jsonl")
DEFAULT_DEV_CLAIMS_PATH = Path("data/raw/scifact/data/claims_dev.jsonl")
DEFAULT_INDEX_PATH = Path("artifacts/scifact_bm25_index.json")
DEFAULT_RETRIEVAL_REPORT_PATH = Path("results/retrieval_baseline_dev.json")


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
    evaluate = subparsers.add_parser("evaluate", help="Run a configured evaluation.")
    evaluate.add_argument("--config", type=str, help="Path to a YAML experiment config.")
    return parser


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
                    "current_phase": "retrieval_baseline",
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

    print(
        f"Command '{args.command}' is not available yet. "
        f"{UNIMPLEMENTED_COMMANDS[args.command]}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
