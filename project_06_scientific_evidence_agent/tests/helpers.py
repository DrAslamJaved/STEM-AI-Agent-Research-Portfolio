"""Small, explicit SciFact fixtures used without network access."""

from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def write_minimal_scifact_dataset(root: Path, *, include_cross_validation: bool = False) -> Path:
    """Write a valid three-claim SciFact-shaped fixture and return its root."""
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        root / "corpus.jsonl",
        [
            {
                "doc_id": 10,
                "title": "Document one",
                "abstract": ["Sentence zero.", "Sentence one."],
                "structured": False,
            },
            {
                "doc_id": 11,
                "title": "Document two",
                "abstract": ["A contradiction sentence."],
                "structured": True,
            },
        ],
    )
    _write_jsonl(
        root / "claims_train.jsonl",
        [
            {
                "id": 1,
                "claim": "A supported claim.",
                "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
                "cited_doc_ids": [10],
            }
        ],
    )
    _write_jsonl(
        root / "claims_dev.jsonl",
        [
            {
                "id": 2,
                "claim": "A contradicted claim.",
                "evidence": {"11": [{"label": "CONTRADICT", "sentences": [0]}]},
                "cited_doc_ids": [11],
            }
        ],
    )
    _write_jsonl(
        root / "claims_test.jsonl",
        [
            {
                "id": 3,
                "claim": "A hidden-test style claim.",
                "evidence": {},
                "cited_doc_ids": [10],
            }
        ],
    )

    if include_cross_validation:
        cross_validation = root / "cross_validation"
        for fold_number in range(1, 6):
            fold = cross_validation / f"fold_{fold_number}"
            fold.mkdir(parents=True)
            _write_jsonl(
                fold / f"claims_train_{fold_number}.jsonl",
                [
                    {
                        "id": 1,
                        "claim": "A supported claim.",
                        "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
                        "cited_doc_ids": [10],
                    }
                ],
            )
            _write_jsonl(
                fold / f"claims_dev_{fold_number}.jsonl",
                [
                    {
                        "id": 2,
                        "claim": "A contradicted claim.",
                        "evidence": {"11": [{"label": "CONTRADICT", "sentences": [0]}]},
                        "cited_doc_ids": [11],
                    }
                ],
            )
    return root


def write_verification_scifact_dataset(root: Path) -> Path:
    """Write a small train/dev fixture containing all verifier label classes."""
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        root / "corpus.jsonl",
        [
            {
                "doc_id": 10,
                "title": "Inflammation study",
                "abstract": [
                    "Aspirin reduces inflammation in the study population.",
                    "The study included adult participants.",
                ],
                "structured": False,
            },
            {
                "doc_id": 11,
                "title": "Contradiction study",
                "abstract": [
                    "Aspirin does not reduce inflammation in this experiment.",
                    "The control group received placebo.",
                ],
                "structured": False,
            },
            {
                "doc_id": 12,
                "title": "Unrelated observation",
                "abstract": [
                    "Aspirin tablets were white.",
                    "The laboratory measured tablet mass.",
                ],
                "structured": False,
            },
        ],
    )
    training_records = [
        {
            "id": 1,
            "claim": "Aspirin reduces inflammation.",
            "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
            "cited_doc_ids": [10],
        },
        {
            "id": 2,
            "claim": "Aspirin reduces inflammation.",
            "evidence": {"11": [{"label": "CONTRADICT", "sentences": [0]}]},
            "cited_doc_ids": [11],
        },
        {
            "id": 3,
            "claim": "Aspirin reduces inflammation.",
            "evidence": {},
            "cited_doc_ids": [12],
        },
    ]
    _write_jsonl(root / "claims_train.jsonl", training_records)
    _write_jsonl(
        root / "claims_dev.jsonl",
        [
            {
                "id": 4,
                "claim": "Aspirin reduces inflammation.",
                "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
                "cited_doc_ids": [10],
            },
            {
                "id": 5,
                "claim": "Aspirin reduces inflammation.",
                "evidence": {"11": [{"label": "CONTRADICT", "sentences": [0]}]},
                "cited_doc_ids": [11],
            },
            {
                "id": 6,
                "claim": "Aspirin reduces inflammation.",
                "evidence": {},
                "cited_doc_ids": [12, 12],
            },
        ],
    )
    _write_jsonl(
        root / "claims_test.jsonl",
        [
            {
                "id": 7,
                "claim": "A hidden-test style claim.",
                "evidence": {},
                "cited_doc_ids": [12],
            }
        ],
    )
    return root


def write_citation_audit_scifact_dataset(root: Path) -> Path:
    """Write a five-fold fixture whose supplied folds also contain dev IDs.

    This deliberately mirrors the SciFact release property that requires the
    calibration code to filter supplied folds to ordinary-training IDs.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        root / "corpus.jsonl",
        [
            {
                "doc_id": 10,
                "title": "Inflammation study",
                "abstract": [
                    "Aspirin reduces inflammation in the study population.",
                    "The study included adult participants.",
                ],
                "structured": False,
            },
            {
                "doc_id": 11,
                "title": "Contradiction study",
                "abstract": [
                    "Aspirin does not reduce inflammation in this experiment.",
                    "The control group received placebo.",
                ],
                "structured": False,
            },
            {
                "doc_id": 12,
                "title": "Unrelated observation",
                "abstract": [
                    "Aspirin tablets were white.",
                    "The laboratory measured tablet mass.",
                ],
                "structured": False,
            },
        ],
    )

    def record(claim_id: int, label: str) -> dict:
        if label == "SUPPORT":
            return {
                "id": claim_id,
                "claim": "Aspirin reduces inflammation.",
                "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
                "cited_doc_ids": [10],
            }
        if label == "CONTRADICT":
            return {
                "id": claim_id,
                "claim": "Aspirin reduces inflammation.",
                "evidence": {"11": [{"label": "CONTRADICT", "sentences": [0]}]},
                "cited_doc_ids": [11],
            }
        return {
            "id": claim_id,
            "claim": "Aspirin reduces inflammation.",
            "evidence": {},
            "cited_doc_ids": [12],
        }

    labels = (
        "SUPPORT",
        "CONTRADICT",
        "NO_EVIDENCE",
        "SUPPORT",
        "CONTRADICT",
        "NO_EVIDENCE",
        "SUPPORT",
        "CONTRADICT",
        "NO_EVIDENCE",
        "SUPPORT",
    )
    training_records = [record(claim_id, label) for claim_id, label in enumerate(labels, start=1)]
    _write_jsonl(root / "claims_train.jsonl", training_records)
    ordinary_development_record = record(100, "SUPPORT")
    _write_jsonl(root / "claims_dev.jsonl", [ordinary_development_record])
    _write_jsonl(root / "claims_test.jsonl", [record(101, "NO_EVIDENCE")])

    cross_validation = root / "cross_validation"
    for fold_number, validation_ids in enumerate(
        ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)), start=1
    ):
        fold = cross_validation / f"fold_{fold_number}"
        fold.mkdir(parents=True)
        _write_jsonl(
            fold / f"claims_dev_{fold_number}.jsonl",
            [
                *(training_records[claim_id - 1] for claim_id in validation_ids),
                ordinary_development_record,
            ],
        )
        _write_jsonl(
            fold / f"claims_train_{fold_number}.jsonl",
            [
                record_item
                for record_item in training_records
                if record_item["id"] not in validation_ids
            ],
        )
    return root
