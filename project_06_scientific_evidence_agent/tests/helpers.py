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
