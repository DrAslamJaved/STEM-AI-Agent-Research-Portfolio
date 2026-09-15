"""Deterministic, leakage-diagnosed evaluation splits for binary DTI records."""

from __future__ import annotations

from math import ceil
from random import Random
from typing import Any, Iterable, Mapping


def _rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    if len(rows) < 2:
        raise ValueError("At least two DTI records are required.")
    required = ("drug_id", "target_id", "label")
    for row in rows:
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError("DTI record missing required field(s): " + ", ".join(missing))
        if int(row["label"]) not in (0, 1):
            raise ValueError("DTI labels must be binary.")
    return rows


def _holdout_count(size: int, fraction: float) -> int:
    if not 0.0 < fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")
    if size < 2:
        raise ValueError("At least two distinct units are required for a split.")
    return min(size - 1, max(1, ceil(size * fraction)))


def pair_random_split(records: Iterable[Mapping[str, Any]], *, test_fraction: float = 0.2, seed: int = 20260915):
    """Return a deterministic label-stratified pair-random split.

    Drug and target entities may appear in both partitions. This split is useful
    as a reference only; it must not be interpreted as a cold-start result.
    """
    rows = _rows(records)
    indices_by_label = {0: [], 1: []}
    for index, row in enumerate(rows):
        indices_by_label[int(row["label"])].append(index)
    rng = Random(seed)
    test_indices: set[int] = set()
    for label, indices in indices_by_label.items():
        if not indices:
            continue
        order = list(indices)
        rng.shuffle(order)
        if len(order) == 1:
            continue
        test_indices.update(order[:_holdout_count(len(order), test_fraction)])
    if not test_indices:
        order = list(range(len(rows)))
        rng.shuffle(order)
        test_indices.add(order[0])
    return ([row for index, row in enumerate(rows) if index not in test_indices],
            [row for index, row in enumerate(rows) if index in test_indices])


def _cold_entity_split(records: Iterable[Mapping[str, Any]], *, entity_field: str, test_fraction: float, seed: int):
    rows = _rows(records)
    entities = sorted({str(row[entity_field]) for row in rows})
    count = _holdout_count(len(entities), test_fraction)
    rng = Random(seed)
    rng.shuffle(entities)
    held_out = set(entities[:count])
    train = [row for row in rows if str(row[entity_field]) not in held_out]
    test = [row for row in rows if str(row[entity_field]) in held_out]
    if not train or not test:
        raise ValueError("Cold split produced an empty partition.")
    return train, test, tuple(sorted(held_out))


def cold_drug_split(records: Iterable[Mapping[str, Any]], *, test_fraction: float = 0.2, seed: int = 20260915):
    """Hold out complete drugs; no drug identity may cross partitions."""
    return _cold_entity_split(records, entity_field="drug_id", test_fraction=test_fraction, seed=seed)


def cold_target_split(records: Iterable[Mapping[str, Any]], *, test_fraction: float = 0.2, seed: int = 20260915):
    """Hold out complete targets; no target identity may cross partitions."""
    return _cold_entity_split(records, entity_field="target_id", test_fraction=test_fraction, seed=seed)


def split_diagnostics(train_records: Iterable[Mapping[str, Any]], test_records: Iterable[Mapping[str, Any]], *, split_name: str) -> dict[str, Any]:
    """Summarise partitions and identity overlap required for interpretation."""
    train, test = _rows(train_records), _rows(test_records)
    train_drugs = {str(row["drug_id"]) for row in train}
    test_drugs = {str(row["drug_id"]) for row in test}
    train_targets = {str(row["target_id"]) for row in train}
    test_targets = {str(row["target_id"]) for row in test}
    train_pairs = {(str(row["drug_id"]), str(row["target_id"])) for row in train}
    test_pairs = {(str(row["drug_id"]), str(row["target_id"])) for row in test}
    return {
        "split_name": split_name,
        "train_records": len(train),
        "test_records": len(test),
        "train_positive_count": sum(int(row["label"]) for row in train),
        "test_positive_count": sum(int(row["label"]) for row in test),
        "train_positive_rate": sum(int(row["label"]) for row in train) / len(train),
        "test_positive_rate": sum(int(row["label"]) for row in test) / len(test),
        "shared_drug_count": len(train_drugs & test_drugs),
        "shared_target_count": len(train_targets & test_targets),
        "shared_pair_count": len(train_pairs & test_pairs),
        "train_unique_drugs": len(train_drugs),
        "test_unique_drugs": len(test_drugs),
        "train_unique_targets": len(train_targets),
        "test_unique_targets": len(test_targets),
    }


def validate_split_diagnostics(diagnostics: Mapping[str, Any], *, cold_entity: str | None = None) -> None:
    """Raise when a split violates its declared no-overlap policy."""
    if int(diagnostics["shared_pair_count"]) != 0:
        raise ValueError("Train and test partitions share DTI pairs.")
    if cold_entity == "drug" and int(diagnostics["shared_drug_count"]) != 0:
        raise ValueError("Cold-drug split shares drug identities.")
    if cold_entity == "target" and int(diagnostics["shared_target_count"]) != 0:
        raise ValueError("Cold-target split shares target identities.")
