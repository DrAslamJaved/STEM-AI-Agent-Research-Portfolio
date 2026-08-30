"""Audit raw-representation duplicates and transparent feature collisions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.representations import (
    DRUG_FEATURE_COLUMNS,
    TARGET_FEATURE_COLUMNS,
    build_drug_feature_table,
    build_target_feature_table,
)


DEFAULT_TOP_N = 10

ENTITY_PAIR_COLUMNS = (
    "entity_type",
    "entity_id_a",
    "entity_id_b",
    "raw_representation_equal",
    "exact_feature_vector_equal",
    "range_normalized_mean_absolute_difference",
    "range_normalized_max_absolute_difference",
    "included_as_nearest_pair",
)


class CollisionAuditError(ValueError):
    """Raised when a representation collision audit cannot be trusted."""


@dataclass(frozen=True)
class EntityCollisionAudit:
    """One entity type's collision evidence and local detailed pair table."""

    summary: dict[str, Any]
    pair_table: pd.DataFrame


@dataclass(frozen=True)
class CollisionAuditRun:
    """Combined Davis entity audit ready for JSON and local CSV outputs."""

    report: dict[str, Any]
    pair_table: pd.DataFrame


def _positive_integer(value: object, name: str) -> int:
    """Validate a positive integer without silently coercing decimals."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise CollisionAuditError(f"{name} must be an integer.") from error

    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise CollisionAuditError(f"{name} must be an integer.")

    integer_value = int(numeric_value)
    if integer_value < 1:
        raise CollisionAuditError(f"{name} must be at least one.")

    return integer_value


def load_representation_mapping(path: str | Path, label: str) -> dict[str, str]:
    """Load a strict non-empty Davis JSON mapping without labels or outcomes."""
    source = Path(path)

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CollisionAuditError(f"Could not load {label}: {source}") from error

    if not isinstance(payload, dict) or not payload:
        raise CollisionAuditError(f"{label} must be a non-empty JSON object.")

    normalized: dict[str, str] = {}
    for identifier, representation in payload.items():
        if not isinstance(identifier, str) or not identifier.strip():
            raise CollisionAuditError(f"{label} contains an invalid identifier.")
        if not isinstance(representation, str) or not representation.strip():
            raise CollisionAuditError(
                f"{label} contains an invalid representation."
            )

        normalized_identifier = identifier.strip()
        if normalized_identifier in normalized:
            raise CollisionAuditError(
                f"{label} contains duplicate normalized identifiers."
            )

        normalized[normalized_identifier] = representation.strip()

    return normalized


def _groups_from_keys(
    entity_ids: list[str],
    keys: list[object],
) -> list[tuple[str, ...]]:
    """Return deterministic duplicate groups for raw or feature signatures."""
    grouped: dict[object, list[str]] = {}

    for entity_id, key in zip(entity_ids, keys, strict=True):
        grouped.setdefault(key, []).append(entity_id)

    duplicate_groups = [
        tuple(sorted(group))
        for group in grouped.values()
        if len(group) > 1
    ]

    return sorted(
        duplicate_groups,
        key=lambda group: (-len(group), group),
    )


def _pair_count(groups: list[tuple[str, ...]]) -> int:
    """Return the total number of unordered pairs represented by groups."""
    return int(sum(len(group) * (len(group) - 1) // 2 for group in groups))


def _group_examples(
    groups: list[tuple[str, ...]],
    top_n: int,
) -> list[dict[str, object]]:
    """Return ID-only duplicate-group examples without raw strings/sequences."""
    return [
        {
            "entity_count": len(group),
            "entity_ids": list(group),
        }
        for group in groups[:top_n]
    ]


def _validated_entity_inputs(
    *,
    entity_type: str,
    representations: dict[str, str],
    feature_table: pd.DataFrame,
    id_column: str,
    feature_columns: tuple[str, ...],
) -> tuple[list[str], list[str], np.ndarray]:
    """Validate an entity feature table against its raw representation mapping."""
    if entity_type not in {"drug", "target"}:
        raise CollisionAuditError("entity_type must be drug or target.")

    required_columns = {id_column, *feature_columns}
    missing_columns = required_columns.difference(feature_table.columns)
    if missing_columns:
        raise CollisionAuditError(
            f"{entity_type} feature table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    table = feature_table.loc[:, [id_column, *feature_columns]].copy()
    if table.empty:
        raise CollisionAuditError(f"{entity_type} feature table is empty.")
    if table[id_column].isna().any():
        raise CollisionAuditError(f"{entity_type} feature table has missing IDs.")

    table[id_column] = table[id_column].astype(str).str.strip()
    if table[id_column].eq("").any() or table[id_column].duplicated().any():
        raise CollisionAuditError(
            f"{entity_type} feature table has invalid or duplicate IDs."
        )

    entity_ids = sorted(table[id_column].tolist())
    if set(entity_ids) != set(representations):
        raise CollisionAuditError(
            f"{entity_type} raw and feature identifiers do not match."
        )

    indexed = table.set_index(id_column).loc[entity_ids, list(feature_columns)]
    try:
        matrix = indexed.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise CollisionAuditError(
            f"{entity_type} feature values must be numeric."
        ) from error

    if not np.isfinite(matrix).all():
        raise CollisionAuditError(
            f"{entity_type} feature values must be finite."
        )

    raw_representations = [representations[entity_id] for entity_id in entity_ids]
    return entity_ids, raw_representations, matrix


def _distance_candidates(
    entity_ids: list[str],
    raw_representations: list[str],
    matrix: np.ndarray,
    feature_columns: tuple[str, ...],
    top_n: int,
) -> tuple[list[tuple[float, float, str, str]], tuple[str, ...]]:
    """Return closest distinct-raw pairs under a fixed range-normalized metric."""
    feature_ranges = matrix.max(axis=0) - matrix.min(axis=0)
    active_mask = feature_ranges > 0.0
    active_columns = tuple(
        column
        for column, active in zip(feature_columns, active_mask, strict=True)
        if bool(active)
    )

    if not active_columns:
        raise CollisionAuditError(
            "Every feature has zero range; descriptor distances are undefined."
        )

    active_matrix = matrix[:, active_mask]
    active_ranges = feature_ranges[active_mask]
    candidates: list[tuple[float, float, str, str]] = []

    for left_index in range(len(entity_ids) - 1):
        differences = np.abs(
            active_matrix[left_index + 1 :] - active_matrix[left_index]
        ) / active_ranges
        mean_distances = differences.mean(axis=1)
        max_distances = differences.max(axis=1)

        for relative_index, (mean_distance, max_distance) in enumerate(
            zip(mean_distances, max_distances, strict=True)
        ):
            right_index = left_index + 1 + relative_index

            if raw_representations[left_index] == raw_representations[right_index]:
                continue

            candidates.append(
                (
                    float(mean_distance),
                    float(max_distance),
                    entity_ids[left_index],
                    entity_ids[right_index],
                )
            )

    return sorted(candidates)[:top_n], active_columns


def audit_entity_representations(
    *,
    entity_type: str,
    representations: dict[str, str],
    feature_table: pd.DataFrame,
    id_column: str,
    feature_columns: tuple[str, ...],
    top_n: int = DEFAULT_TOP_N,
) -> EntityCollisionAudit:
    """Audit raw duplicates, exact descriptor collisions, and nearest pairs."""
    checked_top_n = _positive_integer(top_n, "top_n")
    entity_ids, raw_representations, matrix = _validated_entity_inputs(
        entity_type=entity_type,
        representations=representations,
        feature_table=feature_table,
        id_column=id_column,
        feature_columns=feature_columns,
    )

    raw_groups = _groups_from_keys(entity_ids, raw_representations)
    feature_signatures = [tuple(row.tolist()) for row in matrix]
    feature_groups = _groups_from_keys(entity_ids, feature_signatures)
    id_to_index = {entity_id: index for index, entity_id in enumerate(entity_ids)}
    pair_records: dict[tuple[str, str], dict[str, object]] = {}

    for group in feature_groups:
        for left_position, entity_id_a in enumerate(group[:-1]):
            for entity_id_b in group[left_position + 1 :]:
                index_a = id_to_index[entity_id_a]
                index_b = id_to_index[entity_id_b]
                pair_records[(entity_id_a, entity_id_b)] = {
                    "entity_type": entity_type,
                    "entity_id_a": entity_id_a,
                    "entity_id_b": entity_id_b,
                    "raw_representation_equal": (
                        raw_representations[index_a]
                        == raw_representations[index_b]
                    ),
                    "exact_feature_vector_equal": True,
                    "range_normalized_mean_absolute_difference": 0.0,
                    "range_normalized_max_absolute_difference": 0.0,
                    "included_as_nearest_pair": False,
                }

    nearest_pairs, active_columns = _distance_candidates(
        entity_ids,
        raw_representations,
        matrix,
        feature_columns,
        checked_top_n,
    )

    for mean_distance, max_distance, entity_id_a, entity_id_b in nearest_pairs:
        index_a = id_to_index[entity_id_a]
        index_b = id_to_index[entity_id_b]
        key = (entity_id_a, entity_id_b)
        record = pair_records.get(
            key,
            {
                "entity_type": entity_type,
                "entity_id_a": entity_id_a,
                "entity_id_b": entity_id_b,
                "raw_representation_equal": (
                    raw_representations[index_a]
                    == raw_representations[index_b]
                ),
                "exact_feature_vector_equal": bool(
                    np.array_equal(matrix[index_a], matrix[index_b])
                ),
                "range_normalized_mean_absolute_difference": mean_distance,
                "range_normalized_max_absolute_difference": max_distance,
                "included_as_nearest_pair": True,
            },
        )
        record["included_as_nearest_pair"] = True
        pair_records[key] = record

    pair_table = pd.DataFrame.from_records(
        list(pair_records.values()),
        columns=ENTITY_PAIR_COLUMNS,
    ).sort_values(
        [
            "entity_type",
            "included_as_nearest_pair",
            "range_normalized_mean_absolute_difference",
            "entity_id_a",
            "entity_id_b",
        ],
        ascending=[True, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    exact_feature_pair_count = _pair_count(feature_groups)
    raw_duplicate_pair_count = _pair_count(raw_groups)
    distinct_raw_feature_collision_pair_count = int(
        sum(
            not bool(record["raw_representation_equal"])
            for record in pair_records.values()
            if bool(record["exact_feature_vector_equal"])
        )
    )
    zero_range_columns = tuple(
        column
        for column in feature_columns
        if column not in active_columns
    )

    nearest_records = pair_table.loc[
        pair_table["included_as_nearest_pair"]
    ].head(checked_top_n)

    summary = {
        "entity_type": entity_type,
        "entity_count": len(entity_ids),
        "feature_column_count": len(feature_columns),
        "active_distance_feature_count": len(active_columns),
        "active_distance_feature_columns": list(active_columns),
        "zero_range_feature_columns": list(zero_range_columns),
        "raw_duplicate_group_count": len(raw_groups),
        "raw_duplicate_entity_count": int(sum(map(len, raw_groups))),
        "raw_duplicate_pair_count": raw_duplicate_pair_count,
        "raw_duplicate_group_examples": _group_examples(
            raw_groups,
            checked_top_n,
        ),
        "exact_feature_collision_group_count": len(feature_groups),
        "exact_feature_collision_pair_count": exact_feature_pair_count,
        "distinct_raw_feature_collision_pair_count": (
            distinct_raw_feature_collision_pair_count
        ),
        "exact_feature_collision_group_examples": _group_examples(
            feature_groups,
            checked_top_n,
        ),
        "nearest_distinct_raw_representation_pairs": _records(nearest_records),
    }

    return EntityCollisionAudit(summary=summary, pair_table=pair_table)


def _json_value(value: object) -> object:
    """Return strict JSON-safe values for pandas and NumPy scalars."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a short table to JSON-safe records."""
    return [
        {name: _json_value(value) for name, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def run_collision_audit(
    ligands: dict[str, str],
    proteins: dict[str, str],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> CollisionAuditRun:
    """Audit raw inputs and transparent descriptors without labels or models."""
    checked_top_n = _positive_integer(top_n, "top_n")
    drug_audit = audit_entity_representations(
        entity_type="drug",
        representations=ligands,
        feature_table=build_drug_feature_table(ligands),
        id_column="drug_id",
        feature_columns=DRUG_FEATURE_COLUMNS,
        top_n=checked_top_n,
    )
    target_audit = audit_entity_representations(
        entity_type="target",
        representations=proteins,
        feature_table=build_target_feature_table(proteins),
        id_column="target_id",
        feature_columns=TARGET_FEATURE_COLUMNS,
        top_n=checked_top_n,
    )
    pair_table = pd.concat(
        [drug_audit.pair_table, target_audit.pair_table],
        ignore_index=True,
    ).loc[:, ENTITY_PAIR_COLUMNS]

    if tuple(pair_table.columns) != ENTITY_PAIR_COLUMNS:
        raise CollisionAuditError(
            "Detailed collision-pair columns do not match the frozen contract."
        )

    report = {
        "audit_scope": "unsupervised_raw_and_feature_representation_only",
        "outcome_values_used": False,
        "model_predictions_used": False,
        "outer_holdout_raw_representations_included": True,
        "top_n": checked_top_n,
        "distance_definition": {
            "name": "range_normalized_descriptor_distance",
            "mean_absolute_difference": (
                "Mean absolute feature difference after dividing each active "
                "feature by its entity-level range."
            ),
            "max_absolute_difference": (
                "Maximum absolute feature difference after the same range "
                "normalization."
            ),
            "zero_range_features": (
                "Excluded only from distance because they cannot distinguish "
                "entities."
            ),
        },
        "drug_audit": drug_audit.summary,
        "target_audit": target_audit.summary,
        "detailed_pair_record_count": int(len(pair_table)),
        "interpretation_limits": [
            "Equal raw representations may reflect the benchmark mapping and "
            "are not automatically errors.",
            "Exact feature collisions show information lost by this transparent "
            "representation, not biological equivalence.",
            "Nearest descriptor pairs are a quality diagnostic only; they are "
            "not chemical similarity, sequence homology, or causal evidence.",
            "The audit includes raw representations for all benchmark entities, "
            "including outer-holdout entities, but uses no outcomes. It is "
            "documentation only and must not guide retrospective model changes.",
        ],
    }

    return CollisionAuditRun(report=report, pair_table=pair_table)


def write_collision_audit_summary(
    run: CollisionAuditRun,
    output_path: str | Path,
) -> Path:
    """Write compact version-controlled collision evidence."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(run.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def write_collision_pairs(
    pair_table: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write ignored detailed collision and nearest-pair rows locally."""
    if tuple(pair_table.columns) != ENTITY_PAIR_COLUMNS:
        raise CollisionAuditError(
            "Collision-pair columns do not match the frozen contract."
        )

    if pair_table.empty:
        raise CollisionAuditError("Collision-pair table must not be empty.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(destination, index=False, float_format="%.17g")
    return destination


def main(argv: list[str] | None = None) -> int:
    """Run a Davis raw-input and feature-collision audit."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit Davis raw representations and transparent feature collisions "
            "without using labels or model predictions."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/davis"),
        help="Directory containing ligands_can.txt and proteins.txt.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of duplicate-group and nearest-pair examples to retain.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_feature_collision_audit.json"),
        help="Version-controlled JSON audit destination.",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path("data/interim/davis_feature_collision_pairs.csv"),
        help="Ignored detailed local pair-table CSV destination.",
    )
    args = parser.parse_args(argv)

    try:
        ligands = load_representation_mapping(
            args.data_dir / "ligands_can.txt",
            "ligands_can.txt",
        )
        proteins = load_representation_mapping(
            args.data_dir / "proteins.txt",
            "proteins.txt",
        )
        run = run_collision_audit(ligands, proteins, top_n=args.top_n)
        summary_path = write_collision_audit_summary(run, args.summary_output)
        pairs_path = write_collision_pairs(run.pair_table, args.pairs_output)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Feature-collision audit failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(run.report, indent=2, sort_keys=True))
    print(f"Collision audit written to: {summary_path}")
    print(f"Detailed collision pairs written to: {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
