"""Create fixed, leakage-auditable Davis train/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)

class SplitDesignError(ValueError):
    """Raised when a reproducible split cannot be constructed safely."""


POLICIES = ("random_pair", "cold_drug", "cold_target")

COLD_DRUG_N_SPLITS = 5
COLD_DRUG_OUTER_FOLD = 4
REQUIRED_COLUMNS = ("observed_pair_index", "drug_id", "target_id")


POLICY_INTERPRETATIONS = {
    "random_pair": (
        "Known-drug/known-target interpolation benchmark; entity overlap can "
        "make this estimate optimistic for unseen-entity claims."
    ),
    "cold_drug": (
        "Generalization to drugs absent from training; target overlap may remain."
    ),
    "cold_target": (
        "Generalization to targets absent from training; drug overlap may remain."
    ),
}


@dataclass(frozen=True)
class SplitIndices:
    """Positional train/test indices for one fixed split policy."""

    policy: str
    splitter_name: str
    random_state: int
    requested_test_size: float
    fold_index: int | None
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]


@dataclass(frozen=True)
class SplitAudit:
    """Leakage and class-balance evidence for one split policy."""

    policy: str
    splitter_name: str
    fold_index: int | None
    reference_label_column: str
    random_state: int
    requested_test_size: float
    train_pair_count: int
    test_pair_count: int
    train_positive_count: int
    train_negative_count: int
    train_positive_rate: float
    test_positive_count: int
    test_negative_count: int
    test_positive_rate: float
    train_drug_count: int
    test_drug_count: int
    drug_overlap_count: int
    train_target_count: int
    test_target_count: int
    target_overlap_count: int
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validated_labels(table: pd.DataFrame, label_column: str) -> pd.Series:
    """Validate required identifiers and return binary reference labels."""
    missing_columns = set(REQUIRED_COLUMNS).union({label_column}).difference(
        table.columns
    )
    if missing_columns:
        raise SplitDesignError(
            f"Interaction table is missing columns: {sorted(missing_columns)}"
        )
    if table.empty:
        raise SplitDesignError("Interaction table is empty.")
    if table["observed_pair_index"].isna().any():
        raise SplitDesignError("observed_pair_index contains missing values.")
    if table["observed_pair_index"].duplicated().any():
        raise SplitDesignError("observed_pair_index values must be unique.")
    if table.duplicated(["drug_id", "target_id"]).any():
        raise SplitDesignError("Interaction table contains duplicate drug-target pairs.")

    for column in ("drug_id", "target_id"):
        values = table[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise SplitDesignError(f"{column} contains an empty or missing identifier.")

    labels = pd.to_numeric(table[label_column], errors="coerce")
    if labels.isna().any():
        raise SplitDesignError(f"{label_column} must contain numeric binary labels.")

    invalid_values = set(labels.unique()).difference({0, 1})
    if invalid_values:
        raise SplitDesignError(f"{label_column} must contain only 0 and 1 values.")
    if labels.nunique() != 2:
        raise SplitDesignError(
            f"{label_column} must contain both positive and negative examples."
        )

    return labels.astype("int8")


def _validated_test_size(test_size: float) -> float:
    """Return a legal test fraction."""
    try:
        value = float(test_size)
    except (TypeError, ValueError) as error:
        raise SplitDesignError("test_size must be numeric.") from error

    if not 0 < value < 1:
        raise SplitDesignError("test_size must be strictly between zero and one.")

    return value


def _assert_split_invariants(table: pd.DataFrame, split: SplitIndices) -> None:
    """Raise if records overlap or policy-specific entity rules are violated."""
    train_positions = set(split.train_positions)
    test_positions = set(split.test_positions)
    all_positions = set(range(len(table)))

    if not train_positions or not test_positions:
        raise SplitDesignError("Train and test partitions must both be non-empty.")
    if train_positions.intersection(test_positions):
        raise SplitDesignError("Train and test partitions overlap.")
    if train_positions.union(test_positions) != all_positions:
        raise SplitDesignError("Every interaction must appear in exactly one partition.")

    train_frame = table.iloc[list(split.train_positions)]
    test_frame = table.iloc[list(split.test_positions)]
    drug_overlap = set(train_frame["drug_id"]).intersection(test_frame["drug_id"])
    target_overlap = set(train_frame["target_id"]).intersection(
        test_frame["target_id"]
    )

    if split.policy == "cold_drug" and drug_overlap:
        raise SplitDesignError("cold_drug split contains overlapping drugs.")
    if split.policy == "cold_target" and target_overlap:
        raise SplitDesignError("cold_target split contains overlapping targets.")


def create_train_test_split(
    table: pd.DataFrame,
    policy: str,
    *,
    reference_label_column: str,
    test_size: float = 0.20,
    random_state: int = 20260830,
) -> SplitIndices:
    """Create one reproducible random-pair or entity-cold split."""
    if policy not in POLICIES:
        raise SplitDesignError(
            f"Unknown policy: {policy}. Choose one of {list(POLICIES)}."
        )

    labels = _validated_labels(table, reference_label_column)
    requested_test_size = _validated_test_size(test_size)
    state = int(random_state)
    features = np.zeros((len(table), 1), dtype=np.int8)

    if policy == "random_pair":
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=requested_test_size,
            random_state=state,
        )
        train_positions, test_positions = next(splitter.split(features, labels))
        splitter_name = "StratifiedShuffleSplit"
        fold_index = None

    elif policy == "cold_drug":
        groups = table["drug_id"].astype(str).to_numpy()
        unique_group_count = pd.Series(groups).nunique()

        if unique_group_count < COLD_DRUG_N_SPLITS:
            raise SplitDesignError(
                "cold_drug requires at least "
                f"{COLD_DRUG_N_SPLITS} unique drug identifiers."
            )

        splitter = StratifiedGroupKFold(
            n_splits=COLD_DRUG_N_SPLITS,
            shuffle=True,
            random_state=state,
        )
        folds = list(splitter.split(features, labels, groups=groups))
        train_positions, test_positions = folds[COLD_DRUG_OUTER_FOLD]
        splitter_name = "StratifiedGroupKFold"
        fold_index = COLD_DRUG_OUTER_FOLD

    else:
        groups = table["target_id"].astype(str).to_numpy()

        if pd.Series(groups).nunique() < 2:
            raise SplitDesignError(
                "cold_target requires at least two unique target identifiers."
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=requested_test_size,
            random_state=state,
        )
        train_positions, test_positions = next(
            splitter.split(features, labels, groups=groups)
        )
        splitter_name = "GroupShuffleSplit"
        fold_index = None

    split = SplitIndices(
        policy=policy,
        splitter_name=splitter_name,
        random_state=state,
        requested_test_size=requested_test_size,
        fold_index=fold_index,
        train_positions=tuple(int(position) for position in train_positions),
        test_positions=tuple(int(position) for position in test_positions),
    )
    _assert_split_invariants(table, split)
    return split

def create_all_split_policies(
    table: pd.DataFrame,
    *,
    reference_label_column: str,
    test_size: float = 0.20,
    random_state: int = 20260830,
) -> list[SplitIndices]:
    """Create all required leakage-comparison split policies."""
    return [
        create_train_test_split(
            table,
            policy,
            reference_label_column=reference_label_column,
            test_size=test_size,
            random_state=random_state,
        )
        for policy in POLICIES
    ]


def audit_train_test_split(
    table: pd.DataFrame,
    split: SplitIndices,
    *,
    reference_label_column: str,
) -> SplitAudit:
    """Summarize class prevalence and entity overlap for one split."""
    labels = _validated_labels(table, reference_label_column)
    _assert_split_invariants(table, split)

    train_positions = list(split.train_positions)
    test_positions = list(split.test_positions)
    train_frame = table.iloc[train_positions]
    test_frame = table.iloc[test_positions]
    train_labels = labels.iloc[train_positions]
    test_labels = labels.iloc[test_positions]

    train_positive_count = int(train_labels.sum())
    test_positive_count = int(test_labels.sum())

    return SplitAudit(
        policy=split.policy,
        splitter_name=split.splitter_name,
        fold_index=split.fold_index,
        reference_label_column=reference_label_column,
        random_state=split.random_state,
        requested_test_size=split.requested_test_size,
        train_pair_count=int(len(train_frame)),
        test_pair_count=int(len(test_frame)),
        train_positive_count=train_positive_count,
        train_negative_count=int(len(train_frame) - train_positive_count),
        train_positive_rate=float(train_positive_count / len(train_frame)),
        test_positive_count=test_positive_count,
        test_negative_count=int(len(test_frame) - test_positive_count),
        test_positive_rate=float(test_positive_count / len(test_frame)),
        train_drug_count=int(train_frame["drug_id"].nunique()),
        test_drug_count=int(test_frame["drug_id"].nunique()),
        drug_overlap_count=len(
            set(train_frame["drug_id"]).intersection(test_frame["drug_id"])
        ),
        train_target_count=int(train_frame["target_id"].nunique()),
        test_target_count=int(test_frame["target_id"].nunique()),
        target_overlap_count=len(
            set(train_frame["target_id"]).intersection(test_frame["target_id"])
        ),
        interpretation=POLICY_INTERPRETATIONS[split.policy],
    )


def build_split_assignments(
    table: pd.DataFrame,
    split: SplitIndices,
) -> pd.DataFrame:
    """Return local, row-level assignments for one split policy."""
    if "observed_pair_index" not in table.columns:
        raise SplitDesignError("Interaction table lacks observed_pair_index.")

    _assert_split_invariants(table, split)
    partitions = np.full(len(table), "train", dtype=object)
    partitions[list(split.test_positions)] = "test"

    return pd.DataFrame(
        {
            "split_policy": split.policy,
            "observed_pair_index": table["observed_pair_index"].to_numpy(),
            "partition": partitions,
        }
    )


def write_split_assignments(
    assignments: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write local ignored split assignments for reproducible model fitting."""
    required_columns = {"split_policy", "observed_pair_index", "partition"}
    missing_columns = required_columns.difference(assignments.columns)
    if missing_columns:
        raise SplitDesignError(
            f"Split assignments are missing columns: {sorted(missing_columns)}"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(destination, index=False)
    return destination


def write_split_audit(
    audits: list[SplitAudit],
    output_path: str | Path,
) -> Path:
    """Write a compact version-controlled leakage audit."""
    if not audits:
        raise SplitDesignError("At least one split audit is required.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cold_drug_n_splits": COLD_DRUG_N_SPLITS,
        "cold_drug_outer_fold": COLD_DRUG_OUTER_FOLD,
        "split_policies": [audit.to_dict() for audit in audits],
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create leakage-auditable Davis train/test splits."
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        default=Path("data/interim/davis_interactions_labeled.csv"),
        help="Local labelled Davis interaction table.",
    )
    parser.add_argument(
        "--reference-label-column",
        default="interaction_kd_le_1000_nM",
        help="Binary label used only to stratify the random-pair split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Requested outer-test fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=20260830,
        help="Fixed random seed for every split policy.",
    )
    parser.add_argument(
        "--assignments-output",
        type=Path,
        default=Path("data/interim/davis_split_assignments.csv"),
        help="Local split-assignment CSV destination.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_split_audit.json"),
        help="Version-controlled leakage-audit JSON destination.",
    )
    args = parser.parse_args(argv)

    try:
        table = pd.read_csv(args.input_table)
        splits = create_all_split_policies(
            table,
            reference_label_column=args.reference_label_column,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        audits = [
            audit_train_test_split(
                table,
                split,
                reference_label_column=args.reference_label_column,
            )
            for split in splits
        ]
        assignments = pd.concat(
            [build_split_assignments(table, split) for split in splits],
            ignore_index=True,
        )
        assignment_path = write_split_assignments(
            assignments,
            args.assignments_output,
        )
        summary_path = write_split_audit(audits, args.summary_output)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"Split construction failed: {error}", file=sys.stderr)
        return 2

    payload = {
        "cold_drug_n_splits": COLD_DRUG_N_SPLITS,
        "cold_drug_outer_fold": COLD_DRUG_OUTER_FOLD,
        "split_policies": [audit.to_dict() for audit in audits],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Split assignments written to: {assignment_path}")
    print(f"Split audit written to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())