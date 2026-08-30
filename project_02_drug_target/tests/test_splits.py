import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import (
    SplitDesignError,
    audit_train_test_split,
    build_split_assignments,
    create_all_split_policies,
    create_train_test_split,
    write_split_assignments,
    write_split_audit,
)


LABEL_COLUMN = "interaction_kd_le_1000_nM"


def synthetic_labeled_interactions() -> pd.DataFrame:
    rows = []

    for drug_index in range(8):
        for target_index in range(6):
            rows.append(
                {
                    "observed_pair_index": len(rows),
                    "drug_id": f"drug_{drug_index}",
                    "target_id": f"target_{target_index}",
                    LABEL_COLUMN: int((drug_index + target_index) % 3 == 0),
                }
            )

    return pd.DataFrame(rows)


def test_random_pair_split_is_reproducible_and_exposes_entity_overlap() -> None:
    table = synthetic_labeled_interactions()

    first = create_train_test_split(
        table,
        "random_pair",
        reference_label_column=LABEL_COLUMN,
        test_size=0.25,
        random_state=7,
    )
    second = create_train_test_split(
        table,
        "random_pair",
        reference_label_column=LABEL_COLUMN,
        test_size=0.25,
        random_state=7,
    )
    audit = audit_train_test_split(
        table,
        first,
        reference_label_column=LABEL_COLUMN,
    )

    assert first == second
    assert audit.train_pair_count + audit.test_pair_count == len(table)
    assert audit.train_positive_count + audit.test_positive_count == int(
        table[LABEL_COLUMN].sum()
    )
    assert audit.drug_overlap_count > 0
    assert audit.target_overlap_count > 0


def test_cold_drug_split_has_no_drug_overlap() -> None:
    table = synthetic_labeled_interactions()

    split = create_train_test_split(
        table,
        "cold_drug",
        reference_label_column=LABEL_COLUMN,
        test_size=0.25,
        random_state=7,
    )
    audit = audit_train_test_split(
        table,
        split,
        reference_label_column=LABEL_COLUMN,
    )

    assert audit.drug_overlap_count == 0
    assert audit.target_overlap_count > 0
    assert split.splitter_name == "StratifiedGroupKFold"
    assert split.fold_index == 4


def test_cold_target_split_has_no_target_overlap() -> None:
    table = synthetic_labeled_interactions()

    split = create_train_test_split(
        table,
        "cold_target",
        reference_label_column=LABEL_COLUMN,
        test_size=0.25,
        random_state=7,
    )
    audit = audit_train_test_split(
        table,
        split,
        reference_label_column=LABEL_COLUMN,
    )

    assert audit.drug_overlap_count > 0
    assert audit.target_overlap_count == 0


def test_split_assignments_and_audit_writers_are_reproducible(
    tmp_path: Path,
) -> None:
    table = synthetic_labeled_interactions()
    splits = create_all_split_policies(
        table,
        reference_label_column=LABEL_COLUMN,
        test_size=0.25,
        random_state=7,
    )
    audits = [
        audit_train_test_split(
            table,
            split,
            reference_label_column=LABEL_COLUMN,
        )
        for split in splits
    ]
    assignments = pd.concat(
        [build_split_assignments(table, split) for split in splits],
        ignore_index=True,
    )

    assignments_path = write_split_assignments(
        assignments,
        tmp_path / "assignments.csv",
    )
    audit_path = write_split_audit(audits, tmp_path / "audit.json")

    reloaded_assignments = pd.read_csv(assignments_path)
    report = json.loads(audit_path.read_text(encoding="utf-8"))

    assert len(reloaded_assignments) == len(table) * 3
    assert set(reloaded_assignments["split_policy"]) == {
        "random_pair",
        "cold_drug",
        "cold_target",
    }
    assert len(report["split_policies"]) == 3
    assert report["cold_drug_n_splits"] == 5
    assert report["cold_drug_outer_fold"] == 4


def test_split_builder_rejects_invalid_policy_and_duplicate_indices() -> None:
    table = synthetic_labeled_interactions()

    with pytest.raises(SplitDesignError, match="Unknown policy"):
        create_train_test_split(
            table,
            "cold_both",
            reference_label_column=LABEL_COLUMN,
        )

    duplicated = table.copy()
    duplicated.loc[1, "observed_pair_index"] = 0

    with pytest.raises(SplitDesignError, match="must be unique"):
        create_train_test_split(
            duplicated,
            "random_pair",
            reference_label_column=LABEL_COLUMN,
        )