import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.interaction_table import (
    InteractionTableError,
    build_davis_interaction_table,
    summarize_interaction_table,
    write_interaction_summary,
    write_interaction_table,
)


def write_synthetic_davis(
    data_dir: Path,
    *,
    matrix: np.ndarray | None = None,
    ligands: dict[str, str] | None = None,
    proteins: dict[str, str] | None = None,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "folds").mkdir(exist_ok=True)

    ligands = ligands or {"drug_z": "CCO", "drug_a": "CCC"}
    proteins = proteins or {"target_b": "MKT", "target_a": "MLA"}
    matrix = (
        matrix
        if matrix is not None
        else np.array([[1.0, np.nan], [10.0, 100.0]])
    )

    (data_dir / "ligands_can.txt").write_text(json.dumps(ligands), encoding="utf-8")
    (data_dir / "proteins.txt").write_text(json.dumps(proteins), encoding="utf-8")
    with (data_dir / "Y").open("wb") as handle:
        pickle.dump(matrix, handle)

    (data_dir / "folds" / "train_fold_setting1.txt").write_text(
        json.dumps([[1]]), encoding="utf-8"
    )
    (data_dir / "folds" / "test_fold_setting1.txt").write_text(
        json.dumps([0, 2]), encoding="utf-8"
    )
    return data_dir


def test_interaction_table_preserves_observed_matrix_order(tmp_path: Path) -> None:
    table = build_davis_interaction_table(write_synthetic_davis(tmp_path / "davis"))

    assert table["observed_pair_index"].tolist() == [0, 1, 2]
    assert table["matrix_flat_index"].tolist() == [0, 2, 3]
    assert table["drug_id"].tolist() == ["drug_z", "drug_a", "drug_a"]
    assert table["target_id"].tolist() == ["target_b", "target_b", "target_a"]
    assert table["affinity_kd_nM"].tolist() == [1.0, 10.0, 100.0]
    assert table["pKd"].tolist() == pytest.approx([9.0, 8.0, 7.0])


def test_interaction_table_excludes_missing_affinities(tmp_path: Path) -> None:
    table = build_davis_interaction_table(write_synthetic_davis(tmp_path / "davis"))

    assert len(table) == 3
    assert not table["affinity_kd_nM"].isna().any()


def test_interaction_table_rejects_nonpositive_observed_affinity(
    tmp_path: Path,
) -> None:
    data_dir = write_synthetic_davis(
        tmp_path / "davis",
        matrix=np.array([[0.0, np.nan], [10.0, 100.0]]),
    )

    with pytest.raises(InteractionTableError, match="zero or negative"):
        build_davis_interaction_table(data_dir)


def test_interaction_table_writers_create_reproducible_outputs(tmp_path: Path) -> None:
    table = build_davis_interaction_table(write_synthetic_davis(tmp_path / "davis"))
    summary = summarize_interaction_table(table)

    table_path = write_interaction_table(table, tmp_path / "interactions.csv")
    summary_path = write_interaction_summary(summary, tmp_path / "summary.json")

    reloaded_table = pd.read_csv(table_path)
    reloaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(reloaded_table) == 3
    assert reloaded_summary["row_count"] == 3
    assert reloaded_summary["unique_drug_count"] == 2