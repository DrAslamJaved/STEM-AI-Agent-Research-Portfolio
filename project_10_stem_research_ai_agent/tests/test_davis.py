import json
import pickle

import pytest

from stem_research_agent.davis import affinity_nm_to_pkd, davis_records, load_davis_dataset


def _write_source(tmp_path, matrix=((10.0, 100.0), (1.0, 1000.0))):
    (tmp_path / "ligands_can.txt").write_text(json.dumps({"D1": "CCO", "D2": "CCC"}), encoding="utf-8")
    (tmp_path / "proteins.txt").write_text(json.dumps({"T1": "MAAA", "T2": "MBBB"}), encoding="utf-8")
    with (tmp_path / "Y").open("wb") as handle:
        pickle.dump(matrix, handle)


def test_loader_produces_a_valid_source_report_and_stable_record_order(tmp_path):
    _write_source(tmp_path)
    dataset = load_davis_dataset(tmp_path)
    records = davis_records(dataset)
    assert dataset.source_report.is_valid
    assert dataset.source_report.record_count == 4
    assert len(dataset.source_report.sha256["Y"]) == 64
    assert [(row["drug_id"], row["target_id"]) for row in records] == [
        ("D1", "T1"), ("D1", "T2"), ("D2", "T1"), ("D2", "T2")]
    assert records[0]["pkd"] == pytest.approx(8.0)
    assert records[1]["label"] == 1
    assert records[-1]["label"] == 0


def test_loader_rejects_matrix_dimension_mismatch(tmp_path):
    _write_source(tmp_path, matrix=((10.0,), (1.0,)))
    with pytest.raises(ValueError, match="Invalid Davis dimensions"):
        load_davis_dataset(tmp_path)


def test_loader_rejects_nonpositive_affinity(tmp_path):
    _write_source(tmp_path, matrix=((0.0, 100.0), (1.0, 1000.0)))
    with pytest.raises(ValueError, match="strictly positive"):
        load_davis_dataset(tmp_path)


def test_pkd_conversion_requires_positive_nanomoar_affinity():
    assert affinity_nm_to_pkd(100.0) == pytest.approx(7.0)
    with pytest.raises(ValueError):
        affinity_nm_to_pkd(0.0)
