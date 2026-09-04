"""CLI behaviour tests for the staged evidence-agent workflow."""

from __future__ import annotations

import json
from pathlib import Path

from evidence_agent.cli import main
from evidence_agent.data.acquisition import AcquisitionManifest
from tests.helpers import write_minimal_scifact_dataset


def test_contract_command_prints_machine_readable_project_contract(capsys) -> None:
    assert main(["contract"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["current_phase"] == "retrieval_baseline"
    assert payload["runtime_gold_fields_forbidden"] == ["evidence", "cited_doc_ids"]


def test_cli_without_command_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "Scientific Evidence Verification" in capsys.readouterr().out


def test_acquire_data_command_writes_manifest(monkeypatch, tmp_path: Path, capsys) -> None:
    manifest = AcquisitionManifest(
        source_url="https://example.test/scifact.tar.gz",
        archive_path="data/raw/scifact/scifact_data.tar.gz",
        archive_sha256="a" * 64,
        archive_size_bytes=10,
        dataset_root="data/raw/scifact/data",
        acquired_at_utc="2026-09-04T00:00:00+00:00",
        downloaded_this_run=False,
    )
    monkeypatch.setattr("evidence_agent.cli.acquire_scifact", lambda **_: manifest)
    provenance_path = tmp_path / "acquisition.json"

    assert main(["acquire-data", "--provenance-path", str(provenance_path)]) == 0

    assert json.loads(capsys.readouterr().out)["archive_sha256"] == "a" * 64
    assert json.loads(provenance_path.read_text(encoding="utf-8"))["archive_size_bytes"] == 10


def test_validate_data_command_writes_a_report(tmp_path: Path, capsys) -> None:
    data_dir = write_minimal_scifact_dataset(tmp_path / "scifact")
    report_path = tmp_path / "validation.json"

    assert (
        main(
            [
                "validate-data",
                "--data-dir",
                str(data_dir),
                "--report-path",
                str(report_path),
                "--skip-cross-validation",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["claims_by_split"] == {"dev": 1, "test": 1, "train": 1}
    assert report_path.exists()


def test_cli_builds_and_evaluates_bm25_retrieval_baseline(
    tmp_path: Path, capsys
) -> None:
    dataset = write_minimal_scifact_dataset(tmp_path / "scifact")
    index_path = tmp_path / "artifacts" / "index.json"
    report_path = tmp_path / "results" / "retrieval.json"

    assert (
        main(
            [
                "build-index",
                "--corpus-path",
                str(dataset / "corpus.jsonl"),
                "--index-path",
                str(index_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["document_count"] == 2

    assert (
        main(
            [
                "evaluate-retrieval",
                "--claims-path",
                str(dataset / "claims_dev.jsonl"),
                "--index-path",
                str(index_path),
                "--report-path",
                str(report_path),
                "--cutoffs",
                "1",
                "3",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["cutoffs"] == [1, 3]
    assert report["schema_version"] == "evidence_agent_retrieval_report_v1"
    assert len(report["predictions"]) == 1


def test_future_command_fails_clearly_until_its_phase_is_implemented(capsys) -> None:
    assert main(["evaluate"]) == 2
    assert "not available yet" in capsys.readouterr().err
