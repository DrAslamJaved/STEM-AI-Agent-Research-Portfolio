"""Tests for source fingerprinting and safe archive extraction."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from evidence_agent.data import acquisition
from evidence_agent.data.acquisition import (
    AcquisitionError,
    DEFAULT_ARCHIVE_NAME,
    _download_file,
    acquire_scifact,
    safe_extract_tar,
    sha256_file,
    write_acquisition_manifest,
)


def _write_tar(path: Path, member_name: str, content: bytes = b"content") -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))


def _write_scifact_archive(path: Path) -> None:
    records = {
        "data/corpus.jsonl": [
            {
                "doc_id": 10,
                "title": "Document one",
                "abstract": ["Sentence zero."],
                "structured": False,
            }
        ],
        "data/claims_train.jsonl": [
            {"id": 1, "claim": "A claim.", "evidence": {}, "cited_doc_ids": [10]}
        ],
        "data/claims_dev.jsonl": [
            {"id": 2, "claim": "A claim.", "evidence": {}, "cited_doc_ids": [10]}
        ],
        "data/claims_test.jsonl": [
            {"id": 3, "claim": "A claim.", "evidence": {}, "cited_doc_ids": [10]}
        ],
    }
    with tarfile.open(path, mode="w:gz") as archive:
        for member_name, rows in records.items():
            payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("scientific evidence", encoding="utf-8")

    assert sha256_file(payload) == "fd97642e0d463a4460d872b2a3cb336685354e096bcc2d178aff820e37c3c9da"


def test_download_closes_temporary_file_before_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    class TemporaryFile(io.BytesIO):
        def __init__(self, path: Path) -> None:
            super().__init__()
            self.name = str(path)
            self.closed_before_replace = False

        def __enter__(self) -> "TemporaryFile":
            return self

        def __exit__(self, *args: object) -> None:
            self.closed_before_replace = True
            self.close()

    temporary_path = tmp_path / "download.partial"
    temporary = TemporaryFile(temporary_path)

    def fake_named_temporary_file(**_: object) -> TemporaryFile:
        return temporary

    def fake_replace(path: Path, destination: Path) -> Path:
        assert temporary.closed_before_replace is True
        destination.write_bytes(b"downloaded archive")
        return destination

    monkeypatch.setattr(acquisition.tempfile, "NamedTemporaryFile", fake_named_temporary_file)
    monkeypatch.setattr(acquisition, "urlopen", lambda *args, **kwargs: Response(b"payload"))
    monkeypatch.setattr(Path, "replace", fake_replace)

    destination = tmp_path / "archive.tar.gz"
    _download_file("https://example.org/archive.tar.gz", destination)

    assert destination.read_bytes() == b"downloaded archive"


def test_safe_extract_tar_extracts_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "safe.tar.gz"
    destination = tmp_path / "extracted"
    _write_tar(archive, "data/corpus.jsonl")

    safe_extract_tar(archive, destination)

    assert (destination / "data" / "corpus.jsonl").read_bytes() == b"content"


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_tar(archive, "../../outside.txt")

    with pytest.raises(AcquisitionError, match="unsafe path"):
        safe_extract_tar(archive, tmp_path / "extracted")


def test_acquire_scifact_extracts_existing_archive_and_writes_provenance(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    _write_scifact_archive(output_dir / DEFAULT_ARCHIVE_NAME)

    manifest = acquire_scifact(output_dir)
    provenance_path = tmp_path / "validation" / "acquisition.json"
    write_acquisition_manifest(manifest, provenance_path)

    assert manifest.downloaded_this_run is False
    assert Path(manifest.dataset_root).name == "data"
    assert len(manifest.archive_sha256) == 64
    assert json.loads(provenance_path.read_text(encoding="utf-8"))["source_url"].startswith(
        "https://"
    )
