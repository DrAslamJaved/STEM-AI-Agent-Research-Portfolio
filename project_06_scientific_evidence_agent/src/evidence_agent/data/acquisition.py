"""Reproducible, safe acquisition of the official SciFact archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_SCIFACT_URL = (
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
)
DEFAULT_ARCHIVE_NAME = "scifact_data.tar.gz"


class AcquisitionError(RuntimeError):
    """Raised when a data archive cannot be acquired or safely extracted."""


@dataclass(frozen=True, slots=True)
class AcquisitionManifest:
    """Provenance captured for one immutable SciFact acquisition."""

    source_url: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    dataset_root: str
    acquired_at_utc: str
    downloaded_this_run: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a local file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    """Atomically download a file with a descriptive user agent."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "scientific-evidence-agent/0.1"})

    temporary_path: Path | None = None
    try:
        # The temporary handle must be closed before ``replace``.  Windows does
        # not permit renaming an open NamedTemporaryFile, unlike POSIX systems.
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=destination.parent, suffix=".partial"
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temporary)

        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract only regular files/directories contained within *destination*."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                target = (destination / member.name).resolve()
                if target != destination and destination not in target.parents:
                    raise AcquisitionError(
                        f"Archive contains an unsafe path: {member.name!r}."
                    )
                if not (member.isdir() or member.isfile()):
                    raise AcquisitionError(
                        "Archive contains an unsupported link or device entry: "
                        f"{member.name!r}."
                    )

            for member in members:
                archive.extract(member, path=destination, filter="data")
    except tarfile.TarError as error:
        raise AcquisitionError(f"Unable to read archive {archive_path}: {error}") from error


def _dataset_root_if_present(output_dir: Path) -> Path | None:
    required = {"corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"}
    candidates = [output_dir]
    candidates.extend(path.parent for path in output_dir.rglob("corpus.jsonl"))
    matches = [candidate for candidate in candidates if required.issubset({p.name for p in candidate.iterdir()})]
    unique_matches = list(dict.fromkeys(path.resolve() for path in matches))
    if len(unique_matches) > 1:
        raise AcquisitionError(
            f"Multiple SciFact dataset roots found under {output_dir}: {unique_matches}."
        )
    return unique_matches[0] if unique_matches else None


def acquire_scifact(output_dir: Path, url: str = DEFAULT_SCIFACT_URL) -> AcquisitionManifest:
    """Download, fingerprint, and safely extract SciFact without overwriting data."""
    output_dir = Path(output_dir)
    archive_path = output_dir / DEFAULT_ARCHIVE_NAME
    dataset_root = _dataset_root_if_present(output_dir) if output_dir.exists() else None
    downloaded_this_run = False

    if not archive_path.exists():
        _download_file(url, archive_path)
        downloaded_this_run = True

    if dataset_root is None:
        permitted_existing = {DEFAULT_ARCHIVE_NAME, ".gitkeep"}
        unexpected = [path for path in output_dir.iterdir() if path.name not in permitted_existing]
        if unexpected:
            raise AcquisitionError(
                "Refusing to extract into a non-empty unrecognized directory: "
                f"{output_dir}. Remove or inspect: {unexpected}."
            )
        safe_extract_tar(archive_path, output_dir)
        dataset_root = _dataset_root_if_present(output_dir)

    if dataset_root is None:
        raise AcquisitionError(
            "Archive extraction completed, but the SciFact files could not be located."
        )

    return AcquisitionManifest(
        source_url=url,
        archive_path=str(archive_path),
        archive_sha256=sha256_file(archive_path),
        archive_size_bytes=archive_path.stat().st_size,
        dataset_root=str(dataset_root),
        acquired_at_utc=datetime.now(timezone.utc).isoformat(),
        downloaded_this_run=downloaded_this_run,
    )


def write_acquisition_manifest(manifest: AcquisitionManifest, path: Path) -> None:
    """Write acquisition provenance as stable, human-reviewable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
