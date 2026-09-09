"""Immutable file-provenance records for research inputs."""

from hashlib import sha256
from pathlib import Path

def build_file_provenance(path: str | Path, *, chunk_size: int = 1024 * 1024) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")
    digest = sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return {"path": file_path.as_posix(), "filename": file_path.name, "bytes": file_path.stat().st_size,
            "sha256": digest.hexdigest(), "algorithm": "SHA-256"}
