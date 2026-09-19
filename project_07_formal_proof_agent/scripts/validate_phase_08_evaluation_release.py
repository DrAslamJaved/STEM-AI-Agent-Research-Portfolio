"""Validate the Phase 08 evaluation-and-release contract before staging."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_release import (  # noqa: E402
    EXPECTED_ARMS,
    build_release_bundle,
    load_attempts_jsonl,
    sha256_file,
    validate_release_manifest,
)


def require(text: str, expected: str, label: str) -> None:
    if expected not in text:
        raise AssertionError(f"missing {label}: {expected}")


def main() -> None:
    config = PROJECT_ROOT / "configs" / "phase_08_evaluation_release.yaml"
    module = PROJECT_ROOT / "src" / "formal_math" / "evaluation_release.py"
    prompt = PROJECT_ROOT / "prompts" / "phase_08_release_protocol.md"
    fixture = PROJECT_ROOT / "data" / "processed" / "phase_08_synthetic_test_tasks.jsonl"
    manifest_path = PROJECT_ROOT / "results" / "phase_08_synthetic_final_manifest.json"
    attempts_path = PROJECT_ROOT / "results" / "phase_08_synthetic_attempts.jsonl"
    bundle_path = PROJECT_ROOT / "results" / "phase_08_synthetic_release.json"
    for path in (config, module, prompt, fixture, manifest_path, attempts_path, bundle_path):
        if not path.is_file():
            raise AssertionError(f"required Phase 08 file is missing: {path}")

    configuration = config.read_text(encoding="utf-8")
    require(configuration, "final_evaluation_split: test", "locked final test split")
    require(configuration, "test_split_locked: true", "test split lock")
    require(configuration, "primary_validator: independent_lean_compilation", "formal validity policy")
    require(configuration, "synthetic_smoke_result_is_final_evidence: false", "synthetic boundary")
    require(configuration, "human_review_required_for_release: true", "human review gate")
    for arm in EXPECTED_ARMS:
        require(configuration, f"arm: {arm}", f"{arm} registration")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_release_manifest(manifest)
    provenance = manifest["provenance"]
    if provenance["task_manifest_sha256"] != sha256_file(fixture):
        raise AssertionError("synthetic task-manifest checksum does not match its manifest")
    if provenance["candidate_records_sha256"] != sha256_file(attempts_path):
        raise AssertionError("synthetic candidate-record checksum does not match its manifest")
    attempts = load_attempts_jsonl(attempts_path)
    rebuilt = build_release_bundle(manifest, attempts)
    persisted = json.loads(bundle_path.read_text(encoding="utf-8"))
    if persisted != rebuilt:
        raise AssertionError("persisted Phase 08 synthetic bundle is not reproducible")
    if persisted["release_gate"]["release_ready"]:
        raise AssertionError("a synthetic fixture must never satisfy the release gate")
    if not persisted["synthetic_demo"]:
        raise AssertionError("synthetic fixture must retain its boundary label")
    print("PASS: Phase 08 controlled-evaluation and release contract is locked and valid")


if __name__ == "__main__":
    main()
