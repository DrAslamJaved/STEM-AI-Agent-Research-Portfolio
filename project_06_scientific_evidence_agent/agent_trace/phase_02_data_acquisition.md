# Phase 02 — SciFact acquisition and validation

## Decision

Acquire the canonical SciFact release as an immutable archive, fingerprint it
with SHA-256, and validate its full JSONL structure before model development.

## Controls

- Safe tar extraction rejects path traversal, links, and device entries.
- Raw data are ignored by Git; only provenance and validation artifacts are
  committed.
- Validation inspects gold evidence only on the evaluator side of the boundary.
- Five-fold split layout is checked before it is used for selection or tuning.

## Expected evidence before commit

1. `validation/scifact_acquisition.json` with the release SHA-256.
2. `validation/scifact_validation.json` with structural counts and labels.
3. Passing tests and CLI validation against the acquired data.
