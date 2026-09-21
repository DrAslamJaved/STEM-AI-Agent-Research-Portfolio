# Phase 09 agent trace — real study completion

## Objective

Upgrade the completed Project 07 engineering MVP into a real, auditable
miniF2F comparison without promoting synthetic fixtures to research evidence.

## Controls implemented

- Canonical runtime provisioning under the portfolio-root Project 07 folder.
- Exact miniF2F v1 commit, Lean 3.42.1, and mathlib provenance checks.
- Extraction of theorem declarations only; reference proof bodies never enter
  model prompts.
- Identical three-arm task/sample/model/seed controls.
- Closed-batch generation for LLM-only and LLM+SymPy.
- Bounded (two-iteration) Lean diagnostic repair only for the Lean arm.
- Exact fuzzy counterexample context and a separate refutation outcome.
- Immutable JSONL hashes, token/cost/time records, and a human-review release
  gate.

## Evidence boundary

No live provider was called while installing or testing this phase.  Until a
preregistered, human-reviewed locked test run is executed, the project must
continue to report only an implementation result—not comparative proof
performance.
