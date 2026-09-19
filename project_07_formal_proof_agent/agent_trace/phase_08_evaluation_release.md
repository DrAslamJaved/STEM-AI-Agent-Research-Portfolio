# Phase 08 agent trace

## Scope

Added the final controlled-evaluation and release-gate layer for Project 07.
This overlay is specific to `project_07_formal_proof_agent`; it contains no
Project 10 material.

## Controls locked

- Frozen miniF2F v1 provenance and Lean 3.42.1/mathlib revisions are retained.
- Final aggregation accepts only the locked `test` split.
- The three registered arms share a comparison contract and bounded Lean repair.
- Independent Lean compilation remains the formal-validity decision.
- The release gate rejects synthetic smoke evidence and requires human approval.

## Evidence boundary

The committed fixture uses deterministic synthetic candidate records so that
the pipeline can be tested without making a model-performance claim. It is not
a miniF2F test result, it does not modify the frozen benchmark, and it does not
replace a future controlled final run.

## Verification expected

1. Run the Phase 08 unit tests.
2. Run the synthetic release fixture.
3. Rebuild and validate the evidence bundle.
4. Run the Project 07 GitHub Actions workflow on the pull request.
