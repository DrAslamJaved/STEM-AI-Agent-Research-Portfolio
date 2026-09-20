# Phase 09 deployment: refutation and stress testing

## Objective

Add deterministic placebo-treatment, random-common-cause, and stratified-
subset stress tests downstream of successful identification and estimation.

## Prerequisite

Phase 08 must already be merged, synchronized, validated, and clean.

## Scope

This overlay adds exactly five files:

- `src/causal_audit_agent/refutation.py`;
- `tests/test_refutation.py`;
- `configs/refutation/default.yaml`;
- `docs/refutation_protocol.md`; and
- `docs/deployment/phase_09_refutation.md`.

No existing tracked file is modified.

## Acceptance criteria

- every refuter reuses the approved identification and estimator contract;
- blocked baseline estimates cannot enter the refutation suite;
- failed simulations and instability thresholds are explicit;
- seeded executions are reproducible;
- reports are JSON-compatible and always require human review;
- a pass is never described as validation of causal assumptions;
- the complete Phase 01-09 suite passes;
- `refutation.py` has full line and branch coverage;
- overall effective coverage remains at least 98%; and
- exactly five approved files are introduced.

## Explicit exclusions

Phase 09 does not implement calibrated hidden-confounding sensitivity bounds,
bootstrap interval coverage, benchmark datasets, DoWhy orchestration, or
automated changes to the target population.

## Deployment metadata

- Branch: `feature/project-08-phase-09-refutation`
- Commit: `feat(project08): complete phase 09 causal refutation and stress testing`
