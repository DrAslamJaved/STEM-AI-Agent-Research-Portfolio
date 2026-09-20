# Phase 07 deployment: cross-fitted and heterogeneous-effect estimation

## Objective

Add identification-gated cross-fitted AIPW for ATE and a forest-based doubly
robust learner for CATE without overstating the forest method or its
uncertainty.

## Prerequisite

Phase 06 must already be merged, synchronized, validated, and clean.

## Scope

This overlay adds exactly five files:

- `src/causal_audit_agent/advanced_estimation.py`;
- `tests/test_advanced_estimation.py`;
- `configs/estimation/advanced.yaml`;
- `docs/advanced_estimation_protocol.md`; and
- `docs/deployment/phase_07_advanced_estimation.md`.

No existing tracked file is modified.

## Acceptance criteria

- cross-fitted AIPW produces out-of-fold nuisance predictions and an ATE;
- the forest DR-learner produces out-of-fold heterogeneous-effect predictions;
- identification and adjustment-set gates cannot be bypassed;
- invalid requests, invalid data, and overlap failures are explicit;
- CATE pointwise uncertainty is not fabricated;
- outputs are deterministic under a fixed random seed and JSON-compatible;
- the complete Phase 01-07 suite passes;
- `advanced_estimation.py` has full line and branch coverage;
- overall effective coverage remains at least 98%; and
- exactly five approved files are introduced.

## Explicit exclusions

Phase 07 does not implement generalized random forests, EconML
`CausalForestDML`, bootstrap intervals, PEHE evaluation, benchmark datasets,
refuters, hidden-confounding sensitivity analysis, diagnostics orchestration,
or DoWhy orchestration.

## Deployment metadata

- Branch: `feature/project-08-phase-07-advanced-estimation`
- Commit: `feat(project08): complete phase 07 advanced causal effect estimation`
