# Phase 08 deployment: overlap, balance, and weight diagnostics

## Objective

Prevent practically non-estimable analyses from being reported as reliable
causal results by applying deterministic empirical diagnostic gates.

## Prerequisite

Phase 07 must already be merged, synchronized, tested, and clean.

## Scope

This overlay adds exactly four files:

- `src/causal_audit_agent/diagnostics.py`;
- `tests/test_diagnostics.py`;
- `docs/diagnostic_policy.md`; and
- `docs/deployment/phase_08_diagnostics.md`.

No existing tracked file is modified. Generated data, environments, caches,
credentials, and analysis outputs are excluded.

## Acceptance criteria

- unweighted and causal-weighted standardized mean differences are reported;
- overlap fraction and propensity clipping are reported;
- maximum and extreme-weight fraction are reported;
- total effective sample size and its sample fraction are reported;
- ATE and ATT weighting targets are distinguished;
- invalid inputs fail explicitly;
- all thresholds are configurable and validated;
- a deterministic `PASS` or `REVIEW` status includes machine-readable reasons;
- the complete Phase 01-08 test suite passes;
- the new module has full line and branch coverage;
- overall effective coverage remains at least 98%; and
- the staged whitespace check passes.

## Explicit exclusions

Phase 08 does not implement generalized causal forests, bootstrap coverage
studies, refuters, hidden-confounding sensitivity analysis, benchmark datasets,
or DoWhy orchestration.

## Deployment metadata

- Branch: `feature/project-08-phase-08-diagnostics`
- Commit: `feat(project08): complete phase 08 overlap, balance, and weight diagnostics`
