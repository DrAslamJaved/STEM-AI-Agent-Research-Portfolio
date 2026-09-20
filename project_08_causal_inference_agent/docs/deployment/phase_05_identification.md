# Phase 05 deployment: estimand identification and rejection

## Objective

Add a conservative identification gate that derives observed back-door
estimands and rejects invalid, unsupported, under-specified, or non-identifiable
questions before any estimator is invoked.

## Scope

This phase adds exactly five files:

- `src/causal_audit_agent/identification.py`;
- `tests/test_identification.py`;
- `configs/identification/examples.yaml`;
- `docs/identification_protocol.md`; and
- `docs/deployment/phase_05_identification.md`.

No existing tracked file is modified.

## Safety properties

- Graph-audit errors block identification.
- Missing assumptions produce an explicit insufficient-assumptions state.
- Latent confounding without an observed blocking set is rejected.
- Unsupported direct effects and unknown estimands are not reinterpreted.
- CATE conditioning variables receive separate validity checks.
- Adjustment-set search is bounded to prevent accidental exponential work.
- Every identification claim remains conditional on human-approved assumptions.

## Validation gates

Run from `project_08_causal_inference_agent`:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q

& .\.venv\Scripts\python.exe -m pytest `
    --cov=causal_audit_agent `
    --cov-branch `
    --cov-report=term-missing
```

Deployment is acceptable only when the complete Phase 01-05 suite passes, the
new identification module has full line and branch coverage, overall effective
coverage remains at least 98%, exactly five approved files are introduced, and
the staged whitespace check passes.

## Explicit exclusions

Phase 05 does not estimate effects, fit propensity models, invoke refuters,
perform sensitivity analysis, or claim that graphical assumptions are true.
