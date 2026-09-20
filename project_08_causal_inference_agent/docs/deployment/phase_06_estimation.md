# Phase 06 deployment: causal effect estimation baselines

## Objective

Add identification-gated, reproducible baseline estimators with explicit
overlap diagnostics, approximate uncertainty, and refusal behavior.

## Scope

This phase adds exactly five files:

- `src/causal_audit_agent/estimation.py`;
- `tests/test_estimation.py`;
- `configs/estimation/baselines.yaml`;
- `docs/estimation_protocol.md`; and
- `docs/deployment/phase_06_estimation.md`.

No existing tracked file is modified.

## Validation gates

Run from `project_08_causal_inference_agent`:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q

& .\.venv\Scripts\python.exe -m pytest `
    --cov=causal_audit_agent `
    --cov-branch `
    --cov-report=term-missing
```

Deployment is acceptable only when the complete Phase 01-06 suite passes, the
new estimation module has full line and branch coverage, overall effective
coverage remains at least 98%, exactly five approved files are introduced, and
the staged whitespace check passes.

## Explicit exclusions

Phase 06 does not implement causal forests, cross-fitted learners, bootstrap
coverage studies, refuters, hidden-confounding sensitivity analysis, benchmark
datasets, or DoWhy orchestration. These remain later-phase responsibilities.
