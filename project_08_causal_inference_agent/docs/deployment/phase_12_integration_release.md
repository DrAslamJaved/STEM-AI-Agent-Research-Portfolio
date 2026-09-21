# Phase 12: Integration, reporting, CI, and release closure

## Objective

Connect the Phase 1–11 components through a fail-closed orchestrator, generate qualified audit reports, and establish reproducible release gates.

## Prerequisite

Phase 11 and the restored Phase 10 sensitivity layer must be present on synchronized `main`.

## Acceptance criteria

- [ ] Non-causal, ambiguous, invalid-DAG, and non-identifiable requests stop before estimation.
- [ ] Baseline and advanced estimators consume the identified adjustment set.
- [ ] Diagnostics, refutation, and sensitivity results determine the final state without being conflated.
- [ ] IHDP/Lalonde metrics are included only when a benchmark dataset is explicitly attached.
- [ ] Markdown and JSON reports separate all causal workflow stages.
- [ ] Every run contains a deterministic ordered audit trail.
- [ ] Human approval remains mandatory even when automated robustness gates pass.
- [ ] Windows/Linux CI runs Python 3.11 and 3.12 with branch coverage of at least 98%.
- [ ] The complete Phase 1–12 test suite passes from a clean checkout.

## Files in this overlay

- `.github/workflows/project08-ci.yml`
- `project_08_causal_inference_agent/PROJECT_PHASE_INDEX.md`
- `project_08_causal_inference_agent/docs/deployment/phase_12_integration_release.md`
- `project_08_causal_inference_agent/docs/model_card.md`
- `project_08_causal_inference_agent/docs/orchestration_protocol.md`
- `project_08_causal_inference_agent/docs/project_closure_checklist.md`
- `project_08_causal_inference_agent/src/causal_audit_agent/orchestrator.py`
- `project_08_causal_inference_agent/src/causal_audit_agent/reporting.py`
- `project_08_causal_inference_agent/tests/test_orchestration_reporting.py`

## Recommended branch

`feature/project-08-phase-12-integration-release`

## Recommended commit

`feat(project08): complete phase 12 integration and release readiness`
