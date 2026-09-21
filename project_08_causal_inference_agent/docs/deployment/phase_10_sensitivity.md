# Phase 10: Hidden-confounding sensitivity analysis

## Objective

Quantify how strongly an omitted common cause must relate to treatment and
outcome to erase or materially weaken an estimated causal effect.

## Deployment order

Phase 09 must already be merged. Extract this overlay into the repository root,
review the exact file set, run the complete suite, and stage only the five files
listed below.

## Acceptance criteria

- [ ] Partial-R-squared omitted-variable-bias bounds are calculated correctly.
- [ ] Robustness values are reported for additive linear ATE and ATT estimates.
- [ ] E-values are restricted to genuine risk-ratio estimates.
- [ ] Baseline failures and incompatible analyses are rejected deterministically.
- [ ] Confidence intervals crossing the null are marked inconclusive.
- [ ] Reports state that sensitivity results do not prove causal validity.
- [ ] Tests cover every line and branch in the sensitivity module.

## Files in this overlay

- `project_08_causal_inference_agent/configs/sensitivity/default.yaml`
- `project_08_causal_inference_agent/docs/deployment/phase_10_sensitivity.md`
- `project_08_causal_inference_agent/docs/sensitivity_protocol.md`
- `project_08_causal_inference_agent/src/causal_audit_agent/sensitivity.py`
- `project_08_causal_inference_agent/tests/test_sensitivity.py`

## Recommended branch

`feature/project-08-phase-10-hidden-confounding-sensitivity`

## Recommended commit

`feat(project08): complete phase 10 hidden-confounding sensitivity analysis`

## Scope boundary

This phase does not implement Rosenbaum bounds for matched designs, longitudinal
treatment-confounder feedback, instrumental-variable sensitivity, negative
controls, or benchmark-dataset evaluation.
