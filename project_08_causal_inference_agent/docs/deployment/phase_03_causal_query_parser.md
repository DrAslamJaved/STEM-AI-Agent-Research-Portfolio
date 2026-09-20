# Phase 03: Causal-Question Parser

## Objective

Conservatively distinguish causal, predictive, associational, and ambiguous
research questions without converting prediction or association into causation.

## Deployment order

Phase 02 must already be merged into `main`.

Extract this archive into the repository root. Review `git status` and `git diff`
before staging. Do not commit generated datasets, local environments, caches, or
credentials.

## Safety policy

- Only one active cue family permits a non-ambiguous classification.
- Mixed causal, predictive, or associational cues force abstention.
- Underspecified and blank questions force abstention.
- Matched cue codes remain visible for audit.
- Causal classifications require human review.
- A routing label never establishes identification or causal validity.

## Acceptance criteria

- [ ] Canonical intervention language is classified as causal.
- [ ] Forecasting and risk language is classified as predictive.
- [ ] Correlation and relationship language is classified as associational.
- [ ] Mixed-intent and underspecified questions are classified as ambiguous.
- [ ] Parsed causal questions remain subject to human approval.
- [ ] Non-string inputs fail explicitly.
- [ ] The routing score is documented as non-probabilistic.

## Files in this overlay

- `project_08_causal_inference_agent/docs/causal_question_protocol.md`
- `project_08_causal_inference_agent/docs/deployment/phase_03_causal_query_parser.md`
- `project_08_causal_inference_agent/src/causal_audit_agent/query_parser.py`
- `project_08_causal_inference_agent/tests/test_query_parser.py`

## Required validation

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m pytest `
    --cov=causal_audit_agent `
    --cov-branch `
    --cov-report=term-missing `
    --cov-fail-under=98
```

## Recommended branch

`feature/project-08-phase-03-causal-question-parser`

## Recommended commit

`feat(project08): complete phase 03 causal-question parser`
