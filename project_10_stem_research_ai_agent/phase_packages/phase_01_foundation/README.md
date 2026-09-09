# Phase 01 — Foundation and Research Contract

## Goal
Define the MVP: a human-in-the-loop STEM manuscript assistant for two demonstrations: drug–target interaction (DTI) prediction and cardinality-based fuzzy similarity analysis.

## Deliverables
- `docs/research_contract.md`: purpose, user roles, prohibited fabrications, approval gates.
- `schemas/research_brief.schema.json`: idea, methodology, data description, target outlet, and style profile.
- `src/stem_research_agent/contracts.py`: typed input/output contracts.
- `tests/test_contracts.py`.

## Acceptance criteria
1. Invalid briefs fail with clear errors.
2. Claims are labelled `AGENT_PROPOSED`, `HUMAN_APPROVED`, `EXPERIMENTALLY_SUPPORTED`, or `REJECTED`.
3. No manuscript content can be marked final without human approval.

## Git checkpoint
`feat(project07): establish research contract and input schema`
