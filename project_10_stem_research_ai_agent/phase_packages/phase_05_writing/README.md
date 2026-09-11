# Phase 05 — Controlled Manuscript Drafting

## Goal
Generate a structured draft from approved inputs only.

## Deliverables
- Section templates: title, abstract, introduction, methods, results, discussion, limitations, conclusion.
- Style-profile schema based on researcher-approved preferences, never copied text.
- Draft generator that consumes analysis results and approved evidence ledger.
- Markdown manuscript export with claim identifiers.

## Acceptance criteria
1. Results prose uses only saved analysis values.
2. Unsupported citations and invented numerical values cause generation failure.
3. Every section states its evidence/approval state.

## Git checkpoint
`feat(project07): generate controlled evidence-grounded drafts`
