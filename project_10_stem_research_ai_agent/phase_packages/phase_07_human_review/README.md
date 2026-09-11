# Phase 07 — Human Review and Revision Workflow

## Goal
Keep the researcher accountable and in control.

## Deliverables
- Review actions: approve, reject, request revision, lock, and supersede.
- Append-only version log with author, timestamp, reason, and affected claim IDs.
- CLI review workflow and diff-friendly manuscript snapshots.
- Exportable approval report.

## Acceptance criteria
1. Final export is blocked until required sections are approved.
2. Locked text cannot change without an explicit supersede action.
3. Revision history is reproducible from stored events.

## Git checkpoint
`feat(project07): implement human approval and revision controls`
