# Phase 04 — Evidence Retrieval and Claim Ledger

## Goal
Build an evidence-first research record.

## Deliverables
- Source metadata schema: DOI, authors, venue, year, URL, verification status.
- Claim ledger linking background, method, result, and interpretation claims to evidence.
- Deterministic local ranking baseline (BM25-style token scoring); no external framework required.
- Citation verification statuses: VERIFIED, INCONCLUSIVE, REJECTED.

## Acceptance criteria
1. Every drafted factual claim has a ledger row.
2. Unverified sources cannot support final claims.
3. Evidence reports are exportable as JSON and Markdown.

## Git checkpoint
`feat(project07): add evidence ledger and citation validation`
