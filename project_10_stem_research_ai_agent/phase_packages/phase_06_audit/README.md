# Phase 06 — Quality, Originality, and Consistency Audit

## Goal
Detect defects before human review.

## Deliverables
- Claim support audit and citation consistency checks.
- Numeric consistency check between results JSON, tables, figures, and prose.
- Similarity-risk heuristic report; it reports risk and never guarantees plagiarism-free text.
- Missing-limitations and overclaiming checks.

## Acceptance criteria
1. Audit fails drafts containing uncited factual claims.
2. Changed metrics are detected in stale prose.
3. Every audit finding has severity and a repair recommendation.

## Git checkpoint
`feat(project07): introduce manuscript quality and originality audits`
