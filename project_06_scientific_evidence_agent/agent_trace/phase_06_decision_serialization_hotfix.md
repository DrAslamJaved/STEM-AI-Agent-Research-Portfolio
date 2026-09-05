# Phase 06 — Decision serialization hotfix

## Failure observed

The Phase 6 CLI report builders called `VerificationTrace.decision_dict()`, but
the Phase 5 trace model exposed only `as_dict()`. Consequently, the runtime
verification and citation-audit evaluation commands completed inference and
metric calculation but failed while serializing their compact decision records.

## Corrective action

Add a gold-free `decision_dict()` method to `VerificationTrace`. The method
serializes only claim ID, verdict, confidence, and sentence-specific citation
records. It does not expose SciFact gold evidence fields.

## Regression evidence

The runtime trace test now calls `decision_dict()` and asserts the compact
schema and absence of `evidence` and `cited_doc_ids`. The full CLI evaluation
tests must pass before the corrected Phase 6 commit is pushed.
