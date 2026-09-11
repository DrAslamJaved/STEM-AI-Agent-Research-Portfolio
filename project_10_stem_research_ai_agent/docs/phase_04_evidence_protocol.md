# Phase 4 Evidence Protocol

The evidence ledger separates local metadata validation from scholarly
verification. A DOI-shaped value and URL can be marked VERIFIED by this local
contract, but later retrieval phases must still verify that the cited source
actually supports the claim.

A finalizable claim requires: (1) HUMAN_APPROVED or EXPERIMENTALLY_SUPPORTED
claim status, (2) at least one VERIFIED source, and (3) an explicit supporting
link with a written rationale. Claims supported only by INCONCLUSIVE sources
remain non-finalizable.

The ranking function is a deterministic lexical baseline, not semantic search
and not a substitute for systematic literature review.
