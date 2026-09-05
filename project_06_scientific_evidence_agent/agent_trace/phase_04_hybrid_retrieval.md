# Phase 04 — Hybrid retrieval and transparent reranking

## Decision

Add a fixed corpus-only latent-semantic retriever to the committed BM25
baseline, then combine their ranks with RRF and a fully recorded candidate
reranker.

## Rationale

BM25 is strong for exact scientific terminology but can miss paraphrases and
related terminology. Latent semantic indexing can recover some conceptual
similarity without downloading an opaque language model or training on gold
SciFact annotations.

## Controls

- TF-IDF/LSA fitting consumes only public corpus title and abstract text.
- Both indexes are matched against the corpus SHA-256 before evaluation.
- Final candidate ranks freeze before gold evidence or baseline metrics load.
- The reranker is documented as a deterministic heuristic, not a learned
  cross-encoder.

## Evidence required before commit

1. Tests for semantic indexing, persisted artifact round trips, deterministic
   hybrid ranking, corpus-match checks, and the CLI workflow.
2. Ignored local BM25 and LSA artifacts.
3. A committed `results/hybrid_retrieval_dev.json` report with deltas against
   the committed BM25 development report.
4. Passing tests, coverage, compilation, and whitespace validation.

## Frozen outcome

The fixed development run was retained as an auditable negative control. It
reduced Claim Recall@10 by 0.1489, evidence-document Recall@10 by 0.1435, and
MRR by 0.2295 relative to BM25. The complete result is recorded in
`results/hybrid_retrieval_dev.json`; the readable comparison is in
`reports/phase_04_hybrid_retrieval.md`.

## Next decision

Do not tune the Phase 04 heuristic on this frozen development result. Retain
BM25 as the candidate generator for Phase 05 sentence selection and stance
verification. Any later learned retrieval alternative must use the five-fold
splits for selection before a new frozen development evaluation.
