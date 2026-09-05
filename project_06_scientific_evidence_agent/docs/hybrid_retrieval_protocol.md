# Phase 04 hybrid retrieval and reranking protocol

## Objective

Compare a fixed hybrid candidate generator against the committed BM25
development baseline under an identical corpus, claim file, and final top-*k*
budget. The result is a retrieval comparison, not yet a claim-verification
comparison.

## Fixed pipeline

1. Build a public-corpus TF-IDF matrix using unigram and bigram features.
2. Fit 128-dimensional truncated SVD with `random_seed = 20260904`.
3. Retrieve the top 50 candidates independently from BM25 and latent-semantic
   search.
4. Fuse both ranks with reciprocal-rank fusion, using rank constant 60.
5. Rerank the fused candidate set using fixed weights: RRF 0.45, semantic score
   0.35, BM25 score 0.15, and title-term coverage 0.05.
6. Freeze the final rank list before reading evaluator-only evidence or the
   committed BM25 metrics.

The reranker is intentionally transparent rather than a cross-encoder. It is a
reproducible intermediate system for isolating the value of hybrid retrieval.

## Leakage and comparison controls

- The semantic index receives corpus title and abstract text only.
- Runtime query objects remain `Claim(id, text)` with no gold fields.
- The evaluator loads SciFact `evidence` only after hybrid rankings freeze.
- Both BM25 and semantic artifacts must match the corpus SHA-256.
- The report retains the SHA-256 of the committed BM25 report and calculates
  deltas only after hybrid evaluation.

## Decision rule

Report both positive and negative deltas for Claim Recall@k,
evidence-document Recall@k, and MRR. A gain in retrieval does not by itself
establish a reduction in unsupported scientific claims; that requires the later
evidence-selection, NLI, and citation-audit phases.

## Observed Phase 04 outcome

The frozen development evaluation produced negative deltas against BM25 across
every reported retrieval measure. The full machine-readable output is retained
as `results/hybrid_retrieval_dev.json`, and the human-readable comparison is
in `reports/phase_04_hybrid_retrieval.md`.

This result does not establish that all semantic retrieval is harmful. It does
show that this particular corpus-only TF-IDF/LSA + heuristic-reranking
configuration should not be represented as an improvement. BM25 is retained as
the candidate generator for the next phase. No post-evaluation adjustment of
the recorded hybrid weights is made from this development result.
