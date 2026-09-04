# Phase 03 — BM25 retrieval baseline

## Decision

Establish a transparent BM25 baseline before adding embeddings, reranking, or
generation. Its fixed parameters and no-dependency implementation make any
later hybrid retrieval improvement interpretable.

## Leakage controls

- Runtime loaders create only `Claim(claim_id, text)` objects.
- The retriever receives only those claims and the public corpus index.
- Gold evidence document IDs load only after ranked predictions freeze.
- `cited_doc_ids` are never used as relevance labels.

## Evidence required before commit

1. Unit tests for tokenization, ranking, serialization, safe claim loading,
   metric calculation, and CLI workflow.
2. A generated ignored index under `artifacts/`.
3. A committed `results/retrieval_baseline_dev.json` report with Recall@k.
4. Passing tests, coverage, compilation, and whitespace validation.
