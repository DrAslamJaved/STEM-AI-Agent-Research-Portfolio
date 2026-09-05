# Phase 03 retrieval protocol

## Baseline

The baseline uses deterministic Okapi BM25 over the public SciFact document
title followed by its abstract sentences. It uses no language model, reranker,
or external corpus. Fixed parameters are `k1 = 1.2` and `b = 0.75`.

## Split and tuning policy

The development claim file is the default retrieval evaluation split. The
hidden test split is not used for parameter selection or local comparisons.
No BM25 parameter search is performed in this phase; the fixed baseline serves
as the preregistered comparison point for the later hybrid retriever.

## Leakage-safe execution order

1. Load only `id` and `claim` into runtime `Claim` objects.
2. Retrieve and freeze ranked document IDs and scores.
3. Load evaluator-only `evidence` document IDs.
4. Compute Recall@k from the frozen predictions.

`cited_doc_ids` are not used as relevance labels. The relevant documents are
the keys of SciFact's gold `evidence` object.

## Reported measures

For each cutoff in `{1, 3, 5, 10}`, the report records:

- **Claim Recall@k**: fraction of gold-bearing claims with at least one gold
  evidence document in the top *k*.
- **Evidence-document Recall@k**: micro-recall over all gold evidence document
  instances across gold-bearing claims.
- **MRR**: reciprocal rank of the first gold evidence document, averaged over
  gold-bearing claims.

The report retains ranked IDs and scores, source hashes, fixed parameters, and
the claim-file hash so it can be independently audited without storing raw
SciFact data in Git.
