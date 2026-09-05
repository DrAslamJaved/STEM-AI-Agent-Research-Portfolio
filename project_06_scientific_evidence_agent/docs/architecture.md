# Architecture boundary

```text
claim text
  -> retrieval
  -> reranking
  -> sentence evidence selection
  -> stance verification
  -> citation audit
  -> support / contradict / no-evidence decision
```

Runtime code receives claim text and the public corpus only. Evaluation code
receives frozen predictions and gold annotations separately. This separation is
the primary control against accidental data leakage.

## Phase 03 baseline

The first retriever is a deterministic BM25 index over each document title and
abstract. It is intentionally dependency-free and has fixed `k1=1.2` and
`b=0.75` parameters. Evaluation generates and freezes ranked document IDs from
safe `Claim(id, text)` objects before the evaluator reads SciFact `evidence`.
This establishes a transparent lexical baseline for later hybrid retrieval and
reranking experiments.

## Phase 04 hybrid retrieval and reranking

The hybrid retriever independently produces a BM25 rank list and an
unsupervised TF-IDF + truncated-SVD latent-semantic rank list. Reciprocal-rank
fusion forms a fixed candidate set. A transparent reranker then combines the
two normalized scores, the RRF score, and title-term coverage. It does not use
SciFact evidence, labels, cited documents, or a trained cross-encoder.

The evaluator receives only the final frozen ranked IDs and scores. It reads
gold evidence and the previously committed BM25 report only afterward, allowing
the hybrid-versus-baseline deltas to be audited independently.

The fixed Phase 04 hybrid is retained as a diagnostic comparator rather than
used downstream: its committed report shows lower retrieval performance than
BM25. Phase 05 consequently takes its candidate documents from the BM25 index
while preserving the hybrid implementation and report for reproducibility.
