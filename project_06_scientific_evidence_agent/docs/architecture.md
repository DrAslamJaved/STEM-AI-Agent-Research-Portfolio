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
