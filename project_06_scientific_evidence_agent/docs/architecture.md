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

## Phase 05 evidence selection and stance verification

Phase 05 fits two separate deterministic lexical models from `claims_train`
only. The first uses claim/document TF-IDF relation features (claim vector,
document vector, element-wise overlap, and absolute difference) with logistic
regression to predict `SUPPORT`, `CONTRADICT`, or `NO_EVIDENCE`. The second
uses the same relation representation on claim/sentence pairs to assign an
evidence probability.

At runtime, a safe `Claim(id, text)` is passed through BM25 top-10 retrieval.
The verifier scores only those public-corpus documents and sentences, chooses
one cited document only when both stance and sentence confidence clear fixed
thresholds, otherwise abstains. The complete runtime trace freezes to an
ignored local artifact before the evaluator reads development `evidence` or
`cited_doc_ids`; the committed report retains only a compact claim-level audit
record and the trace checksum. This makes the controlled cited-document stance
benchmark and the end-to-end evidence audit separate, auditable measurements
without committing a large generated trace.

## Phase 06 cross-validated citation audit

The supplied SciFact cross-validation files partition the union of ordinary
training and ordinary development claims. Therefore Phase 06 does **not** train
on `claims_train_i.jsonl` directly. It takes each supplied `claims_dev_i` ID
assignment, retains only IDs already in `claims_train.jsonl`, and trains on the
complement within that same ordinary training split. This preserves
`claims_dev.jsonl` for one final evaluation.

Each fold freezes a complete BM25/verifier candidate trace with zero thresholds
before any fold labels are loaded for policy scoring. A policy specifies the
minimum assertion confidence, minimum sentence-evidence confidence, and a
maximum number of citation sentences. The selected policy minimizes pooled
out-of-fold unsupported-assertion rate subject to at least 20% assertion
coverage. It is then applied, without re-selection, to a final model trained on
the full ordinary training split and to the untouched ordinary development
claims. Both the selected and Phase 05 policies are applied to the same frozen
development trace for a fair coverage-aware comparison.
