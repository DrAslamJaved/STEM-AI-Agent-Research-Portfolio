# Phase 05 — Evidence selection and stance-verification baseline

## Decision

Add a reproducible lexical verifier before introducing a larger pretrained NLI
model. The phase creates a clear train/runtime/evaluation boundary, records a
complete decision trace, and establishes the metrics that later citation-audit
work must improve.

## Fixed design

- Training uses only `claims_train.jsonl` and public corpus text.
- A three-way logistic model predicts `SUPPORT`, `CONTRADICT`, or
  `NO_EVIDENCE` from TF-IDF relation features.
- A separate binary logistic model scores claim/sentence evidence relevance.
- Runtime uses BM25 top-10 candidates and fixed thresholds: assertion 0.45,
  sentence 0.50, and at most two citation sentences.
- Development evidence, cited documents, and claim labels load only after all
  runtime decisions freeze.

## Evidence required before commit

1. Unit and CLI tests for the train-only data adapter, artifact round-trip,
   runtime trace, stance macro-F1, evidence F1, citation correctness,
   faithfulness, and unsupported-assertion rate.
2. A locally trained ignored model artifact, an ignored full diagnostic trace,
   and a compact committed development report.
3. Passing tests, coverage, compilation, and whitespace validation.

## Reference outcome

The baseline executes end-to-end but is not sufficient for reliable scientific
assertions. With the recorded fixed settings, its development pair-stance
macro-F1 was 0.4414, end-to-end claim macro-F1 was 0.4005, sentence evidence
F1 was 0.0531, and unsupported-assertion rate was 0.9419 at 0.86 coverage.

This is retained as an honest baseline. It identifies the next research task:
select an abstention and citation-audit policy on the supplied five-fold splits
and compare it with a direct-RAG baseline without concealing the coverage
trade-off.

The committed report retains one citation decision per claim; the complete
candidate and sentence trace remains an ignored local artifact with its
checksum recorded in that report. This prevents generated diagnostics from
inflating repository history while keeping the run auditable.
