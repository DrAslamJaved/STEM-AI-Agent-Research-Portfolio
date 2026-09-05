# Phase 05 — Lexical evidence-verification baseline

## What was evaluated

The verifier was trained only on 809 SciFact training claims and public corpus
text. This produced 919 labelled claim/document pairs (370 `SUPPORT`, 194
`CONTRADICT`, and 355 `NO_EVIDENCE`) and 8,426 claim/sentence examples. The
development run used 300 claims and BM25 top-10 candidates. The development
split never entered model fitting.

The report contains two different evaluations:

1. a controlled cited-document stance task, in which the classifier receives a
   claim and an evaluation document; and
2. the end-to-end agent, which sees only a claim plus BM25-retrieved public
   documents, selects sentence citations, and may abstain.

## Reference development result

| Measure | Result |
| --- | ---: |
| Controlled stance pair macro-F1 | 0.4414 |
| End-to-end claim macro-F1 | 0.4005 |
| Sentence evidence F1 | 0.0531 |
| Strict citation correctness F1 | 0.0067 |
| Faithfulness among assertive decisions | 0.0581 |
| Coverage | 0.8600 |
| Unsupported-assertion rate | 0.9419 |
| Runtime latency per claim | 12.1 ms |

These values are not a claim that the agent reduces unsupported scientific
assertions. They show that transparent lexical relation features are inadequate
for the difficult SciFact stance and rationale task, even though the pipeline
is leakage-safe and reproducible.

## Why retain this result

The baseline is essential scientific evidence. It separates a genuinely useful
or audited verifier from one that merely appears plausible. In particular, the
very high unsupported-assertion rate makes a citation audit and calibrated
abstention mechanism necessary; a later system must report both its reduction
in unsupported assertions and the coverage it sacrifices.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m evidence_agent build-index
& .\.venv\Scripts\python.exe -m evidence_agent train-verifier
& .\.venv\Scripts\python.exe -m evidence_agent evaluate-verifier
```

The last command writes a compact `results/verification_dev.json` containing
model metadata, exact runtime settings, controlled stance metrics, end-to-end
metrics, and one auditable decision per claim. Its full candidate and sentence
diagnostic trace is written locally to
`artifacts/verification_dev_trace.json` (ignored by Git); the compact report
records that trace's SHA-256 and schema metadata.
