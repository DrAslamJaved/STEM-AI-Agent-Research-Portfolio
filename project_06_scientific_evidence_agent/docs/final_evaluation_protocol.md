# Phase 07 final-evaluation protocol

## Purpose

`evaluate --config configs/final.yaml` reproduces the Phase 06 selected-policy
comparison as a self-contained, config-driven, hash-verified run, and adds
deterministic paired-bootstrap confidence intervals around the comparison.
It is a **held-out development evaluation, not an independent test**: it
reuses `claims_dev.jsonl`, the same split Phase 06 already reported against.

## What this command does not do

`evaluate` only *loads* prebuilt, fixed artifacts. It never rebuilds the BM25
index, retrains the lexical verifier, or recalibrates the citation-audit
policy grid. If any of those artifacts are missing, run their normal Phase 03,
Phase 05, or Phase 06 commands first (`build-index`, `train-verifier`,
`calibrate-citation-audit`); `evaluate` will refuse to run against artifacts
whose SHA-256 does not match what `configs/final.yaml` declares.

## Configuration contract

`configs/final.yaml` is loaded with `yaml.safe_load` and schema-validated
before anything else happens. Every path is resolved relative to the
directory containing `final.yaml` (i.e. `configs/`), never the process
working directory, so the same config runs identically regardless of the
caller's current directory. The config declares:

- `artifacts.{corpus,claims_dev,bm25_index,verifier_model,calibration_report}`
  as `{path, sha256}` pairs -- all five are required and hash-checked before
  any file is loaded. `artifacts.train_claims` is optional additional
  provenance: when present, it is cross-checked against the verifier bundle's
  own recorded training-claims hash and against the calibration report's
  training-split provenance, exactly as `evaluate-citation-audit` already
  does.
- `runtime.retrieval_k`: BM25 documents supplied to the frozen verifier.
- `bootstrap.{enabled,resamples,seed,confidence_level}`: deterministic
  paired-bootstrap settings.
- `output.{result_path,trace_path,report_path,agent_trace_path}`: where the
  Phase 07 result JSON, raw runtime trace, narrative report, and agent trace
  will live. Loading the config raises immediately if any output path would
  resolve to `results/citation_audit_dev.json` or
  `results/citation_audit_cross_validation.json`.

## Ordering guarantee

Exactly as in Phase 06: the frozen verifier is run once at zero thresholds,
the complete raw trace is written to `output.trace_path` and its SHA-256
recorded, and only *then* is `load_gold_claim_annotations` called (this is
where `evidence` / `cited_doc_ids` enter). Both the fixed Phase 05 policy and
the Phase 06 selected policy are applied to that one already-frozen trace, in
the identical claim order, so neither policy's numbers can differ because of
a different runtime pass.

## Paired-bootstrap confidence intervals

When `bootstrap.enabled` is true, claims are resampled with replacement
(`numpy.random.default_rng(seed)`), and for every resample the six audit
metrics (citation-correctness F1, claim macro-F1, coverage, evidence-sentence
F1, faithfulness, unsupported-assertion rate) are **recomputed from that
resample's pooled counts and labels** -- never by averaging per-claim F1
values, which is not the same quantity as the reported pooled F1. Each draw
is given an occurrence-specific namespace internally so a claim drawn twice
in one resample contributes two distinct entries rather than collapsing under
set deduplication. The same resampled claim-id sequence is applied to both
policies in a given replicate (a paired design), isolating the policy effect
from resampling noise.

## Result JSON

`results/final_evaluation_dev.json` records `evaluation_label:
"held_out_development_evaluation"` and `is_independent_test: false`, every
input artifact's path and SHA-256, the frozen trace's own SHA-256
(`trace_artifact.sha256`), both policies' full summaries, the
selected-minus-Phase-05 deltas, and the bootstrap confidence intervals. The
result JSON cannot record its own hash (it does not exist yet while being
written); the narrative report and agent trace record it after the fact.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m evidence_agent build-index
& .\.venv\Scripts\python.exe -m evidence_agent train-verifier
& .\.venv\Scripts\python.exe -m evidence_agent calibrate-citation-audit
& .\.venv\Scripts\python.exe -m evidence_agent evaluate --config configs/final.yaml
```

The committed output is `results/final_evaluation_dev.json`. The raw runtime
trace is a local ignored artifact under `artifacts/`; the committed report
records its path and SHA-256.
