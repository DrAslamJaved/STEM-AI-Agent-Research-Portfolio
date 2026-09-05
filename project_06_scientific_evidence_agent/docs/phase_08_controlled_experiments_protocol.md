# Phase 08 — Controlled experiments protocol

## Purpose

Phase 08 measures the trade-off between a deliberately simple direct-RAG
baseline and the frozen citation-audit agent. It does **not** retrain the
verifier, rebuild retrieval, recalibrate thresholds, or select a policy. Those
operations belong to earlier phases.

The experiment is a held-out development evaluation. It must not be described
as an independent test or used to tune configuration after the run.

## Fixed comparison

| Arm | Retrieval and evidence rule | Thresholding |
| --- | --- | --- |
| Direct RAG | Cite BM25 rank 1 and its up-to-three highest-scoring sentences | No assertion or sentence acceptance threshold |
| Audited agent | Apply the already-selected Phase 06 citation-audit policy | Frozen assertion, sentence, and maximum-sentence policy |

Both arms receive the same raw `VerificationTrace` objects, in the same claim
order. The raw trace contains only claims, public corpus text, BM25 retrieval,
and verifier outputs. It has no SciFact `evidence` or `cited_doc_ids` fields.

## Leakage boundary and provenance

1. Validate the SHA-256 of corpus, development claims, BM25 index, verifier
   bundle, calibration report, and (when declared) training claims.
2. Build one raw runtime trace using the fixed artifacts.
3. Write and hash that trace.
4. Produce and hash both official-format prediction JSONL files from that same
   trace.
5. Only then load development gold annotations for metrics and bootstrap
   resampling.

The evaluator rejects output paths that would overwrite a declared input or a
Phase 06/07 artifact. Result provenance paths are project-relative POSIX paths
so reports remain portable across Windows and CI environments.

## Metrics

The report retains the project's audit metrics (claim macro-F1, coverage,
faithfulness, unsupported-assertion rate, strict citation correctness, and
evidence-sentence F1) and adds SciFact-compatible abstract and sentence F1.

The official SciFact semantics are implemented directly from the
[evaluation guidance](https://github.com/allenai/scifact/blob/master/doc/evaluation.md):

- An abstract is correct only when its document and stance are correct and at
  least one complete gold rationale set is contained in the first three
  predicted rationale sentences.
- A predicted evidence sentence is correct only when its document and stance
  are correct, it belongs to a gold rationale set, and every other sentence in
  that same gold set was also predicted.

The unit suite includes adversarial checks for a wrong document, wrong stance,
and incomplete multi-sentence rationale.

## Statistical comparison

Paired nonparametric bootstrap resampling draws claim IDs once per replicate
and applies that exact draw to both arms. Duplicate draws receive distinct
occurrence IDs before scoring, so repeated samples never collapse in a map or
set. Every resample recomputes pooled metrics from the full resampled evidence,
rather than averaging per-claim F1 values.

The reported deltas are `audited_agent - direct_rag`. A positive value is not a
blanket improvement: coverage, faithfulness, unsupported assertions, and the
confidence intervals must be interpreted together. A qualitative superiority
claim is warranted only for a stated metric when its paired interval excludes
zero and the trade-off remains acceptable.

## Frozen execution

From the project directory, after the six artifact hashes in
`configs/controlled_experiments.yaml` have been verified:

```powershell
& .\.venv\Scripts\python.exe -m evidence_agent controlled-experiments `
  --config configs/controlled_experiments.yaml
```

This writes separate Phase 08 result, trace, direct-RAG predictions,
audited-agent predictions, Markdown report, and execution trace. Commit those
generated Phase 08 artifacts only after the test suite, compilation, hash
checks, and review pass.

## Phase 08 artifact re-freeze record

On 2026-09-05, the original Phase 07 verifier artifact
`827fca5323c73d6a54fe281cb56038eb9287f685fdec08db93f18dc39c8c55cb`
was no longer present after Phase 07 worktree cleanup. Phase 08 therefore
re-froze only its verifier-model input to the surviving deterministic artifact
`13ceccf71f78a00d3f9b4c3919d0a7a2ba72c24c0693b41b44e34d4f8a846c43`.

The replacement bundle records the identical corpus SHA-256
`b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62`
and training-claims SHA-256
`f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0`;
it uses the same lexical-verifier format, random seed 20260904, 40,000
features, and scikit-learn 1.9.0. Phase 06/07 reports remain unchanged.