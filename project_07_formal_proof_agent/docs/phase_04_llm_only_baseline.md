# Phase 04 — Controlled LLM-Only Baseline

## Objective

Create the comparator against which the later SymPy and Lean repair workflows
will be measured. The baseline is deliberately narrow: it generates Lean proof
bodies from frozen theorem declarations without access to tools, diagnostics,
counterexamples, retrieval, prior candidates, or repair feedback.

## Why this boundary matters

Every study arm must ultimately be scored by the same independent Lean
compilation policy. The baseline differs only in what it can use **during
generation**: nothing beyond the task prompt. Its candidates are captured in a
closed batch and then handed to the Phase 03 evaluation harness.

This avoids an invalid comparison in which one arm receives compiler errors or
symbolic evidence while another arm is evaluated only as prose. It also makes
the source of an improvement interpretable in later phases.

## Contract

| Control | Phase 04 policy |
| --- | --- |
| Generation arm | `llm_only` |
| Verification feedback during generation | Forbidden |
| Tool calls during generation | Forbidden |
| Repair iterations | `0` |
| Development split | miniF2F `valid` only |
| Held-out split | `test` locked until final evaluation |
| Formal validity decision | Independent post-hoc Lean compilation |

The generator does not import or call the evaluator. The synthetic script calls
the evaluator only after all candidates have been created; this ordering is
captured in the evidence bundle.

## Evidence captured

For each candidate, retain its candidate ID, theorem ID, arm, prompt ID, sample
index, model identifier, seed, complete prompt, raw completion, proof-body
source, SHA-256 digests, and available token counts. After evaluation, retain
the static-policy result, Lean exit code and diagnostics, elapsed time, and
final status.

The reported measures are:

- **compile@1:** fraction of theorem instances whose first captured sample
  compiles;
- **observed pass@k:** fraction with at least one compiling proof among the
  first \(k\) captured samples.

These are descriptive, controlled outcomes. They do not establish mathematical
generalization or model superiority from the synthetic fixture.

## Demonstration boundary

`run_phase_04_synthetic_baseline.py` uses two tiny synthetic valid-split tasks
and an offline scripted provider. It demonstrates the temporal boundary between
generation and evaluation, candidate provenance, and metric calculation. It
does not call an LLM, compile miniF2F, or report a benchmark result.

Phase 05 will add SymPy-supported diagnostics and exact counterexample search
under a separate controlled policy; Phase 04 remains unchanged as the baseline.
