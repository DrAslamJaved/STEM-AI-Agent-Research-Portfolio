# Phase 06 — Lean Generate–Verify–Repair

## Objective

Implement the decisive experimental arm, `llm_lean_repair`. It begins with a
frozen theorem declaration, independently evaluates the generated proof body,
then feeds the prior body plus its static-policy result or Lean diagnostics into
a bounded repair request. The declaration, imports, split, and benchmark record
remain immutable throughout the loop.

## Controlled workflow

1. Generate an initial Lean proof body from the frozen task.
2. Apply the Phase 03 static policy; reject `sorry`, `admit`, and `axiom`.
3. Compile non-rejected candidates with the pinned Lean 3.42.1 toolchain.
4. If unsuccessful, send the prior body and recorded evaluator feedback to the
   provider for one bounded repair attempt.
5. Stop when a candidate compiles or the two-repair budget is exhausted.
6. Preserve every candidate, prompt, completion, diagnostic, and evaluation
   result for audit and later paired comparison.

## Why Lean feedback is permitted here

The earlier arms establish the comparison boundary: LLM-only sees no tool
feedback, and LLM+SymPy sees only symbolic diagnostics and exact rational
witnesses. This phase deliberately introduces compiler feedback, making its
effect measurable under the same frozen tasks and final validity policy.

| Control | Phase 06 policy |
| --- | --- |
| Arm | `llm_lean_repair` |
| Initial attempt | Frozen theorem only |
| Repair signal | Independent static result and Lean stdout/stderr |
| Theorem edits | Forbidden |
| Repair budget | At most two repairs per theorem |
| Development data | miniF2F `valid` only |
| Held-out data | `test` locked until final evaluation |
| Formal validity | Independent Lean compilation |

## Toolchain integrity

The real compiler factory calls `elan run leanprover-community/lean:3.42.1
lean` from the Phase 02 pinned miniF2F working directory. The miniF2F commit
is `f0dcc8b59e630fba00ba9569ca6714700e0a8801`; its mathlib revision is
`cb2b02fff213ed6f65bebd64446baac64137dcda`. Newer toolchains, benchmark
revisions, or unrecorded imports must not be substituted.

## Evidence boundary

`run_phase_06_synthetic_lean_repair.py` uses an offline scripted provider and
a deterministic compiler fixture to demonstrate retry ordering and provenance.
It is not a miniF2F result and does not claim model performance. In a real run,
the same loop uses `make_pinned_lean3_compiler`; only its independent Lean
compilation result may be reported as formal proof validity.
