# Phase 03 — Evaluation Harness

## Objective

Create the reproducible boundary between proof generation and proof validity.
The harness captures every candidate, prevents prohibited shortcuts, protects
the held-out `test` split, and applies an independent post-hoc Lean decision.

## Controlled comparison

| Arm | Feedback available while generating | Final decision |
| --- | --- | --- |
| `llm_only` | None | Independent post-hoc Lean compilation |
| `llm_sympy` | SymPy diagnostics and exact counterexample search | Independent post-hoc Lean compilation |
| `llm_lean_repair` | Bounded Lean diagnostics and repair feedback | Final Lean compilation |

The common final validator matters: a fluent explanation, a SymPy result, or a
successful intermediate repair message cannot establish a formal proof. Lean
compilation of the frozen theorem declaration is the outcome used for metrics.

## Split control

- **Development mode** accepts only miniF2F `valid` tasks.
- **Final-evaluation mode** accepts only `test` tasks and is explicit in the
  call site and captured result bundle.
- A mixed batch fails closed. This prevents an accidental held-out test result
  from entering prompt design, repair-policy tuning, or development reporting.

## Candidate and decision provenance

Each captured candidate has a stable candidate ID, theorem ID, arm, prompt ID,
repair iteration, source text, metadata, and source SHA-256. The result bundle
records the static policy outcome, compiler exit code, diagnostics, elapsed
time, and a final status of `compiled`, `failed_compile`, or
`rejected_static`.

The theorem declaration is stored in the task manifest and rendered by the
harness. Candidates contain only proof bodies, so they cannot silently change
assumptions or the proposition under review.

## Safety policy

Candidates containing `sorry`, `admit`, or `axiom` are rejected before Lean is
called. This is a first-line guard; the independent compilation result remains
the decisive validator for clean candidates.

## Demonstration boundary

`run_phase_03_synthetic_demo.py` produces deterministic harness smoke evidence
only. It intentionally uses a stub compiler to exercise all three outcomes:
one accepted candidate, one static rejection, and one compile failure. It is
not a miniF2F experiment, no LLM is called, and it must never be reported as a
proof-success or model-performance result.

The included `LeanSubprocessCompiler` is the production-facing interface. A
subsequent phase will connect it to the pinned miniF2F task manifest and real
candidate outputs under the Phase 02 Lean 3.42.1 environment.
