# Phase 05 — SymPy Diagnostics and Exact Counterexamples

## Objective

Introduce the second controlled study arm, `llm_sympy`. It may use SymPy to
simplify algebraic residuals and find exact rational counterexamples, including
small fuzzy-set similarity examples. It may not access Lean compiler output,
compilation status, proof search, prior candidate text, or the held-out test
split while generating candidates.

## Why this is a distinct arm

The LLM-only baseline has no tool feedback. This arm is allowed a narrow,
recorded kind of feedback: SymPy diagnostics and finite-grid exact witnesses.
That makes a later difference interpretable as a consequence of symbolic
assistance rather than a hidden Lean repair loop.

| Capability during generation | LLM-only | LLM + SymPy | Lean repair (Phase 06) |
| --- | --- | --- | --- |
| Frozen theorem declaration | Yes | Yes | Yes |
| SymPy residual simplification | No | Yes | Optional diagnostic only |
| Exact rational counterexample search | No | Yes | Optional diagnostic only |
| Lean compiler diagnostics | No | No | Yes |
| Formal validity decision | Post-hoc Lean | Post-hoc Lean | Lean compilation |

## Exact fuzzy-set demonstration

The synthetic workflow treats a singleton fuzzy set as a membership value in
the interval \([0,1]\), with

\[
J(a,b) = \frac{\min(a,b)}{\max(a,b)},
\]

and the explicit convention \(J(0,0)=1\). It searches the exact grid
\(\{1/4, 1/2, 3/4, 1\}\) against the false claim that every pair of nonzero
memberships has Jaccard similarity one. The first witness is
\(a=1/4, b=1/2\), for which \(J(a,b)=1/2\).

This is a counterexample to the stated universal claim over that domain; it is
not a Lean proof and a failed finite search would not prove a claim outside the
declared grid.

## Guardrails

| Control | Phase 05 policy |
| --- | --- |
| Development split | miniF2F `valid` only |
| Held-out split | `test` locked until final evaluation |
| Allowed tool signal | Exact SymPy diagnostics and witnesses only |
| Lean/compiler feedback during generation | Forbidden |
| Repair budget | At most one SymPy-guided iteration |
| Numeric evidence | Forbidden; use exact rationals only |
| Formal verdict | Independent post-hoc Lean compilation |
| Prohibited proof shortcuts | `sorry`, `admit`, `axiom` |

## Evidence emitted

The workflow stores the exact diagnostic context, witness assignments, prompt
and completion digests, model/run metadata, initial and repaired candidate
records, then the independent evaluation results. The synthetic fixture uses an
offline scripted provider and evaluator. It demonstrates provenance and timing,
not benchmark performance or mathematical generalization.
