# Phase 06 Agent Trace — Lean Generate–Verify–Repair

## Decisions locked

1. The third study arm is labelled `llm_lean_repair`.
2. Initial generation receives only a frozen theorem prompt.
3. A repair request may receive the immediately prior proof body plus the
   independent evaluator's static result and Lean stdout/stderr.
4. Every failed attempt is retained; repairs cannot overwrite history.
5. The theorem declaration, imports, benchmark record, and split are immutable.
6. Repairs stop at successful compilation or a maximum of two iterations.
7. Development accepts only `valid`; the held-out `test` split remains locked.
8. `sorry`, `admit`, and `axiom` are statically rejected before compilation.
9. Lean 3.42.1 with the Phase 02 miniF2F/mathlib pins is the real formal
   validator; synthetic fixtures only demonstrate workflow control.

## Phase gate

Phase 06 is ready when the policy and unit tests pass, the synthetic evidence
records compiler-informed repair ordering, and the compiler factory is pinned
to the Phase 02 Lean environment. Phase 07 may formalize the custom fuzzy
axioms and counterexamples under the same validity policy.
