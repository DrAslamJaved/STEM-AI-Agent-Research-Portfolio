# Phase 01 — Research Contract

**Status:** `PENDING HUMAN APPROVAL`

## Primary question

Does tool-mediated generate–verify–repair improve formally valid proof completion
relative to LLM-only proof generation under a controlled budget?

## Hypotheses

- **H1:** Lean-guided repair improves post-hoc Lean compilation rate over LLM-only generation.
- **H2:** SymPy improves algebraic diagnosis and exact counterexample discovery,
  but does not replace formal proof verification.
- **H3:** The workflow improves valid proof completion while adding time and
  computational cost.

## Experimental arms

| Arm | Feedback during generation | Decisive final validator |
|---|---|---|
| LLM-only | None | Independent post-hoc Lean compilation |
| LLM + SymPy | SymPy simplification and exact counterexample search | Independent post-hoc Lean compilation |
| LLM + Lean repair | Lean compiler errors and bounded repair | Lean compilation |

## Primary outcomes

- `compile@1`
- `pass@k`, for `k ∈ {1, 4, 8}`
- initial versus repaired compilation rate
- repair iterations used
- valid counterexamples found for false claims
- wall-clock time and controlled token budget

## Benchmark policy

The project will pin frozen miniF2F v1 and its exact compatible Lean 3 toolchain
in Phase 02. The validation split may guide development; the test split remains
locked until final evaluation.

## Human approval gates

Human approval is required before:

1. benchmark acquisition and environment installation;
2. use of any external model or paid API;
3. release of results or benchmark claims;
4. manuscript preparation or submission.