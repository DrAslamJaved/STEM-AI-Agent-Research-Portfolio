# Phase 04 Agent Trace — LLM-Only Baseline

## Decisions locked

1. The baseline receives only frozen Lean imports and theorem declarations.
2. Generation requests contain no compiler diagnostics, proof-search results,
   SymPy output, counterexamples, prior candidates, or repair suggestions.
3. Every generated candidate is labelled `llm_only`, uses repair iteration `0`,
   and captures prompt/completion provenance before evaluation begins.
4. The complete generation batch closes before the Phase 03 evaluator is run.
5. Development mode accepts only `valid`; a `test` task fails closed before any
   provider call.
6. `compile@1` and observed `pass@k` are calculated from independently
   evaluated candidates, not model self-reports.
7. The supplied fixture is a synthetic control demonstration, not a miniF2F or
   LLM-performance result.

## Phase gate

Phase 04 is ready when contract/tests pass, provenance records are emitted
before evaluation, and the synthetic run demonstrates expected compile@1 and
pass@k calculation. Phase 05 may add SymPy only through a separate controlled
arm and must not modify this baseline policy.
