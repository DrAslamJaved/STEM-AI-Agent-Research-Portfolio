# Phase 05 Agent Trace — SymPy Diagnostics and Exact Counterexamples

## Decisions locked

1. The second study arm is labelled `llm_sympy`; it is distinct from the
   tool-free `llm_only` baseline and the later Lean repair arm.
2. The only generation-time tool context is SymPy symbolic residual output and
   finite-grid exact rational counterexamples.
3. Lean compiler diagnostics, compilation status, prover-search results,
   previous candidate text, and held-out `test` tasks are unavailable to this
   arm during generation.
4. All fuzzy-membership witnesses use exact rationals in the closed unit
   interval; no floating-point counterexample evidence is accepted.
5. `J(0,0)=1` is explicitly recorded for singleton Jaccard similarity.
6. A missing finite-grid witness is never reported as a proof of the claim.
7. Candidate capture closes before the Phase 03 evaluator is invoked; Lean
   compilation remains the formal-validity decision.
8. The supplied smoke run is synthetic and does not support a miniF2F or LLM
   performance claim.

## Phase gate

Phase 05 is ready when SymPy dependency, policy validator, exact-witness unit
tests, provenance capture, `valid`/`test` lock, and synthetic post-hoc
evaluation ordering all pass. Phase 06 may add Lean diagnostics only under a
separate repair policy.
