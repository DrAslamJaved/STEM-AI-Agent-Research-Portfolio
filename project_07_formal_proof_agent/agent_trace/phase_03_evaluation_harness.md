# Phase 03 Agent Trace — Evaluation Harness

## Decisions locked

1. Candidate generation is separated from outcome evaluation. All three arms
   produce the same captured candidate schema.
2. `valid` is the only development split. `test` is unavailable until an
   explicit final-evaluation invocation; mixed batches fail closed.
3. The task manifest owns the theorem declaration. Candidates supply proof
   bodies only, which prevents untracked theorem or assumption changes.
4. `sorry`, `admit`, and `axiom` are statically rejected before compilation.
5. Lean compiler diagnostics and exit code are captured as evidence; successful
   compilation is the formal validity decision for clean candidates.
6. The synthetic run is labelled as a harness smoke test and has no benchmark
   or model-performance interpretation.

## Phase gate

Phase 03 is ready when its tests pass, the validator confirms split and policy
locks, and the synthetic artifact records candidate source plus all three final
statuses. Phase 04 will connect the baseline generator to this fixed interface.
