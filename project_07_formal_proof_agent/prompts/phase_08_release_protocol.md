# Phase 08 release protocol

Run this protocol only after the three study arms have been configured against
the same frozen miniF2F test manifest. The `valid` split is for development;
do not use it for final Phase 08 reporting.

1. Freeze the task manifest, prompt identifier, model identifier, decoding
   budget, seed schedule, and samples per theorem before generating candidates.
2. Capture every candidate attempt, including failed attempts, static-policy
   rejections, elapsed time, token counts, estimated cost, and any recorded
   human intervention.
3. Treat an independent Lean compilation result as the only proof-validity
   decision. SymPy may contribute symbolic diagnostics or counterexample search
   for its arm, but it cannot establish a formal proof.
4. Keep theorem declarations immutable. Reject `sorry`, `admit`, `axiom`, and
   any other policy violation before measuring proof success.
5. Run `run_phase_08_final_evaluation.py` only on a manifest marked
   `synthetic_demo: false` and on locked `test`-split attempt records.
6. Review the generated JSON evidence, report limitations, record the human
   review decision, and publish a release only when the release gate reports
   `release_ready: true`.

The repository fixture is deliberately synthetic. It is a reproducibility and
release-gate check, not a benchmark score or a claim that one model arm wins.
