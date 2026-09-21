# Phase 09 live-study prompt protocol

The actual prompts are constructed by `formal_math.live_study`; this file is
the auditable human-readable protocol.

1. The model returns only a Lean proof body.
2. It never edits, repeats, or weakens the frozen theorem declaration.
3. `sorry`, `admit`, and `axiom` are prohibited and statically rejected.
4. The LLM-only arm receives no tool evidence and no evaluation feedback.
5. The SymPy arm receives only serialized exact diagnostics/counterexamples;
   it never receives Lean output, compile status, or a prior proof body.
6. The Lean arm may receive only the immediately prior candidate and the
   independent compiler/static result, and only for repair iterations 1–2.
7. Lean compilation is the final proof-validity decision in every arm.

All prompt, feedback, context, and raw-completion SHA-256 values are retained
in the raw run directory for audit.
