# Phase 08 - Synthetic smoke report

- Run ID: phase_08_synthetic_release_v1
- Evaluation split: test
- Formal validity decision: independent_lean_compilation
- Release ready: False

## Arm metrics

| Arm | Compile@1 | Pass@k | Compile after allowed workflow | Refutation success |
| --- | ---: | ---: | ---: | ---: |
| llm_only | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_sympy | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_lean_repair | 0.667 | 0.667 | 1.000 | 1.000 |

## Release gate

- Blocking condition: synthetic_demo=true; this fixture cannot establish a final research result
- Blocking condition: human_review.approved is false

## Evidence boundary

Only independent Lean compilation establishes formal proof validity. SymPy evidence supports diagnostics and refutation search, not formal proof. Synthetic fixtures are workflow checks, never model-performance results.
