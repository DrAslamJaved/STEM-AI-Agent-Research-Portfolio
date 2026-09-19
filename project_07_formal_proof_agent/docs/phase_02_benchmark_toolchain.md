# Phase 02 — Frozen Benchmark and Toolchain Validation

## Objective

Make the miniF2F evaluation environment reproducible before implementing any proof-generation workflow. This phase validates the original Lean 3 miniF2F `v1` branch, rather than a newer Lean 4 translation or an unpinned default branch.

## Locked environment

| Component | Locked value |
| --- | --- |
| Benchmark repository | `https://github.com/openai/miniF2F.git` |
| Benchmark branch | `v1` |
| Benchmark commit | `f0dcc8b59e630fba00ba9569ca6714700e0a8801` |
| Lean toolchain | `leanprover-community/lean:3.42.1` |
| Lean release commit | `68455b087d87` |
| mathlib revision | `cb2b02fff213ed6f65bebd64446baac64137dcda` |
| Development split | `valid` |
| Held-out split | `test` |

The `test` split is locked: it must not be used for prompt design, repair-policy tuning, or model selection.

## Evidence of validation

The generated file `reports/phase_02/toolchain_validation.json` is the machine-readable record of a successful configuration and build. A valid record requires:

1. the exact benchmark commit and mathlib revision above;
2. `configure_status: "passed"`;
3. `build_status: "passed"`; and
4. `build_exit_code: 0`.

The local `upstream/` checkout, its `_target/` cache, and build/configuration logs are operational artifacts. They must not be committed. The provenance manifest and concise JSON result are the durable reproducibility evidence.

## Interpreting `sorry` warnings

The frozen miniF2F source files intentionally contain theorem declarations with `sorry`; this is how benchmark goals are represented. These warnings do not invalidate a successful baseline build. They are categorically different from a generated candidate proof: Project 07's validity policy prohibits `sorry`, `admit`, and unapproved axioms in every submitted candidate.

## Re-run procedure

Use PowerShell 7, then invoke the exact toolchain:

```powershell
& $Elan run 'leanprover-community/lean:3.42.1' leanpkg configure
& $Elan run 'leanprover-community/lean:3.42.1' leanpkg build
```

Do not replace the branch, Lean release, or mathlib revision to make an error disappear. Record the failure and investigate the locked environment instead.
