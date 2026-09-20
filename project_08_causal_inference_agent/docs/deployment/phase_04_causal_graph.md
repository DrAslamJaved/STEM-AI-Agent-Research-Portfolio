# Phase 04 deployment: causal DAG and assumption audit

## Objective

Add an auditable causal-graph contract between question parsing and formal
estimand identification.

## Scope

This phase adds:

- `src/causal_audit_agent/causal_graph.py`;
- `configs/graphs/example_dag.yaml`;
- `docs/causal_graph_protocol.md`;
- `docs/deployment/phase_04_causal_graph.md`; and
- `tests/test_causal_graph.py`.

No existing tracked file is modified. The implementation uses the `networkx`
dependency already declared in `pyproject.toml`.

## Safety properties

- Cyclic, malformed, temporally inconsistent, and treatment/outcome-invalid
  graphs are blocked.
- Latent common causes remain visible in the audit record.
- Proposed adjustment variables are screened for elementary invalid choices.
- A passing structural audit never claims identification.
- Human review remains mandatory for every DAG.

## Validation gates

Run from `project_08_causal_inference_agent`:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q

& .\.venv\Scripts\python.exe -m pytest `
    --cov=causal_audit_agent `
    --cov-branch `
    --cov-report=term-missing
```

Deployment is acceptable only when:

1. the complete Phase 01-04 suite passes;
2. the new graph module has full line and branch coverage;
3. repository-wide effective coverage remains at least 98%;
4. exactly the five approved Phase 04 files are introduced; and
5. `git diff --check` reports no whitespace errors.

## Explicit exclusions

Phase 04 does not run DoWhy identification, derive adjustment sets, estimate an
effect, test positivity empirically, or certify the scientific truth of a DAG.
Those responsibilities belong to later phases.
