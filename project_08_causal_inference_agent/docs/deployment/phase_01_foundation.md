        # Phase 01: Foundation and Mathematical Contract

        ## Objective

        Create the package, causal contracts, reproducible configuration, and initial quality gates.

        ## Deployment order

        None

        Extract this archive into the repository root. Review `git status` and `git diff` before staging.
        Do not commit generated datasets, local environments, caches, or credentials.

        ## Acceptance criteria

        - [ ] The package installs from pyproject.toml.
- [ ] A causal question validates treatment, outcome, estimand, population, and timing.
- [ ] Invalid or incomplete questions fail explicitly.
- [ ] The mathematical assumptions are documented and human-reviewable.

        ## Files in this overlay

        - `project_08_causal_inference_agent/README.md`
- `project_08_causal_inference_agent/configs/example_problem.yaml`
- `project_08_causal_inference_agent/docs/deployment/phase_01_foundation.md`
- `project_08_causal_inference_agent/docs/mathematical_contract.md`
- `project_08_causal_inference_agent/pyproject.toml`
- `project_08_causal_inference_agent/src/causal_audit_agent/__init__.py`
- `project_08_causal_inference_agent/src/causal_audit_agent/cli.py`
- `project_08_causal_inference_agent/src/causal_audit_agent/contracts.py`
- `project_08_causal_inference_agent/tests/test_contracts.py`

        ## Recommended branch

        `feature/project-08-phase-01-foundation`

        ## Recommended commit

        `feat(project08): complete phase 01 foundation and mathematical contract`
