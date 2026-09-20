# Phase 02: Synthetic Structural Causal Models

## Objective

Create reproducible structural causal models with known finite-sample average
treatment effects (ATEs) and unit-level treatment effects.

## Deployment order

Phase 01 must already be merged into `main`.

Extract this archive into the repository root. Review `git status` and `git diff`
before staging. Do not commit generated datasets, local environments, caches, or
credentials.

## Causal design

- `randomized`: treatment probability is 0.5 and independent of covariates.
- `observed_confounding`: observed covariates affect treatment and outcome.
- `nonlinear`: nonlinear treatment and outcome mechanisms are present.
- `heterogeneous`: the individual effect varies with `X1`.
- `hidden_confounding`: a latent variable affects treatment and outcome but is
  deliberately excluded from the estimator-facing data.
- `positivity_stress`: extreme propensity scores challenge overlap.
- `null`: the treatment effect is exactly zero.

The generator uses common exogenous outcome noise under both potential outcomes,
so `true_ite` equals `Y(1) - Y(0)`. The reported `true_ate` is the finite-sample
mean of `true_ite`.

## Acceptance criteria

- [ ] Synthetic generation is reproducible under a fixed seed.
- [ ] Ground-truth ATE and individual effects are returned with each dataset.
- [ ] All seven declared scenarios generate valid estimator-facing data.
- [ ] The randomized scenario has propensity 0.5.
- [ ] Hidden confounding is generated but not exposed as an observed feature.
- [ ] The positivity-stress scenario produces extreme propensities.
- [ ] Invalid scenarios, seeds, and sample sizes fail explicitly.

## Files in this overlay

- `project_08_causal_inference_agent/configs/synthetic/scenarios.yaml`
- `project_08_causal_inference_agent/docs/deployment/phase_02_synthetic_scm.md`
- `project_08_causal_inference_agent/src/causal_audit_agent/synthetic/__init__.py`
- `project_08_causal_inference_agent/src/causal_audit_agent/synthetic/scm.py`
- `project_08_causal_inference_agent/tests/test_synthetic.py`

## Required validation

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m pytest `
    --cov=causal_audit_agent `
    --cov-branch `
    --cov-report=term-missing
```

## Recommended branch

`feature/project-08-phase-02-synthetic-scm`

## Recommended commit

`feat(project08): complete phase 02 synthetic structural causal models`
