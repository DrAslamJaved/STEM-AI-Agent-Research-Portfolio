# Agentic Causal Inference and Assumption-Audit System

Project 08 asks a stricter question than predictive machine learning: **what would happen under an intervention, and are the assumptions sufficient to justify estimating it?**

The package implements a fail-closed causal workflow:

1. classify and validate the causal question;
2. construct and audit the causal DAG;
3. identify the estimand or reject the request;
4. estimate average or heterogeneous effects;
5. diagnose overlap, balance, and weighting stability;
6. run refutation and hidden-confounding sensitivity analyses;
7. evaluate against synthetic truth, IHDP, or Lalonde evidence where appropriate;
8. emit machine-readable JSON and qualified Markdown audit reports.

## Safety rule

No numerical causal conclusion is emitted after an invalid question, invalid DAG, insufficient assumptions, or non-identifiable estimand. Automated robustness checks never replace human approval of the DAG, assumptions, or final interpretation.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Optional DoWhy/EconML integrations can be installed with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[causal,test]"
```

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest --cov=causal_audit_agent --cov-branch --cov-report=term-missing --cov-fail-under=98
```

## Programmatic orchestration

Construct a `CausalQuestion`, a reviewed `CausalDAG`, and a pandas data frame, then call `causal_audit_agent.orchestrator.run_analysis`. The returned `AnalysisRun` contains every reached stage, the final decision state, warnings, and an ordered audit trail. Use `render_json` or `render_markdown` from `causal_audit_agent.reporting` for release artifacts.

See `docs/orchestration_protocol.md`, `docs/model_card.md`, and `PROJECT_PHASE_INDEX.md` for the complete scientific and operational contract.
