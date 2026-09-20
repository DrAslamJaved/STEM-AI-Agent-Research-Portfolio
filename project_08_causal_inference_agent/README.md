# Project 08 — Agentic Causal Inference and Assumption-Audit System

This project develops an identification-aware STEM Research AI Agent that distinguishes
prediction from causation, formalizes causal questions, audits assumptions, estimates effects,
and refuses unsupported causal conclusions.

## Scientific workflow

1. Model causal assumptions.
2. Identify the estimand.
3. Estimate the effect.
4. Refute and stress-test the result.

## Safety principle

A numerical estimate is not a causal conclusion unless the estimand is identified under an
explicit, approved causal model and the resulting estimate passes diagnostics and stress tests.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest
```

The optional causal stack can be installed later with `python -m pip install -e ".[causal,test]"`.
