# Project 04: Agentic Cybersecurity Anomaly Detection Using PCA

## Objective

Develop and validate an agentic cybersecurity anomaly-detection workflow that
integrates:

- linear algebra;
- principal component analysis;
- machine learning;
- cybersecurity;
- software testing;
- AI agent evaluation.

## Research question

Can PCA reconstruction error reliably distinguish anomalous cybersecurity
observations from normal network traffic, and under what data and modelling
assumptions does this reasoning fail?

## Core hypothesis

If PCA is fitted using representative normal network traffic, the retained
principal components should represent the dominant linear correlation structure
of normal behaviour.

Observations that lie far from this learned normal subspace should have large
reconstruction errors and can therefore be treated as anomaly candidates.

A large reconstruction error is statistical evidence of unusual behaviour. It
is not proof that an observation is malicious.

## Workflow

```text
Raw cybersecurity data
↓
Data validation
↓
Leakage-safe preprocessing
↓
Feature standardization
↓
Covariance matrix
↓
Eigenvalue decomposition
↓
Principal components
↓
Dimensionality reduction
↓
Reconstruction
↓
Reconstruction error
↓
Normal-only threshold calibration
↓
Anomaly prediction
↓
Evaluation
↓
AI agent reasoning assessment
```

## Phase 1 verification evidence

The verified Phase 1 execution produced:

- 60 passing tests;
- 93.26% total coverage;
- seven passing independent mathematical checks;
- deterministic JSON validation output;
- successful package compilation;
- successful module and console entry points.

Evidence files:

```text
reports/validation/math_validation.json
reports/validation/phase_01_final_pytest.xml
reports/validation/phase_01_final_coverage.xml
agent_trace/phase_01.md
```

These results validate the mathematical and software foundation. They do not
represent cybersecurity anomaly-detection performance.

## Current implementation status

| Component | Status |
|---|---|
| Git feature branch | Completed |
| Isolated Python 3.12 environment | Completed |
| Project architecture | Completed |
| Python package configuration | Completed |
| Mathematical specification | Completed |
| Manual PCA implementation | Completed |
| Mathematical tests | Completed |
| Mathematical validation CLI | Completed |
| Synthetic cybersecurity data | Next phase |
| Anomaly detector | To be implemented |
| UNSW-NB15 experiment | To be implemented |
| Agent reasoning evaluation | To be implemented |
| Continuous integration | To be implemented |
