# Model card: Agentic Causal Inference and Assumption-Audit System

## Intended use

Research, education, and reproducible evaluation of cross-sectional causal-effect workflows with explicit human-reviewed assumptions. Supported tasks include causal-question classification, DAG audit, backdoor identification, effect estimation, empirical diagnostics, refutation, hidden-confounding sensitivity analysis, and IHDP/Lalonde evaluation.

## Intended users

Researchers and analysts who understand that causal conclusions depend on substantive assumptions that cannot generally be learned from observational data alone.

## Prohibited use

- Autonomous clinical, legal, financial, employment, or public-policy decisions.
- Individual treatment recommendations.
- Causal claims without a reviewed DAG, temporal ordering, and assumption record.
- Reporting a numerical effect after a non-identifiable or insufficient-assumptions decision.
- Treating refutation success as proof that all causal assumptions hold.

## Safety architecture

The system uses ordered gates. Invalid questions, invalid DAGs, unsupported estimands, and non-identifiable structures stop before estimation. Poor overlap produces an `IDENTIFIED_BUT_NOT_ESTIMABLE` state. Refutation or sensitivity concerns produce an `IDENTIFIED_BUT_FRAGILE` state. Every terminal state retains human-review requirements and an auditable stage trace.

## Known limitations

- The identifier implements a deliberately narrow backdoor-adjustment scope.
- Unmeasured confounding cannot be ruled out using observed data alone.
- Sensitivity analyses depend on their parameterization and model class.
- Refutation tests detect some forms of fragility but do not validate assumptions.
- Benchmark performance may not transport to a new population or treatment regime.
- PEHE is meaningful only where individual-effect ground truth is available.
- The current workflow focuses on binary treatments and cross-sectional outcomes.

## Human oversight

Human approval is required for the DAG, variable roles, identifying assumptions, estimator choice, diagnostic thresholds, sensitivity scenarios, and final interpretation.
