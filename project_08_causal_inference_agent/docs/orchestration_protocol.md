# End-to-end orchestration protocol

## Governing principle

The agent fails closed. It may emit a numerical causal estimate only after the question, DAG, assumptions, and estimand pass their respective gates. Identification is conceptual; estimability and robustness are empirical and remain separate decisions.

## Ordered workflow

1. Validate the causal-question contract and classify the question.
2. Audit the DAG, temporal ordering, variable roles, and declared assumptions.
3. Identify the requested estimand or stop with a non-identifiable/insufficient-assumptions state.
4. Estimate the effect with a declared baseline estimator.
5. Optionally run cross-fitted AIPW or heterogeneous-effect estimation.
6. Diagnose propensity overlap, balance, weights, and effective sample size.
7. Run refutation and perturbation tests.
8. Quantify hidden-confounding sensitivity and robustness bounds.
9. When benchmark truth is supplied, compute only the metrics supported by that truth level.
10. Produce structured JSON and qualified Markdown reports with a complete stage trace.

## Decision states

- `INVALID_CAUSAL_QUERY`: prediction, association, or an invalid contract was submitted.
- `INSUFFICIENT_ASSUMPTIONS`: the DAG or assumption record cannot support identification review.
- `NON_IDENTIFIABLE`: no supported identifying strategy exists under the declared graph.
- `IDENTIFIED_BUT_NOT_ESTIMABLE`: empirical overlap or estimation conditions are inadequate.
- `IDENTIFIED_BUT_FRAGILE`: refutation, advanced estimation, or sensitivity results are concerning.
- `IDENTIFIED_AND_ROBUST`: all enabled automated gates passed; human approval is still required.
- `HUMAN_REVIEW_REQUIRED`: optional gates were omitted or classification remains ambiguous.

## Reporting constraints

- A blocked run must state that no numerical causal estimate was emitted.
- Identification, estimation, diagnostics, refutation, sensitivity, and benchmarking are separate sections.
- PEHE is reported only when individual treatment-effect truth exists.
- Refutation and sensitivity results never prove causal assumptions.
- Observational evidence must not be described as autonomous proof of causation.
- The DAG and final interpretation require recorded human approval.
