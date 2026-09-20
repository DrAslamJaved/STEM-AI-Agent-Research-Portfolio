# Estimand-identification and rejection protocol

Phase 05 determines whether a total causal effect can be expressed from the
observed distribution under the approved DAG and declared assumptions. It does
not estimate the effect.

## Supported identification scope

- estimands: ATE, ATT, and CATE;
- effect type: total effect only;
- strategy: observed-variable back-door adjustment; and
- output: all minimal adjustment sets found within the bounded search space,
  plus a symbolic observed-data expression.

For CATE, conditioning variables must be explicit, observed, pre-treatment,
and neither the treatment, outcome, nor an explicitly declared collider.

## Rejection states

The identifier never silently substitutes another causal question or strategy.
It returns one of:

- `IDENTIFIED`: an observed back-door adjustment expression was found;
- `NON_IDENTIFIABLE`: no observed back-door adjustment set exists;
- `INSUFFICIENT_ASSUMPTIONS`: required assumptions were not declared;
- `INVALID_QUERY`: the graph or requested conditioning is invalid; or
- `UNSUPPORTED_QUERY`: the estimand, effect type, or bounded search is outside
  the implemented contract.

Direct latent confounding therefore produces `NON_IDENTIFIABLE`, not a numeric
estimate. Direct effects, front-door identification, instrumental-variable
identification, longitudinal regimes, and general do-calculus are explicitly
out of scope for this phase.

## Assumption boundary

Identification is conditional on consistency, exchangeability, positivity, no
interference, a well-defined intervention, temporal order, and the approved
DAG. Automated d-separation cannot prove any of those scientific assumptions.
Every result retains `requires_human_review = true`.

## DoWhy handoff

The resulting treatment, outcome, graph, estimand, and adjustment sets form a
deterministic preflight record for the later DoWhy model/identify/estimate
workflow. A later adapter must preserve rejection states and may not call an
estimator when this phase does not return `IDENTIFIED`.
