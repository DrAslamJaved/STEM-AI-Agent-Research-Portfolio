# Causal graph and assumption-audit protocol

Phase 04 converts an approved causal question into an explicit directed acyclic
graph (DAG) specification. The graph is an assumption ledger, not a learned
ground truth and not evidence that an effect is identifiable.

## Required declarations

Every graph declares:

- one observed treatment and one distinct observed outcome;
- every variable, its causal role, observability, and optional temporal order;
- every directed causal edge;
- the proposed adjustment variables, if any; and
- consistency, exchangeability, positivity, no interference, intervention
  definition, and temporal-order assumptions.

## Automated audit

The Phase 04 audit checks:

1. unique, non-empty variable names and declared edge endpoints;
2. treatment/outcome presence, roles, observability, and distinctness;
3. acyclicity, absence of self-loops, and a directed treatment-to-outcome path;
4. declared temporal order along every edge;
5. latent ancestors shared by treatment and outcome;
6. adjustment attempts involving the treatment, outcome, unknown or latent
   variables, post-treatment variables, or explicitly declared colliders; and
7. missing causal assumptions.

## Interpretation boundary

`valid_for_identification_review` means only that the specification may proceed
to a later identification stage. It does **not** mean that:

- the arrows are scientifically correct;
- unmeasured confounding is absent;
- the proposed adjustment set satisfies a back-door criterion;
- positivity holds in the target data; or
- the requested effect is identified.

Every graph therefore retains `requires_human_review = true`. Structural errors
block progression. Missing or untestable assumptions remain visible as warnings
for domain-expert review.
