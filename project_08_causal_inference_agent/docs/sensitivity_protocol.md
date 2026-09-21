# Hidden-confounding sensitivity protocol

Phase 10 asks how strong an omitted common cause would have to be to materially
change an estimated effect. It does not claim that hidden confounding is absent.

## Linear omitted-variable-bias analysis

The `linear_ovb` analysis is restricted to additive ATE or ATT coefficients from
linear outcome regression. Let `R2_DU_X` be the partial R-squared of an omitted
confounder with treatment conditional on measured covariates, and let
`R2_YU_DX` be its partial R-squared with outcome conditional on treatment and
measured covariates. The absolute bias bound is

`SE(tau) * sqrt(df * R2_YU_DX * R2_DU_X / (1 - R2_DU_X))`.

The robustness value is the minimum equal partial R-squared strength required
to reduce the estimated effect toward the configured null by the requested
fraction. Scenario bounds report both bias directions and whether the null is
crossed.

## E-value analysis

The `e_value` analysis is accepted only for effects already expressed as risk
ratios with the unit null. It reports the point-estimate E-value and the E-value
for the confidence-limit closest to the null. It does not convert odds ratios,
hazard ratios, or mean differences into risk ratios.

## Status policy

- `ROBUST`: the configured robustness threshold is met.
- `SENSITIVE`: the threshold is not met.
- `INCONCLUSIVE`: the baseline confidence interval includes the null or the
  risk-ratio confidence interval is unavailable.
- `BLOCKED`: baseline effect estimation did not succeed.
- `INVALID_REQUEST`: the analysis is mathematically incompatible with the
  estimand, estimator, effect scale, or supplied uncertainty information.

Every report requires human review. A robust result is conditional on the
chosen sensitivity model and thresholds; it does not validate the DAG,
exchangeability, positivity, consistency, measurement quality, or estimator.

## Method references

- Cinelli, C., and Hazlett, C. (2020). Making sense of sensitivity: Extending
  omitted variable bias. *Journal of the Royal Statistical Society: Series B*,
  82(1), 39–67. https://doi.org/10.1111/rssb.12348
- VanderWeele, T. J., and Ding, P. (2017). Sensitivity analysis in observational
  research: Introducing the E-value. *Annals of Internal Medicine*, 167(4),
  268–274. https://doi.org/10.7326/M16-2607
