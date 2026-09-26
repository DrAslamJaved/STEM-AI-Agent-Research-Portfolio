# Phase 06 protocol: disagreement-scaled conformal intervals

## Trigger from Phase 05

The Phase 05 official five-fold evaluation contained 110,496 prediction-level records. Its rank correlation between bootstrap ensemble disagreement and absolute error ranged from 0.602 to 0.789 across policy and label budget. At 1,600 labels, fixed-width nominal 90% intervals covered 0.747, 0.774 and 0.754 of the highest-disagreement third for random, uncertainty and uncertainty-plus-diversity acquisition, respectively. Overall coverage was close to nominal because lower-disagreement records were overcovered.

This motivates a calibration evaluation. It does not motivate changing the Phase 03 conclusion about active-learning label efficiency.

## Locked calculation

For each existing policy, outer fold, seed, and scheduled budget:

1. Fit the same bootstrap forest ensemble to the selected pool labels.
2. Use the fixed calibration records already reserved by composition group.
3. Compute each calibration residual \(r_i=|y_i-\hat y_i|\) and ensemble disagreement \(s_i\).
4. Set the positive spread floor to \(f=\max(q_{0.10}(s_{\rm cal}),10^{-12})\).
5. Compute the finite-sample split-conformal quantile of \(r_i/\max(s_i,f)\) at \(\alpha=0.10\).
6. For a test record, form the interval \(\hat y\pm q\max(s,f)\).

No test label, error, coverage indicator, or post-hoc group enters either the model fit, the calibration quantile, or acquisition. The original constant-width interval is measured in parallel.

## Evaluation

Use the official five folds and seeds 17 and 23 with the existing budgets 200, 400, 800, and 1600. For each policy/budget, report fold-averaged fixed and scaled coverage, average interval width, and coverage in the bottom and top thirds of predicted disagreement. The top-third bins are diagnostic summaries after prediction, not calibration groups. Compare coverage with interval width; no predeclared superiority claim is made.

## Interpretation bounds

Split-conformal coverage is marginal under exchangeability. The scaled interval can redistribute width toward high-disagreement records, but it may be wider on average and it does not establish conditional coverage guarantees. Results are an exploratory follow-up to Phase 05. The original 0.60 eV ensemble threshold, its crossings, and the negative Phase 03 label-saving conclusion remain fixed.
