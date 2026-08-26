# Residual Anomaly Report

## Purpose

This report identifies unusually large out-of-sample forecast residuals. Detected rows are candidate alerts, not automatically confirmed real-world anomalies.

## Detection method

Residuals are defined as actual demand minus forecast demand. A median-and-MAD modified z-score threshold of 3.5 is used.

Known service closures are preserved and scored but excluded from threshold calibration and actionable alerts.

## Detection summary

| Measure | Value |
|---|---:|
| Out-of-sample rows | 2016 |
| Nonclosure reference rows | 1769 |
| Known closures | 247 |
| Actionable anomaly hours | 115 |
| Positive anomaly hours | 87 |
| Negative anomaly hours | 28 |
| Actionable rate | 6.50% |

## Episode summary

The 115 actionable hours form 37 consecutive episodes.

- Forecast-floor positive episodes: 12 episodes covering 62 hours.
- Rain-coincident negative episodes: 5 episodes covering 22 hours.
- Other residual episodes: 20 episodes covering 31 hours.
- Share of anomalous hours occurring on the ten most concentrated dates: 80.87%.

## Ten largest episodes

| ID | Start | End | Hours | Direction | Context | Maximum score |
|---:|---|---|---:|---|---|---:|
| 13 | 2018-10-05 12:00:00 | 2018-10-05 23:00:00 | 12 | negative | rain_coincident_negative_episode | 8.03 |
| 7 | 2018-09-17 14:00:00 | 2018-09-17 23:00:00 | 10 | positive | forecast_floor_positive_episode | 10.30 |
| 34 | 2018-11-02 13:00:00 | 2018-11-02 22:00:00 | 10 | positive | forecast_floor_positive_episode | 7.36 |
| 32 | 2018-11-01 15:00:00 | 2018-11-01 22:00:00 | 8 | positive | forecast_floor_positive_episode | 7.23 |
| 3 | 2018-09-15 16:00:00 | 2018-09-15 23:00:00 | 8 | positive | forecast_floor_positive_episode | 5.51 |
| 14 | 2018-10-07 13:00:00 | 2018-10-07 20:00:00 | 8 | positive | other_residual_episode | 5.45 |
| 27 | 2018-10-30 16:00:00 | 2018-10-30 21:00:00 | 6 | positive | forecast_floor_positive_episode | 6.60 |
| 36 | 2018-11-08 16:00:00 | 2018-11-08 21:00:00 | 6 | negative | rain_coincident_negative_episode | 6.33 |
| 29 | 2018-10-31 16:00:00 | 2018-10-31 20:00:00 | 5 | positive | forecast_floor_positive_episode | 6.72 |
| 23 | 2018-10-27 13:00:00 | 2018-10-27 17:00:00 | 5 | positive | forecast_floor_positive_episode | 4.22 |

## Ten strongest anomalous hours

| Rank | Timestamp | Actual | Forecast | Residual | Score | Type | Rainfall |
|---:|---|---:|---:|---:|---:|---|---:|
| 1 | 2018-09-17 18:00:00 | 3277 | 67.51 | 3209.49 | 10.30 | positive_demand_spike | 0.0 |
| 2 | 2018-10-05 18:00:00 | 74 | 2601.21 | -2527.21 | 8.03 | negative_demand_drop | 1.0 |
| 3 | 2018-09-17 19:00:00 | 2489 | 0.00 | 2489.00 | 8.00 | positive_demand_spike | 0.0 |
| 4 | 2018-11-02 18:00:00 | 2314 | 25.63 | 2288.37 | 7.36 | positive_demand_spike | 0.0 |
| 5 | 2018-11-01 18:00:00 | 2254 | 3.97 | 2250.03 | 7.23 | positive_demand_spike | 0.0 |
| 6 | 2018-09-17 20:00:00 | 2232 | 0.00 | 2232.00 | 7.18 | positive_demand_spike | 0.0 |
| 7 | 2018-09-17 21:00:00 | 2100 | 0.00 | 2100.00 | 6.75 | positive_demand_spike | 0.0 |
| 8 | 2018-10-31 18:00:00 | 2094 | 3.97 | 2090.03 | 6.72 | positive_demand_spike | 0.0 |
| 9 | 2018-10-05 20:00:00 | 44 | 2151.26 | -2107.26 | 6.69 | negative_demand_drop | 1.0 |
| 10 | 2018-10-05 19:00:00 | 55 | 2161.78 | -2106.78 | 6.69 | negative_demand_drop | 1.5 |

## Interpretation

Rain-coincident negative episodes identify demand drops observed during rainfall. This association does not by itself prove that rainfall caused the drop.

Forecast-floor positive episodes occur when recursive forecasts approach zero before observed demand returns to ordinary or high levels. These are important model-recovery failures rather than confirmed unusual demand.

Other residual episodes require contextual review. They may reflect unmodeled events, changing demand, weather not represented in the forecast, or ordinary forecast error.

## Limitations

- The dataset contains no externally verified anomaly labels.
- Statistical alerts are candidates requiring domain review.
- The forecasting model excludes future weather because reliable future weather was not assumed available.
- Recursive forecasting can propagate errors through later lag and rolling features.
- The 3.5 threshold is a documented statistical rule, not a guarantee of operational importance.
