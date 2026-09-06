# Dataset Card

## Dataset identification

- **Dataset name:** Seoul Bike Sharing Demand
- **Repository:** UCI Machine Learning Repository
- **UCI dataset identifier:** 560
- **Source URL:** https://archive.ics.uci.edu/dataset/560/seoul%2Bbike%2Bsharing%2Bdemand
- **DOI:** https://doi.org/10.24432/C5F62R
- **Licence:** Creative Commons Attribution 4.0 International
- **Date accessed:** 25 August 2026
- **Local raw-data path:** `data/raw/SeoulBikeData.csv`
- **File size:** 604,166 bytes
- **File encoding:** Latin-1
- **SHA-256:** `373339B71A8935D69E9AF0ABF26A70744632119862EEB3919EFB389A7B749C60`

## Intended use

The dataset is used to develop and evaluate a reproducible
time-series forecasting and residual-based anomaly-detection agent.

The principal forecasting target is the hourly number of rented bicycles.

## Unit of observation

Each row represents one hour in the Seoul Bike Sharing System.

## Time information

- **Raw date column:** `Date`
- **Raw hour column:** `Hour`
- **Constructed timestamp column:** `timestamp`
- **Date format:** `DD/MM/YYYY`
- **First timestamp:** `2017-12-01 00:00:00`
- **Last timestamp:** `2018-11-30 23:00:00`
- **Nominal frequency:** Hourly
- **Observed rows:** 8,760
- **Expected hourly timestamps:** 8,760
- **Missing hourly timestamps:** 0
- **Duplicate timestamps:** 0
- **Timezone:** Not specified in the source documentation

The modelling timestamp is constructed by combining `Date` and
`Hour`.

## Target variable

- **Column:** `Rented Bike Count`
- **Definition:** Number of bicycles rented during the hour
- **Type:** Nonnegative integer count
- **Minimum:** 0
- **Maximum:** 3,556
- **Mean:** 704.602
- **Median:** 504.500
- **Missing values:** 0
- **Negative values:** 0
- **Zero values:** 295

All 295 zero target values occur when `Functioning Day` is `No`.
Consequently, these values are interpreted as documented service-closure
or nonfunctioning periods rather than ordinary missing observations.

They will not automatically be labelled as anomalies.

## Additional variables

| Variable | Description |
|---|---|
| `Temperature(°C)` | Hourly air temperature |
| `Humidity(%)` | Relative humidity |
| `Wind speed (m/s)` | Wind speed |
| `Visibility (10m)` | Visibility measurement |
| `Dew point temperature(°C)` | Dew-point temperature |
| `Solar Radiation (MJ/m2)` | Solar-radiation measurement |
| `Rainfall(mm)` | Rainfall |
| `Snowfall (cm)` | Snowfall |
| `Seasons` | Winter, Spring, Summer, or Autumn |
| `Holiday` | Holiday indicator |
| `Functioning Day` | Whether the bike system was functioning |

## Data-quality audit

The initial audit found:

- 8,760 rows and 14 columns;
- no missing cells;
- no duplicate complete rows;
- no invalid dates;
- no invalid or out-of-range hours;
- no duplicate timestamps;
- no missing hourly timestamps;
- no invalid or negative target values;
- complete chronological ordering;
- an exact hourly interval throughout the dataset.

The audit evidence is stored in:

`reports/validation/phase_02_raw_data_audit.txt`

## Preprocessing decisions

No raw observations were deleted or imputed during structural
preprocessing.

Implemented preprocessing includes:

1. parsing `Date` using the explicit `%d/%m/%Y` format;
2. validating that `Hour` lies between 0 and 23;
3. combining `Date` and `Hour` into `timestamp`;
4. sorting observations chronologically;
5. preserving the original raw dataset byte-for-byte;
6. distinguishing documented nonfunctioning periods from unexplained
   candidate anomalies;
7. creating lagged and rolling features without using future
   observations;
8. excluding the initial 168 hours only from the model-ready feature
   matrix because the longest historical feature is unavailable there;
9. fitting any data-derived transformation using training data only.

Structural preprocessing retains all 8,760 hourly observations. The
lag-based feature matrix contains 8,592 usable observations after the
initial 168-hour history requirement is applied.

## Licence and attribution

The dataset is distributed through the UCI Machine Learning Repository
under the Creative Commons Attribution 4.0 International licence.

Suggested dataset citation:

> Seoul Bike Sharing Demand. (2020). UCI Machine Learning Repository.
> https://doi.org/10.24432/C5F62R

The dataset may be shared and adapted provided appropriate attribution
is supplied.

## Ethical and privacy considerations

The dataset contains aggregated hourly bicycle-rental counts and
weather-related variables. The supplied file does not contain direct
personal identifiers.

Results should not be interpreted as individual-level travel behaviour.

## Known limitations

- The observations cover one city.
- The observations cover approximately one year.
- The source does not specify a timezone in its repository description.
- Rental counts measure completed rentals, not unconstrained demand.
- Availability of bicycles may limit observed counts.
- Service-closure periods generate structural zero values.
- Historical relationships may not generalize to other cities or years.
- Unrecorded public events may influence unusually high or low demand.
- A large forecast residual is not automatically a confirmed real-world
  anomaly.

## Current status

Dataset acquisition, checksum verification, structural validation,
preprocessing, and feature engineering are complete.

The dataset has been used for:

- chronological holdout evaluation;
- 12-fold expanding-window evaluation;
- baseline, Holt-Winters, and recursive Gradient Boosting forecasting;
- out-of-sample residual collection;
- candidate anomaly and episode analysis;
- evidence-based model recommendation.

The raw source file is protected from automatic line-ending conversion
by `.gitattributes`. Its byte-exact SHA-256 checksum is verified by the
automated test suite on Windows and Linux.

The dataset contains no verified anomaly labels. Residual-based findings
are therefore reported as candidate alerts requiring contextual review.