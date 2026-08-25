# Methodology

## 1. Study design

This project develops a reproducible agent for hourly time-series
forecasting and residual-based anomaly detection.

The forecasting target is `Rented Bike Count` from the Seoul Bike
Sharing Demand dataset.

## 2. Data preservation

The original CSV is stored unchanged in `data/raw/`.

The raw-file SHA-256 checksum is:

`373339B71A8935D69E9AF0ABF26A70744632119862EEB3919EFB389A7B749C60`

Transformations are written to `data/processed/`. The raw file is never
overwritten.

## 3. Timestamp construction

The canonical timestamp is constructed as:

```text
timestamp = parsed Date + Hour