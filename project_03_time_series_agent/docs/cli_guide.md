# Command-Line Guide

## Purpose

The command-line interface provides one consistent way to run the forecasting agent. It connects the individual project scripts into named workflows and executes them in the correct order.

## Installation

Activate the virtual environment and install the project:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Help

Use the installed console command:

```powershell
time-series-agent --help
```

The same interface is available through Python:

```powershell
python -m time_series_agent --help
```

## Available workflows

| Command | Purpose |
|---|---|
| `audit-raw` | Inspect the original dataset |
| `validate` | Validate timestamps, targets, and closure information |
| `preprocess` | Create the clean hourly dataset |
| `explore` | Generate exploratory metrics and figures |
| `forecast` | Produce the next 24 Gradient Boosting forecasts |
| `evaluate` | Run chronological model evaluation |
| `anomalies` | Collect residuals, detect anomalies, form episodes, and generate the anomaly report |
| `recommend` | Select the preferred forecasting model and fallback |
| `run-all` | Execute the complete 17-script pipeline |

## Common examples

Generate the next 24 hourly forecasts:

```powershell
time-series-agent forecast
```

Run the complete anomaly workflow:

```powershell
time-series-agent anomalies
```

Generate the model recommendation:

```powershell
time-series-agent recommend
```

Preview the complete project pipeline:

```powershell
time-series-agent run-all --dry-run
```

Execute the complete pipeline:

```powershell
time-series-agent run-all
```

The complete pipeline can take several minutes because forecasting models are repeatedly trained across chronological validation windows.

## Safety behavior

The CLI:

1. uses the Python interpreter from the active environment;
2. runs scripts from the project root;
3. checks that required scripts exist;
4. follows the required dependency order;
5. prints numbered progress information;
6. stops immediately when a step fails;
7. returns a nonzero exit code after failure;
8. supports `--dry-run` for safe workflow inspection.

## Important outputs

The forecast workflow creates:

```text
reports/metrics/gradient_boosting_next_24_hours.csv
reports/metrics/gradient_boosting_diagnostics.json
```

The anomaly workflow creates:

```text
reports/metrics/gradient_boosting_oos_residuals.csv
reports/metrics/gradient_boosting_anomaly_labels.csv
reports/metrics/anomaly_episodes.csv
reports/anomaly_report.md
```

The recommendation workflow creates:

```text
reports/metrics/model_recommendation.json
reports/model_recommendation.md
```

## Troubleshooting

If `time-series-agent` is not recognized, reinstall the project:

```powershell
python -m pip install -e ".[dev]"
```

Alternatively, use:

```powershell
python -m time_series_agent forecast
```

When a workflow fails, read the reported script name and error. The CLI stops deliberately so later stages do not use incomplete or invalid outputs.