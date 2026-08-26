# Command-Line Guide

## Purpose

The command-line interface provides one consistent way to run the forecasting agent. It connects the individual project scripts into named workflows and runs them in the correct order.

## Installation

Activate the virtual environment and install the project:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"


- Mark the unified command-line forecasting pipeline as completed.
- Leave continuous integration and the final project report as remaining work.

### 3. Update the methodology

Open:

```powershell
notepad .\docs\methodology.md


## 21. Unified command-line orchestration

The project exposes a unified command-line interface through both `time-series-agent` and `python -m time_series_agent`.

Named workflows group existing scripts by purpose. The complete `run-all` workflow contains 17 scripts arranged in dependency order, beginning with raw-data inspection and ending with the evidence-based model recommendation.

Each script is launched as a subprocess using `sys.executable`. This ensures that the same Python interpreter and virtual environment used to start the CLI are also used by the workflow scripts.

The CLI executes from the project root, validates script existence before execution, reports progress using numbered steps, and stops immediately when a child process returns a nonzero exit code. A `--dry-run` option displays the planned scripts without executing them.

A real CLI smoke test verified the forecast, anomaly, and recommendation workflows. During testing, the anomaly workflow exposed an ordering defect: display columns were selected before sorting by `absolute_modified_z_score`. Candidate ranking was moved into a tested package function so sorting occurs before presentation columns are removed.

The completed interface was validated with 151 automated tests, 90% package coverage, compilation checks, and warnings treated as errors.