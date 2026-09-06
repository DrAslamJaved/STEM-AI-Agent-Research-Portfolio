# Reproducibility Environment

## Validated Environment

- Operating System: Windows
- Python: 3.12.8
- pytest: 7.4.4
- Git: 2.55.0.windows.4

## Environment Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The project uses the following explicit external test dependency:

```text
pytest==7.4.4
```

## Run the Validation Suite

```powershell
python -m pytest
```

Expected validated result:

```text
18 passed
```

## Generate Machine-Readable Test Evidence

```powershell
python -m pytest --junitxml=results\pytest_results.xml
```

Expected artifact:

```text
results/pytest_results.xml
```

## Reproducibility Principle

A successful result on the original development machine is not by itself sufficient evidence of reproducibility.

The project records:

- source code;
- test suite;
- dependency specification;
- Python and pytest versions;
- execution commands;
- machine-readable test output;
- human-in-the-loop debugging traces;
- Git version history.

This allows the validated workflow to be reconstructed and independently rerun.