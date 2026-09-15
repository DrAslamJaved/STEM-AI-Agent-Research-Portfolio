# v0.2 Model Environment

Create a clean virtual environment in the Project 10 worktree before running
the DTI model increment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The initial baseline environment uses NumPy 1.x, SciPy 1.10--1.14, and
scikit-learn 1.4--1.6. This avoids mixing a NumPy 2 installation with binary
extensions compiled for NumPy 1.x. Do not modify the global Anaconda packages
to run this project.
