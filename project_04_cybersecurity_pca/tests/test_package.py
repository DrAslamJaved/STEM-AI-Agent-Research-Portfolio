"""Tests for package exports and reproducibility artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from cyber_pca import (
    ManualPCA,
    __version__,
    run_math_validation,
)


def test_public_package_exports() -> None:
    assert ManualPCA.__name__ == "ManualPCA"
    assert callable(run_math_validation)
    assert __version__ == "0.1.0"


def test_baseline_configuration_contract() -> None:
    configuration_path = Path(
        "configs/baseline.yaml"
    )

    configuration = yaml.safe_load(
        configuration_path.read_text(
            encoding="utf-8"
        )
    )

    assert configuration["project"]["random_seed"] == 42

    assert (
        configuration["pca"][
            "explained_variance_target"
        ]
        == 0.95
    )

    assert (
        configuration["pca"]["eigensolver"]
        == "numpy.linalg.eigh"
    )

    assert (
        configuration["threshold"]["quantile"]
        == 0.99
    )

    assert (
        configuration["data"][
            "test_labels_hidden_until_evaluation"
        ]
        is True
    )


def test_required_phase_one_documents_exist() -> None:
    required_paths = [
        Path("README.md"),
        Path("docs/research_protocol.md"),
        Path("docs/mathematical_foundation.md"),
        Path("docs/critical_reasoning.md"),
        Path("prompts/phase_01_foundation.md"),
        Path("agent_trace/phase_01.md"),
    ]

    for required_path in required_paths:
        assert required_path.is_file()
        assert required_path.stat().st_size > 0