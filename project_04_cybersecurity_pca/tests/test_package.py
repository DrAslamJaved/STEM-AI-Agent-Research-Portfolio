"""Tests for package exports and reproducibility artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from cyber_pca import (
    ATTACK_TYPES,
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    ManualPCA,
    __version__,
    generate_synthetic_network_data,
    run_math_validation,
    RawDataSplits,
    StandardizedDataSplits,
    split_normal_calibration_test,
    standardize_splits,
)


def test_public_package_exports() -> None:
    assert ManualPCA.__name__ == "ManualPCA"
    assert callable(run_math_validation)
    assert __version__ == "0.1.0"
    assert len(FEATURE_COLUMNS) == 10

    assert ATTACK_TYPES == (
        "port_scan",
        "dos",
        "brute_force",
        "exfiltration",
    )

    assert len(OUTPUT_COLUMNS) == 13
    assert callable(generate_synthetic_network_data)
    assert RawDataSplits.__name__ == (
        "RawDataSplits"
    )

    assert StandardizedDataSplits.__name__ == (
        "StandardizedDataSplits"
    )

    assert callable(
        split_normal_calibration_test
    )

    assert callable(standardize_splits)

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

    synthetic_configuration = configuration[
        "synthetic_data"
    ]

    assert (
        synthetic_configuration["normal_observations"]
        == 4000
    )

    assert (
        synthetic_configuration[
            "attack_observations_per_type"
        ]
        == 250
    )

    assert synthetic_configuration["attack_types"] == [
        "port_scan",
        "dos",
        "brute_force",
        "exfiltration",
    ]

    assert synthetic_configuration["shuffle"] is True

    assert synthetic_configuration["feature_columns"] == [
        "duration_ms",
        "packets_in",
        "packets_out",
        "bytes_in",
        "bytes_out",
        "syn_count",
        "ack_count",
        "connection_rate",
        "unique_dest_ports",
        "failed_logins",
    ]

    assert (
        synthetic_configuration["label_column"]
        == "is_anomaly"
    )

    assert (
        synthetic_configuration["scenario_column"]
        == "scenario"
    )

    assert (
        synthetic_configuration["id_column"]
        == "flow_id"
    )

    preprocessing_configuration = configuration[
        "preprocessing"
    ]

    assert (
        preprocessing_configuration[
            "normal_fit_fraction"
        ]
        == 0.60
    )

    assert (
        preprocessing_configuration[
            "normal_calibration_fraction"
        ]
        == 0.20
    )

    assert (
        preprocessing_configuration[
            "normal_test_fraction"
        ]
        == 0.20
    )

    assert (
        preprocessing_configuration["random_seed"]
        == 42
    )

    assert (
        preprocessing_configuration[
            "attack_assignment"
        ]
        == "test_only"
    )

    assert (
        preprocessing_configuration[
            "scaler_fit_split"
        ]
        == "normal_fit_only"
    )

    assert (
        preprocessing_configuration["standardizer"]
        == "sklearn.preprocessing.StandardScaler"
    )

    assert (
        preprocessing_configuration[
            "scaler_variance_ddof"
        ]
        == 0
    )

    assert preprocessing_configuration[
        "excluded_columns"
    ] == [
        "flow_id",
        "is_anomaly",
        "scenario",
    ]

    assert (
        preprocessing_configuration[
            "reject_zero_variance"
        ]
        is True
    )

def test_required_project_documents_exist() -> None:
    required_paths = [
        Path("README.md"),
        Path("docs/preprocessing_contract.md"),
        Path("docs/synthetic_data_contract.md"),
        Path("prompts/phase_02_synthetic_data.md"),
        Path("agent_trace/phase_02.md"),
        Path("docs/synthetic_data_contract.md"),
        Path("docs/research_protocol.md"),
        Path("docs/mathematical_foundation.md"),
        Path("docs/critical_reasoning.md"),
        Path("prompts/phase_01_foundation.md"),
        Path("agent_trace/phase_01.md"),
        Path("prompts/phase_03_preprocessing.md"),
        Path("agent_trace/phase_03.md"),
    ]

    for required_path in required_paths:
        assert required_path.is_file()
        assert required_path.stat().st_size > 0

def test_markdown_code_fences_are_closed() -> None:
    markdown_paths = [
        Path("README.md"),
        *Path("docs").rglob("*.md"),
        *Path("prompts").rglob("*.md"),
        *Path("agent_trace").rglob("*.md"),
    ]

    for markdown_path in markdown_paths:
        markdown_text = markdown_path.read_text(
            encoding="utf-8"
        )
        fence_count = markdown_text.count("```")

        assert fence_count % 2 == 0, (
            "Unbalanced fenced code block: "
            f"{markdown_path} ({fence_count} fences)"
        )

    readme_text = Path("README.md").read_text(
        encoding="utf-8"
    )

    assert readme_text.count("```") == 4

    assert (
        "AI agent reasoning assessment\n```\n\n"
        "## Phase 1 verification evidence"
        in readme_text
    )

    assert (
        "agent_trace/phase_01.md\n```\n\n"
        "These results validate"
        in readme_text
    )

def test_text_artifacts_end_with_newline() -> None:
    text_paths = [
        Path(".gitignore"),
        Path("README.md"),
        Path("pyproject.toml"),
        *Path("configs").rglob("*.yaml"),
        *Path("docs").rglob("*.md"),
        *Path("prompts").rglob("*.md"),
        *Path("agent_trace").rglob("*.md"),
        *Path("src").rglob("*.py"),
        *Path("tests").rglob("*.py"),
    ]

    for text_path in text_paths:
        assert text_path.read_bytes().endswith(
            b"\n"
        ), f"Missing final newline: {text_path}"
