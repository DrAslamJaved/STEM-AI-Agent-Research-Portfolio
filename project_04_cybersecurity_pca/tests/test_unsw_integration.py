"""Integration test for locally available UNSW-NB15 files."""

from pathlib import Path

import pytest

from cyber_pca.unsw_data import (
    build_unsw_nb15_manifest,
    load_unsw_nb15,
    resolve_unsw_nb15_paths,
    validate_unsw_nb15,
)
from cyber_pca.unsw_preprocessing import (
    build_unsw_preprocessing_evidence,
    split_unsw_normal_calibration_test,
    standardize_unsw_splits,
)


RAW_PATHS = resolve_unsw_nb15_paths(
    Path("data/raw")
)
RAW_FILES_AVAILABLE = all(
    path.is_file()
    for path in (
        RAW_PATHS.training,
        RAW_PATHS.testing,
        RAW_PATHS.feature_descriptions,
    )
)


@pytest.mark.skipif(
    not RAW_FILES_AVAILABLE,
    reason=(
        "Official ignored UNSW-NB15 raw files "
        "are not locally available."
    ),
)
def test_official_unsw_nb15_pipeline() -> None:
    data = load_unsw_nb15(RAW_PATHS)
    validate_unsw_nb15(data)

    manifest = build_unsw_nb15_manifest(
        RAW_PATHS,
        data,
    )

    assert manifest["files"]["training"][
        "sha256"
    ] == (
        "bec7dd5ec88dc2a0ccc7a07879d33839"
        "5ed7421750f675fd0339e07dfe0648fa"
    )
    assert manifest["files"]["testing"][
        "sha256"
    ] == (
        "734fe6642edf758f7c94d7d9149426b4"
        "9d202fe8e7bf0bef47392489c3c0a559"
    )
    assert manifest["files"][
        "feature_descriptions"
    ]["sha256"] == (
        "c55f19cceebb6360dc50f44f8a5f246"
        "ccefbcf8a6c604ac1ad46e643869cafce"
    )

    raw_splits = (
        split_unsw_normal_calibration_test(
            data,
            normal_fit_fraction=0.75,
            random_seed=42,
        )
    )
    standardized = standardize_unsw_splits(
        raw_splits
    )
    evidence = (
        build_unsw_preprocessing_evidence(
            raw_splits,
            standardized,
        )
    )

    assert standardized.normal_fit.shape == (
        42000,
        64,
    )
    assert (
        standardized.normal_calibration.shape
        == (14000, 64)
    )
    assert standardized.test.shape == (
        82332,
        64,
    )

    assert evidence["partitions"][
        "normal_fit"
    ]["id_sha256"] == (
        "b8ec94affc717d98f1c5d2db12e1d830"
        "4f82f9e761590ec66c2b18a3c827cc68"
    )
    assert evidence["features"][
        "feature_name_sha256"
    ] == (
        "f47ce5e1981c3a4eae3d51cd45c20f50"
        "d02f764714a57f3ec57a15c7b4c62bad"
    )
    assert evidence["standardization"][
        "scaler_state_sha256"
    ] == (
        "610ea7a2e37f669878a10a34d18e63b9"
        "3b36fa4a0251e92399b5d680ab841685"
    )
    assert evidence["standardization"][
        "maximum_absolute_fitting_mean"
    ] <= 1.0e-12
    assert evidence["standardization"][
        "maximum_fitting_std_error"
    ] <= 2.0e-12

    assert evidence["guards"] == {
        "training_attacks_excluded": True,
        "encoder_fit_normal_only": True,
        "scaler_fit_normal_only": True,
        "test_labels_used_for_fitting": False,
        "pca_fitted": False,
        "threshold_calibrated": False,
    }
