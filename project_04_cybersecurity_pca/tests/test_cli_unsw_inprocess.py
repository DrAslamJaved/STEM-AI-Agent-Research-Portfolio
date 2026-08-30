"""In-process tests for official UNSW-NB15 CLI execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cyber_pca.cli as cli


def test_executes_unsw_evaluation_in_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_directory = tmp_path / "raw"
    output_root = tmp_path / "output"

    dataset = object()

    raw_test = object()
    raw_splits = SimpleNamespace(
        test=raw_test
    )

    standardized_splits = object()

    test_errors = object()
    test_predictions = object()

    detection_result = SimpleNamespace(
        fit_result=SimpleNamespace(
            n_components=34,
            achieved_explained_variance=(
                0.9521414327676875
            ),
        ),
        reconstruction_errors=(
            SimpleNamespace(
                test=test_errors
            )
        ),
        threshold_result=SimpleNamespace(
            threshold=0.4923769885740442
        ),
        test_predictions=test_predictions,
    )

    evaluation_data = object()

    binary_result = SimpleNamespace(
        confusion_matrix=(
            (35974, 1026),
            (42977, 2355),
        ),
        precision=0.6965394853593612,
        recall=0.05195005735462808,
        f1=0.09668876891178946,
        false_positive_rate=(
            0.02772972972972973
        ),
        false_negative_rate=(
            0.9480499426453719
        ),
    )

    attack_category_result = object()

    artifacts = SimpleNamespace(
        summary_json=(
            output_root
            / "results"
            / "unsw_nb15_evaluation.json"
        )
    )

    calls: list[
        tuple[str, object]
    ] = []

    def fake_load(
        source: Path,
    ) -> object:
        calls.append(("load", source))
        assert source == raw_directory
        return dataset

    def fake_split(
        loaded: object,
    ) -> object:
        calls.append(("split", loaded))
        assert loaded is dataset
        return raw_splits

    def fake_standardize(
        splits: object,
    ) -> object:
        calls.append(
            ("standardize", splits)
        )
        assert splits is raw_splits
        return standardized_splits

    def fake_detect(
        splits: object,
    ) -> object:
        calls.append(("detect", splits))
        assert splits is standardized_splits
        return detection_result

    def fake_align(
        received_raw_test: object,
        received_errors: object,
        received_predictions: object,
    ) -> object:
        calls.append(
            (
                "align",
                received_raw_test,
            )
        )
        assert received_raw_test is raw_test
        assert received_errors is test_errors
        assert (
            received_predictions
            is test_predictions
        )
        return evaluation_data

    def fake_binary(
        received: object,
    ) -> object:
        calls.append(("binary", received))
        assert received is evaluation_data
        return binary_result

    def fake_categories(
        received: object,
    ) -> object:
        calls.append(
            ("categories", received)
        )
        assert received is evaluation_data
        return attack_category_result

    def fake_write(
        received_evaluation: object,
        received_detection: object,
        received_binary: object,
        received_categories: object,
        *,
        output_root: Path,
        dpi: int,
    ) -> object:
        calls.append(
            ("write", output_root)
        )
        assert (
            received_evaluation
            is evaluation_data
        )
        assert (
            received_detection
            is detection_result
        )
        assert (
            received_binary
            is binary_result
        )
        assert (
            received_categories
            is attack_category_result
        )
        assert output_root == (
            tmp_path / "output"
        )
        assert dpi == 150
        return artifacts

    monkeypatch.setattr(
        cli,
        "load_unsw_nb15",
        fake_load,
    )
    monkeypatch.setattr(
        cli,
        "split_unsw_normal_calibration_test",
        fake_split,
    )
    monkeypatch.setattr(
        cli,
        "standardize_unsw_splits",
        fake_standardize,
    )
    monkeypatch.setattr(
        cli,
        "run_unsw_detection",
        fake_detect,
    )
    monkeypatch.setattr(
        cli,
        "align_unsw_evaluation_data",
        fake_align,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_binary_predictions",
        fake_binary,
    )
    monkeypatch.setattr(
        cli,
        "evaluate_unsw_attack_categories",
        fake_categories,
    )
    monkeypatch.setattr(
        cli,
        "write_unsw_evaluation_artifacts",
        fake_write,
    )

    exit_code = (
        cli._execute_unsw_evaluation(
            raw_directory,
            output_root,
            150,
        )
    )

    output = capsys.readouterr().out

    assert exit_code == 0

    assert [
        call[0]
        for call in calls
    ] == [
        "load",
        "split",
        "standardize",
        "detect",
        "align",
        "binary",
        "categories",
        "write",
    ]

    assert (
        "Official UNSW-NB15 evaluation: "
        "PASSED"
    ) in output
    assert "- selected components: 34" in output
    assert (
        "- confusion matrix: "
        "((35974, 1026), (42977, 2355))"
    ) in output
    assert "- precision:" in output
    assert "- recall:" in output
    assert "- f1:" in output
    assert (
        "unsw_nb15_evaluation.json"
        in output
    )

    assert not output_root.exists()
