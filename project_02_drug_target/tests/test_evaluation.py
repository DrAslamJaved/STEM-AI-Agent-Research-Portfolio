import pytest

from src.models.evaluation import (
    EvaluationError,
    evaluate_binary_classification,
)


def test_known_scores_produce_expected_binary_metrics() -> None:
    metrics = evaluate_binary_classification(
        [0, 0, 1, 1],
        [0.1, 0.6, 0.4, 0.9],
    )

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.average_precision == pytest.approx(5 / 6)

    assert (
        metrics.true_negative,
        metrics.false_positive,
        metrics.false_negative,
        metrics.true_positive,
    ) == (1, 1, 1, 1)


def test_threshold_uses_greater_than_or_equal_to() -> None:
    metrics = evaluate_binary_classification(
        [0, 1],
        [0.5, 0.5],
    )

    assert metrics.false_positive == 1
    assert metrics.true_positive == 1


def test_no_predicted_positives_has_zero_precision_without_warning() -> None:
    metrics = evaluate_binary_classification(
        [0, 1],
        [0.2, 0.3],
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


@pytest.mark.parametrize(
    ("y_true", "y_score", "message"),
    [
        ([0, 0], [0.1, 0.2], "both positive and negative"),
        ([0, 2], [0.1, 0.2], "only 0 and 1"),
        ([0, 1], [0.1], "same length"),
        ([0, 1], [0.1, 1.2], "between 0 and 1"),
    ],
)
def test_invalid_evaluation_inputs_fail_clearly(
    y_true,
    y_score,
    message,
) -> None:
    with pytest.raises(EvaluationError, match=message):
        evaluate_binary_classification(y_true, y_score)


def test_invalid_threshold_fails_clearly() -> None:
    with pytest.raises(EvaluationError, match="between 0 and 1"):
        evaluate_binary_classification(
            [0, 1],
            [0.1, 0.9],
            decision_threshold=1.1,
        )