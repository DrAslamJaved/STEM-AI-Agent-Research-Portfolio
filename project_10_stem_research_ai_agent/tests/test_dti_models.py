import numpy as np

from stem_research_agent.dti_features import FEATURE_NAMES, dti_feature_vector, feature_matrix
from stem_research_agent.dti_models import evaluate_dti_models


def _record(index, label):
    return {
        "drug_id": f"D{index}", "target_id": f"T{index}", "label": label,
        "smiles": "CCN" if label else "COO", "protein_sequence": "MKKKK" if label else "MAAAA",
    }


def test_features_are_finite_and_have_declared_dimension():
    vector = dti_feature_vector(_record(1, 1))
    assert len(vector) == len(FEATURE_NAMES)
    assert np.isfinite(vector).all()
    assert feature_matrix([_record(1, 1), _record(2, 0)]).shape == (2, len(FEATURE_NAMES))


def test_models_produce_metric_records_for_binary_training_data():
    train = [_record(index, index % 2) for index in range(1, 13)]
    test = [_record(index, index % 2) for index in range(13, 19)]
    outcomes = evaluate_dti_models(train, test, seed=7)
    assert set(outcomes) == {"prevalence", "logistic_regression", "random_forest"}
    assert all("roc_auc" in item and "pr_auc" in item and "brier_score" in item for item in outcomes.values())


def test_models_require_both_training_classes():
    train = [_record(index, 0) for index in range(1, 5)]
    test = [_record(index, index % 2) for index in range(5, 9)]
    try:
        evaluate_dti_models(train, test)
    except ValueError as exc:
        assert "both binary classes" in str(exc)
    else:
        raise AssertionError("Expected single-class training data to be rejected.")
