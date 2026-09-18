import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "run_phase_10r.py"
SPEC = importlib.util.spec_from_file_location("phase10r", MODULE_PATH)
phase10r = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(phase10r)


def test_frequency_is_membership_vector_with_other_bucket():
    vector = phase10r.frequency("CC?", "C")
    assert vector == [2 / 3, 1 / 3]
    assert sum(vector) == 1.0


def test_top_order_breaks_exact_ties_by_train_index():
    assert phase10r.top_order(np.array([0.8, 0.5, 0.8, 0.2]), 2).tolist() == [0, 2]


def test_probability_boundary_roundoff_is_clipped_but_larger_error_stops():
    assert phase10r.bounded_probability(1.0 + 5e-13) == 1.0
    try:
        phase10r.bounded_probability(1.01)
    except ValueError:
        pass
    else:
        raise AssertionError("material probability violation must fail")


def test_predictions_cover_all_methods_and_fixed_k_values():
    train_x = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]] * 9)
    train_y = np.array([1, 1, 0] * 9)
    result = phase10r.predictions(train_x, train_y, np.array([[1.0, 0.0]]))
    assert set(result) == set(phase10r.METHODS)
    assert all(0.0 <= result[method][3][0] <= 1.0 for method in phase10r.METHODS)
