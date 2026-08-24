import pytest

from src.statistics import calculate_mean


# ============================================================
# T01 — Basic functional test
# ============================================================

def test_mean_basic():
    assert calculate_mean([2, 4, 6]) == 4


# ============================================================
# T02 — Ordinary dataset
# ============================================================

def test_mean_ordinary_dataset():
    assert calculate_mean([1, 2, 3, 4, 5]) == 3


# ============================================================
# T03 — Smallest valid dataset
# Boundary test
# ============================================================

def test_mean_single_value():
    assert calculate_mean([5]) == 5


# ============================================================
# T04 — All-zero dataset
# Boundary/domain test
# ============================================================

def test_mean_all_zeros():
    assert calculate_mean([0, 0, 0]) == 0


# ============================================================
# T05 — Negative values
# ============================================================

def test_mean_negative_values():
    assert calculate_mean([-2, 2]) == 0


# ============================================================
# T06 — Mixed positive and negative values
# ============================================================

def test_mean_mixed_signs():
    assert calculate_mean([-5, 0, 5]) == 0


# ============================================================
# T07 — Decimal values
# Numerical test
# ============================================================

def test_mean_decimal_values():
    assert calculate_mean([1.5, 2.5, 3.5]) == 2.5


# ============================================================
# T08 — Large numerical values
# ============================================================

def test_mean_large_values():
    assert calculate_mean([10**9, 10**9]) == 10**9


# ============================================================
# T09 — Empty dataset
# Negative/boundary test
# ============================================================

def test_mean_empty_dataset():
    with pytest.raises(ValueError):
        calculate_mean([])


# ============================================================
# T10 — Non-numeric element
# Negative test
# ============================================================

def test_mean_invalid_data():
    with pytest.raises(TypeError):
        calculate_mean([2, "hello", 4])


# ============================================================
# T11 — None input
# Invalid input test
# ============================================================

def test_mean_none_input():
    with pytest.raises((TypeError, ValueError)):
        calculate_mean(None)


# ============================================================
# T12 — String input
# Invalid input test
# ============================================================

def test_mean_string_input():
    with pytest.raises((TypeError, ValueError)):
        calculate_mean("246")


# ============================================================
# T13 — Tuple input
# Compatibility test
# ============================================================

def test_mean_tuple():
    assert calculate_mean((2, 4, 6)) == 4

# ============================================================
# T14 — Translation invariance
# Mathematical property
# ============================================================

def test_mean_translation_invariance():
    data = [2, 4, 6]
    constant = 10

    original_mean = calculate_mean(data)
    translated_data = [x + constant for x in data]

    assert calculate_mean(translated_data) == original_mean + constant


# ============================================================
# T15 — Scaling property
# Mathematical property
# ============================================================

def test_mean_scaling_property():
    data = [2, 4, 6]
    scale = 3

    original_mean = calculate_mean(data)
    scaled_data = [scale * x for x in data]

    assert calculate_mean(scaled_data) == scale * original_mean


# ============================================================
# T16 — Constant dataset
# Mathematical property
# ============================================================

def test_mean_constant_dataset():
    data = [7, 7, 7, 7]

    assert calculate_mean(data) == 7


# ============================================================
# T17 — Permutation invariance
# Mathematical property
# ============================================================

def test_mean_permutation_invariance():
    data_1 = [2, 4, 6]
    data_2 = [6, 2, 4]

    assert calculate_mean(data_1) == calculate_mean(data_2)

# ============================================================
# T19 — Floating-point numerical validation
# ============================================================

def test_mean_fractional_result():
    result = calculate_mean([0, 1, 0])

    assert result == pytest.approx(1 / 3, rel=1e-12, abs=1e-12)