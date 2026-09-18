"""Hypothesis-based validation of the approved fuzzy-similarity contract.

These generative tests provide computational evidence. They do not replace
the analytic proofs or establish novelty.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from fuzzy_similarity import (
    Parameters,
    fuzzy_cosine,
    fuzzy_dice,
    fuzzy_jaccard,
    rational_similarity,
    similarity_components,
)

TOLERANCE = 1.0e-10

MEMBERSHIP = st.integers(min_value=0, max_value=1000).map(
    lambda value: value / 1000.0
)
COEFFICIENT = st.integers(min_value=0, max_value=200).map(
    lambda value: value / 10.0
)


@st.composite
def fuzzy_vectors(draw):
    size = draw(st.integers(min_value=1, max_value=12))
    return draw(
        st.lists(MEMBERSHIP, min_size=size, max_size=size)
    )


@st.composite
def fuzzy_pairs(draw):
    size = draw(st.integers(min_value=1, max_value=12))
    u = draw(st.lists(MEMBERSHIP, min_size=size, max_size=size))
    v = draw(st.lists(MEMBERSHIP, min_size=size, max_size=size))
    return u, v


@st.composite
def approved_parameters(draw):
    a = draw(COEFFICIENT)
    b = draw(COEFFICIENT)
    c = draw(COEFFICIENT)
    d = draw(COEFFICIENT)
    e = draw(COEFFICIENT)

    b_gap = draw(COEFFICIENT)
    c_gap = draw(COEFFICIENT)
    e_gap = draw(COEFFICIENT)

    return Parameters(
        a=a,
        b=b,
        c=c,
        d=d,
        e=e,
        a_prime=a,
        b_prime=b + b_gap,
        c_prime=c + c_gap,
        d_prime=d,
        e_prime=e + e_gap,
    )


def assert_unit_interval(value: float) -> None:
    assert math.isfinite(value)
    assert -TOLERANCE <= value <= 1.0 + TOLERANCE


@settings(max_examples=200, deadline=None, derandomize=True)
@given(pair=fuzzy_pairs())
def test_components_are_nonnegative_and_decompose_universe(pair):
    u, v = pair
    components = similarity_components(u, v)

    assert components.alpha >= 0.0
    assert components.beta >= 0.0
    assert components.delta >= 0.0
    assert components.gamma >= 0.0

    total = (
        components.alpha
        + components.beta
        + components.delta
        + components.gamma
    )
    assert total == pytest.approx(
        float(len(u)), rel=0.0, abs=TOLERANCE
    )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(pair=fuzzy_pairs())
def test_components_are_symmetric(pair):
    u, v = pair
    forward = similarity_components(u, v)
    reverse = similarity_components(v, u)

    for attribute in ("alpha", "beta", "delta", "gamma"):
        assert getattr(forward, attribute) == pytest.approx(
            getattr(reverse, attribute),
            rel=0.0,
            abs=TOLERANCE,
        )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(pair=fuzzy_pairs(), parameters=approved_parameters())
def test_rational_similarity_is_bounded(pair, parameters):
    u, v = pair
    assert_unit_interval(rational_similarity(u, v, parameters))


@settings(max_examples=200, deadline=None, derandomize=True)
@given(pair=fuzzy_pairs(), parameters=approved_parameters())
def test_rational_similarity_is_symmetric(pair, parameters):
    u, v = pair

    assert rational_similarity(u, v, parameters) == pytest.approx(
        rational_similarity(v, u, parameters),
        rel=0.0,
        abs=TOLERANCE,
    )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(vector=fuzzy_vectors(), parameters=approved_parameters())
def test_rational_similarity_is_reflexive(vector, parameters):
    assert rational_similarity(vector, vector, parameters) == pytest.approx(
        1.0, rel=0.0, abs=TOLERANCE
    )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    pair=fuzzy_pairs(),
    parameters=approved_parameters(),
    data=st.data(),
)
def test_rational_similarity_is_permutation_invariant(
    pair, parameters, data
):
    u, v = pair
    permutation = data.draw(
        st.permutations(tuple(range(len(u)))),
        label="simultaneous permutation",
    )

    permuted_u = [u[index] for index in permutation]
    permuted_v = [v[index] for index in permutation]

    assert rational_similarity(
        u, v, parameters
    ) == pytest.approx(
        rational_similarity(permuted_u, permuted_v, parameters),
        rel=0.0,
        abs=TOLERANCE,
    )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(pair=fuzzy_pairs())
def test_baselines_are_bounded_and_symmetric(pair):
    u, v = pair

    for measure in (fuzzy_jaccard, fuzzy_dice, fuzzy_cosine):
        forward = measure(u, v)
        reverse = measure(v, u)

        assert_unit_interval(forward)
        assert forward == pytest.approx(
            reverse, rel=0.0, abs=TOLERANCE
        )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(vector=fuzzy_vectors())
def test_baseline_reflexivity_and_zero_conventions(vector):
    assert fuzzy_jaccard(vector, vector) == pytest.approx(
        1.0, rel=0.0, abs=TOLERANCE
    )

    if any(value != 0.0 for value in vector):
        assert fuzzy_dice(vector, vector) == pytest.approx(1.0)
        assert fuzzy_cosine(vector, vector) == pytest.approx(1.0)
    else:
        assert fuzzy_dice(vector, vector) == 0.0
        assert fuzzy_cosine(vector, vector) == 0.0


@pytest.mark.parametrize("bad_input", [None, 1.0])
def test_non_iterable_memberships_are_rejected(bad_input):
    with pytest.raises(TypeError, match="iterable"):
        rational_similarity(
            bad_input, [0.0], Parameters.jaccard_control()
        )


def test_empty_membership_vectors_are_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        similarity_components([], [])


def test_non_real_parameters_are_rejected():
    with pytest.raises(TypeError, match="real numbers"):
        Parameters(1, 0, 0, 0, 0, 1, 0, 0, 0, object())


@pytest.mark.parametrize(
    "bad_parameter",
    [-1.0, math.nan, math.inf, -math.inf],
)
def test_negative_or_nonfinite_parameters_are_rejected(bad_parameter):
    with pytest.raises(ValueError):
        Parameters(
            bad_parameter, 0, 0, 0, 0,
            bad_parameter, 0, 0, 0, 0,
        )


def test_non_parameter_argument_is_rejected():
    with pytest.raises(TypeError, match="Parameters instance"):
        rational_similarity([0.0], [0.0], object())


@pytest.mark.parametrize(
    "measure",
    [similarity_components, fuzzy_dice, fuzzy_cosine],
)
def test_two_argument_measures_reject_unequal_lengths(measure):
    with pytest.raises(ValueError, match="equal length"):
        measure([0.0], [0.0, 1.0])


def test_rational_similarity_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        rational_similarity(
            [0.0],
            [0.0, 1.0],
            Parameters.jaccard_control(),
        )


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    invalid_membership=st.one_of(
        st.integers(max_value=-1),
        st.integers(min_value=2, max_value=1000),
    )
)
def test_generated_out_of_range_memberships_are_rejected(
    invalid_membership,
):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rational_similarity(
            [invalid_membership],
            [0.0],
            Parameters.jaccard_control(),
        )


def test_cosine_zero_convention_is_symmetric():
    zero = [0.0, 0.0]
    nonzero = [0.2, 0.8]

    assert fuzzy_cosine(zero, nonzero) == 0.0
    assert fuzzy_cosine(nonzero, zero) == 0.0