"""Core functions for the human-approved Phase 04/05 similarity subfamily.

This module implements a *formulation*, not a novelty claim.  The analytic
results recorded in Phase 05 apply only under ``a == a_prime``,
``d == d_prime``, and coordinatewise coefficient dominance.  Test execution
checks that this implementation follows that contract; it is not a proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterable, Sequence


def _as_memberships(values: Iterable[Real], name: str) -> tuple[float, ...]:
    """Convert and validate one finite fuzzy-membership vector."""
    try:
        vector = tuple(float(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real memberships") from exc

    if not vector:
        raise ValueError(f"{name} must be non-empty")
    for index, value in enumerate(vector):
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name}[{index}] must be a finite membership in [0, 1], got {value!r}"
            )
    return vector


@dataclass(frozen=True)
class Parameters:
    """Parameters of the human-approved reflexive rational subfamily.

    The implemented contract is

    ``a == a_prime``, ``d == d_prime``,
    ``0 <= b <= b_prime``, ``0 <= c <= c_prime``, and
    ``0 <= e <= e_prime``.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    a_prime: float
    b_prime: float
    c_prime: float
    d_prime: float
    e_prime: float

    def __post_init__(self) -> None:
        names = (
            "a", "b", "c", "d", "e", "a_prime", "b_prime", "c_prime",
            "d_prime", "e_prime",
        )
        values = tuple(getattr(self, name) for name in names)
        if any(not isinstance(value, Real) for value in values):
            raise TypeError("all parameters must be real numbers")
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("all parameters must be finite and non-negative")
        if self.a != self.a_prime or self.d != self.d_prime:
            raise ValueError(
                "the approved reflexive subfamily requires a == a_prime and d == d_prime"
            )
        if self.b > self.b_prime or self.c > self.c_prime or self.e > self.e_prime:
            raise ValueError(
                "the approved bounded subfamily requires b <= b_prime, "
                "c <= c_prime, and e <= e_prime"
            )

    @classmethod
    def jaccard_control(cls) -> "Parameters":
        """Return the approved Phase 04 min-max fuzzy Jaccard control setting."""
        return cls(
            a=1.0, b=0.0, c=0.0, d=0.0, e=0.0,
            a_prime=1.0, b_prime=1.0, c_prime=1.0, d_prime=0.0, e_prime=0.0,
        )


@dataclass(frozen=True)
class SimilarityComponents:
    """Cardinality components for the approved residual-difference convention."""

    alpha: float
    beta: float
    delta: float
    gamma: float


def similarity_components(
    u: Iterable[Real], v: Iterable[Real]
) -> SimilarityComponents:
    """Compute alpha, beta, delta, gamma with min/max/residual operations.

    ``|U \\ V|`` uses the approved residual membership
    ``max(mu_U - mu_V, 0)`` and all cardinalities are sigma-counts.
    """
    u_vector = _as_memberships(u, "u")
    v_vector = _as_memberships(v, "v")
    if len(u_vector) != len(v_vector):
        raise ValueError("u and v must have equal length")

    u_minus_v = sum(max(left - right, 0.0) for left, right in zip(u_vector, v_vector))
    v_minus_u = sum(max(right - left, 0.0) for left, right in zip(u_vector, v_vector))
    intersection = sum(min(left, right) for left, right in zip(u_vector, v_vector))
    co_union = sum(1.0 - max(left, right) for left, right in zip(u_vector, v_vector))
    return SimilarityComponents(
        alpha=max(u_minus_v, v_minus_u),
        beta=min(u_minus_v, v_minus_u),
        delta=intersection,
        gamma=co_union,
    )


def rational_similarity(
    u: Iterable[Real], v: Iterable[Real], parameters: Parameters
) -> float:
    """Evaluate the approved rational similarity with an explicit zero rule.

    For a zero denominator, return 1.0 for equal inputs and 0.0 otherwise.
    """
    if not isinstance(parameters, Parameters):
        raise TypeError("parameters must be a Parameters instance")
    u_vector = _as_memberships(u, "u")
    v_vector = _as_memberships(v, "v")
    if len(u_vector) != len(v_vector):
        raise ValueError("u and v must have equal length")

    comp = similarity_components(u_vector, v_vector)
    numerator = (
        parameters.a * comp.delta**2
        + parameters.e * comp.alpha * comp.beta
        + comp.delta * (parameters.b * comp.alpha + parameters.c * comp.beta + parameters.d * comp.gamma)
    )
    denominator = (
        parameters.a_prime * comp.delta**2
        + parameters.e_prime * comp.alpha * comp.beta
        + comp.delta * (
            parameters.b_prime * comp.alpha
            + parameters.c_prime * comp.beta
            + parameters.d_prime * comp.gamma
        )
    )
    if denominator == 0.0:
        return 1.0 if u_vector == v_vector else 0.0
    return numerator / denominator


def fuzzy_jaccard(u: Sequence[Real], v: Sequence[Real]) -> float:
    """Evaluate the min-max fuzzy Jaccard control through the approved family."""
    return rational_similarity(u, v, Parameters.jaccard_control())


def fuzzy_dice(u: Iterable[Real], v: Iterable[Real]) -> float:
    """Return min-max fuzzy Dice similarity; return 0.0 for two zero vectors.

    The explicit zero convention means that this comparison baseline is not
    reflexive at the all-zero vector. This is intentional and tested.
    """
    u_vector = _as_memberships(u, "u")
    v_vector = _as_memberships(v, "v")
    if len(u_vector) != len(v_vector):
        raise ValueError("u and v must have equal length")
    denominator = sum(u_vector) + sum(v_vector)
    if denominator == 0.0:
        return 0.0
    return 2.0 * sum(min(left, right) for left, right in zip(u_vector, v_vector)) / denominator


def fuzzy_cosine(u: Iterable[Real], v: Iterable[Real]) -> float:
    """Return cosine similarity; return 0.0 when either vector has zero norm.

    The explicit zero-norm convention means that this comparison baseline is
    not reflexive at the all-zero vector. This is intentional and tested.
    """
    u_vector = _as_memberships(u, "u")
    v_vector = _as_memberships(v, "v")
    if len(u_vector) != len(v_vector):
        raise ValueError("u and v must have equal length")
    norm_u_squared = sum(value * value for value in u_vector)
    norm_v_squared = sum(value * value for value in v_vector)
    if norm_u_squared == 0.0 or norm_v_squared == 0.0:
        return 0.0
    dot_product = sum(left * right for left, right in zip(u_vector, v_vector))
    return dot_product / (norm_u_squared * norm_v_squared) ** 0.5
