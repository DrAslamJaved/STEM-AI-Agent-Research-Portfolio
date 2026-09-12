"""Unit, boundary, negative, and deterministic randomized property tests.

These tests validate the implementation against the approved contract.  They
do not replace the Phase 05 analytic proofs or establish any new theorem.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzy_similarity import (  # noqa: E402
    Parameters,
    fuzzy_cosine,
    fuzzy_dice,
    fuzzy_jaccard,
    rational_similarity,
    similarity_components,
)


class TestComponents(unittest.TestCase):
    def test_residual_components_decompose_universe_cardinality(self) -> None:
        components = similarity_components([1.0, 0.25, 0.0], [0.5, 0.75, 0.0])
        self.assertAlmostEqual(components.alpha, 0.5)
        self.assertAlmostEqual(components.beta, 0.5)
        self.assertAlmostEqual(components.delta, 0.75)
        self.assertAlmostEqual(components.gamma, 1.25)
        self.assertAlmostEqual(
            components.alpha + components.beta + components.delta + components.gamma, 3.0
        )

    def test_components_are_symmetric(self) -> None:
        self.assertEqual(
            similarity_components([0.1, 1.0], [0.8, 0.4]),
            similarity_components([0.8, 0.4], [0.1, 1.0]),
        )


class TestSimilarity(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = Parameters(
            a=1.0, b=0.25, c=0.5, d=0.75, e=0.2,
            a_prime=1.0, b_prime=0.5, c_prime=0.75, d_prime=0.75, e_prime=0.8,
        )

    def test_jaccard_control_known_value(self) -> None:
        self.assertAlmostEqual(fuzzy_jaccard([1.0, 0.0], [1.0, 1.0]), 0.5)

    def test_dice_known_value_and_symmetry(self) -> None:
        self.assertAlmostEqual(fuzzy_dice([1.0, 0.0], [1.0, 1.0]), 2.0 / 3.0)
        self.assertAlmostEqual(fuzzy_dice([0.2, 1.0], [0.8, 0.4]), fuzzy_dice([0.8, 0.4], [0.2, 1.0]))

    def test_cosine_known_value_and_symmetry(self) -> None:
        self.assertAlmostEqual(fuzzy_cosine([1.0, 0.0], [1.0, 1.0]), 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(fuzzy_cosine([0.2, 1.0], [0.8, 0.4]), fuzzy_cosine([0.8, 0.4], [0.2, 1.0]))

    def test_baseline_zero_conventions_are_explicit(self) -> None:
        self.assertEqual(fuzzy_dice([0.0, 0.0], [0.0, 0.0]), 0.0)
        self.assertEqual(fuzzy_cosine([0.0, 0.0], [0.0, 0.0]), 0.0)
        self.assertEqual(fuzzy_cosine([0.0, 0.0], [0.2, 0.8]), 0.0)

    def test_reflexivity(self) -> None:
        self.assertEqual(rational_similarity([0.1, 0.4, 1.0], [0.1, 0.4, 1.0], self.parameters), 1.0)

    def test_symmetry(self) -> None:
        left = rational_similarity([0.1, 0.9], [0.8, 0.4], self.parameters)
        right = rational_similarity([0.8, 0.4], [0.1, 0.9], self.parameters)
        self.assertAlmostEqual(left, right)

    def test_all_zero_equal_boundary(self) -> None:
        zero_parameters = Parameters(
            a=0.0, b=0.0, c=0.0, d=0.0, e=0.0,
            a_prime=0.0, b_prime=0.0, c_prime=0.0, d_prime=0.0, e_prime=0.0,
        )
        self.assertEqual(rational_similarity([0.0, 0.0], [0.0, 0.0], zero_parameters), 1.0)

    def test_zero_denominator_unequal_boundary(self) -> None:
        zero_parameters = Parameters(
            a=0.0, b=0.0, c=0.0, d=0.0, e=0.0,
            a_prime=0.0, b_prime=0.0, c_prime=0.0, d_prime=0.0, e_prime=0.0,
        )
        self.assertEqual(rational_similarity([0.0, 0.0], [1.0, 1.0], zero_parameters), 0.0)

    def test_identity_is_not_assumed(self) -> None:
        equal_coefficient_parameters = Parameters(
            a=1.0, b=1.0, c=1.0, d=0.0, e=0.0,
            a_prime=1.0, b_prime=1.0, c_prime=1.0, d_prime=0.0, e_prime=0.0,
        )
        self.assertEqual(rational_similarity([1.0], [0.5], equal_coefficient_parameters), 1.0)

    def test_self_complementarity_is_not_assumed(self) -> None:
        self.assertAlmostEqual(fuzzy_jaccard([1.0, 0.0], [1.0, 1.0]), 0.5)
        self.assertEqual(fuzzy_jaccard([0.0, 1.0], [0.0, 0.0]), 0.0)


class TestNegativeInputs(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = Parameters.jaccard_control()

    def test_rejects_unequal_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            rational_similarity([0.0], [0.0, 1.0], self.parameters)

    def test_rejects_invalid_memberships(self) -> None:
        for vector in ([-0.1], [1.1], [math.nan], [math.inf]):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    rational_similarity(vector, [0.0], self.parameters)

    def test_rejects_invalid_parameter_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "a == a_prime"):
            Parameters(1, 0, 0, 0, 0, 2, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "b <= b_prime"):
            Parameters(1, 2, 0, 0, 0, 1, 1, 0, 0, 0)
        with self.assertRaises(ValueError):
            Parameters(-1, 0, 0, 0, 0, -1, 0, 0, 0, 0)


class TestRandomizedProperties(unittest.TestCase):
    """Fixed-seed randomized tests; a reproducible test, not a theorem proof."""

    def test_bounds_symmetry_and_reflexivity_on_random_vectors(self) -> None:
        rng = random.Random(20260905)
        parameter_sets = (
            Parameters.jaccard_control(),
            Parameters(1, 0.25, 0.5, 0.75, 0.2, 1, 0.5, 0.75, 0.75, 0.8),
            Parameters(0, 0, 0, 0, 0, 0, 1, 1, 0, 1),
        )
        for parameters in parameter_sets:
            for _ in range(100):
                u = [rng.random() for _ in range(5)]
                v = [rng.random() for _ in range(5)]
                uv = rational_similarity(u, v, parameters)
                vu = rational_similarity(v, u, parameters)
                self.assertGreaterEqual(uv, 0.0)
                self.assertLessEqual(uv, 1.0)
                self.assertAlmostEqual(uv, vu)
                self.assertEqual(rational_similarity(u, u, parameters), 1.0)

    def test_baseline_ranges_on_random_nonzero_vectors(self) -> None:
        rng = random.Random(20260905)
        for _ in range(100):
            u = [rng.random() for _ in range(5)]
            v = [rng.random() for _ in range(5)]
            for measure in (fuzzy_dice, fuzzy_cosine):
                self.assertGreaterEqual(measure(u, v), 0.0)
                self.assertLessEqual(measure(u, v), 1.0)


if __name__ == "__main__":
    unittest.main()
