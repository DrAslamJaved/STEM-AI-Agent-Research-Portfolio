"""Exact-rational audit for Phase 05 proof drafts.

This script checks selected finite cases using fractions.  It is audit evidence,
not a substitute for the analytic proofs in phase_05_proof_drafts.md.
"""

from fractions import Fraction as F
from itertools import product
import json


VALUES = (F(0), F(1, 2), F(1))


def cardinality(values):
    return sum(values, F(0))


def components(u, v):
    p = cardinality(tuple(max(x - y, F(0)) for x, y in zip(u, v)))
    q = cardinality(tuple(max(y - x, F(0)) for x, y in zip(u, v)))
    delta = cardinality(tuple(min(x, y) for x, y in zip(u, v)))
    gamma = cardinality(tuple(F(1) - max(x, y) for x, y in zip(u, v)))
    return max(p, q), min(p, q), delta, gamma


def similarity(u, v, theta):
    alpha, beta, delta, gamma = components(u, v)
    n = theta["a"] * delta * delta + theta["e"] * alpha * beta
    n += delta * (theta["b"] * alpha + theta["c"] * beta + theta["d"] * gamma)
    d = theta["ap"] * delta * delta + theta["ep"] * alpha * beta
    d += delta * (theta["bp"] * alpha + theta["cp"] * beta + theta["dp"] * gamma)
    if d:
        return n / d, n, d, (alpha, beta, delta, gamma)
    return (F(1) if u == v else F(0)), n, d, (alpha, beta, delta, gamma)


THETAS = (
    {"a": F(1), "b": F(0), "c": F(0), "d": F(0), "e": F(0),
     "ap": F(1), "bp": F(1), "cp": F(1), "dp": F(0), "ep": F(0)},
    {"a": F(1), "b": F(1, 4), "c": F(1, 3), "d": F(1, 2), "e": F(1, 5),
     "ap": F(1), "bp": F(3, 4), "cp": F(2, 3), "dp": F(1, 2), "ep": F(4, 5)},
    {"a": F(0), "b": F(0), "c": F(0), "d": F(0), "e": F(0),
     "ap": F(0), "bp": F(1), "cp": F(1), "dp": F(0), "ep": F(0)},
)


def audit():
    checked = 0
    for theta in THETAS:
        assert theta["a"] == theta["ap"] and theta["d"] == theta["dp"]
        assert theta["b"] <= theta["bp"] and theta["c"] <= theta["cp"]
        assert theta["e"] <= theta["ep"]
        for n in (1, 2):
            vectors = tuple(product(VALUES, repeat=n))
            for u, v in product(vectors, repeat=2):
                s, numerator, denominator, (alpha, beta, delta, gamma) = similarity(u, v, theta)
                assert delta + alpha + beta + gamma == n
                assert F(0) <= s <= F(1)
                assert s == similarity(v, u, theta)[0]
                assert similarity(u, u, theta)[0] == F(1)
                if denominator:
                    assert F(0) <= numerator <= denominator
                checked += 1

    identity_theta = {"a": F(1), "b": F(1), "c": F(0), "d": F(0), "e": F(0),
                      "ap": F(1), "bp": F(1), "cp": F(0), "dp": F(0), "ep": F(0)}
    assert similarity((F(1),), (F(1, 2),), identity_theta)[0] == F(1)

    jaccard = THETAS[0]
    assert similarity((F(1), F(0)), (F(1), F(1)), jaccard)[0] == F(1, 2)
    assert similarity((F(0), F(1)), (F(0), F(0)), jaccard)[0] == F(0)

    return {"audit": "exact_rational_grid", "parameter_sets": len(THETAS), "checked_pairs": checked,
            "membership_values": ["0", "1/2", "1"], "result": "passed"}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
