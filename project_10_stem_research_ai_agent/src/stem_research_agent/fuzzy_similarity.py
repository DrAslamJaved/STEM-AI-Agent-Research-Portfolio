"""Reference fuzzy similarities for finite vectors in [0, 1]."""
from math import sqrt

def _vectors(left, right):
    a, b = [float(x) for x in left], [float(x) for x in right]
    if not a or len(a) != len(b) or any(x < 0 or x > 1 for x in a + b):
        raise ValueError("Non-empty equal-length membership vectors in [0, 1] are required.")
    return a, b

def jaccard_similarity(left, right):
    a, b = _vectors(left, right); union = sum(max(x, y) for x, y in zip(a, b))
    return 1.0 if union == 0 else sum(min(x, y) for x, y in zip(a, b)) / union

def dice_similarity(left, right):
    a, b = _vectors(left, right); denominator = sum(a) + sum(b)
    return 1.0 if denominator == 0 else 2 * sum(min(x, y) for x, y in zip(a, b)) / denominator

def cosine_similarity(left, right):
    a, b = _vectors(left, right); denominator = sqrt(sum(x*x for x in a)) * sqrt(sum(y*y for y in b))
    return 1.0 if denominator == 0 and a == b else 0.0 if denominator == 0 else sum(x*y for x, y in zip(a, b)) / denominator

def cardinality_balance_similarity(left, right):
    """Equal cardinality is not a claim of equal fuzzy sets."""
    a, b = _vectors(left, right)
    return 1 - abs(sum(a) - sum(b)) / len(a)
