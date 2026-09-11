"""Deterministic binary metrics without a machine-learning dependency."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class BinaryMetrics:
    true_negative: int; false_positive: int; false_negative: int; true_positive: int
    accuracy: float; precision: float; recall: float; f1: float
    roc_auc: float | None; pr_auc: float | None
    def to_dict(self): return asdict(self)

def _auc(labels, scores):
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives: return None
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores); start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]: end += 1
        rank = ((start + 1) + end) / 2
        for position in range(start, end): ranks[ordered[position][0]] = rank
        start = end
    return (sum(rank for label, rank in zip(labels, ranks) if label) - positives * (positives + 1) / 2) / (positives * negatives)

def _pr_auc(labels, scores):
    positives = sum(labels)
    if not positives: return None
    tp = fp = 0; previous_recall = area = 0.0
    for _, label in sorted(zip(scores, labels), reverse=True):
        tp += label; fp += 1 - label
        recall, precision = tp / positives, tp / (tp + fp)
        area += (recall - previous_recall) * precision; previous_recall = recall
    return area

def evaluate_binary_predictions(labels, scores, *, threshold=0.5):
    labels, scores = [int(x) for x in labels], [float(x) for x in scores]
    if not labels or len(labels) != len(scores) or any(x not in (0, 1) for x in labels):
        raise ValueError("Non-empty, equally sized binary labels and scores are required.")
    tn = fp = fn = tp = 0
    for actual, score in zip(labels, scores):
        predicted = int(score >= threshold)
        if actual == predicted == 0: tn += 1
        elif actual == 0: fp += 1
        elif predicted == 0: fn += 1
        else: tp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return BinaryMetrics(tn, fp, fn, tp, (tn + tp) / len(labels), precision, recall,
        2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        _auc(labels, scores), _pr_auc(labels, scores))
