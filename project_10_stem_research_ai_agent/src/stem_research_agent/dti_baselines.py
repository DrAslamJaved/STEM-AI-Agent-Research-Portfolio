"""Reproducible DTI baseline and split diagnostics."""
from random import Random

def prevalence_score(records):
    rows = list(records)
    if not rows: raise ValueError("Training records cannot be empty.")
    labels = [int(row["label"]) for row in rows]
    if any(label not in (0, 1) for label in labels): raise ValueError("Labels must be binary.")
    return sum(labels) / len(labels)

def seeded_random_split(records, *, test_fraction, seed):
    rows = list(records)
    if len(rows) < 2 or not 0 < test_fraction < 1: raise ValueError("Need at least two rows and 0 < test_fraction < 1.")
    order = list(range(len(rows))); Random(seed).shuffle(order)
    test_indices = set(order[:max(1, min(len(rows) - 1, round(len(rows) * test_fraction)))])
    return ([row for index, row in enumerate(rows) if index not in test_indices],
            [row for index, row in enumerate(rows) if index in test_indices])

def split_overlap(train_records, test_records):
    train, test = list(train_records), list(test_records)
    train_drugs, test_drugs = {str(x["drug_id"]) for x in train}, {str(x["drug_id"]) for x in test}
    train_targets, test_targets = {str(x["target_id"]) for x in train}, {str(x["target_id"]) for x in test}
    return {"shared_drugs": len(train_drugs & test_drugs), "shared_targets": len(train_targets & test_targets),
            "train_records": len(train), "test_records": len(test)}

def cold_drug_split(records, *, held_out_drugs):
    rows = list(records)
    train = [row for row in rows if str(row["drug_id"]) not in held_out_drugs]
    test = [row for row in rows if str(row["drug_id"]) in held_out_drugs]
    if not held_out_drugs or not train or not test: raise ValueError("Held-out drugs must create non-empty partitions.")
    return train, test
