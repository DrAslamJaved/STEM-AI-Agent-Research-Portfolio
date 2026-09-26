import unittest
from unittest.mock import patch

import numpy as np

from p09.core import Settings
from p09.experiment import run


class IntegerFoldTask:
    folds = [0]

    def get_train_and_val_data(self, fold_number):
        if fold_number != 0 or isinstance(fold_number, str):
            raise TypeError("Matbench fold numbers must be integers")
        return list(range(80)), np.arange(80, dtype=float) / 100

    def get_test_data(self, fold_number, include_target=False):
        if fold_number != 0 or not include_target:
            raise TypeError("Expected integer fold with target")
        return list(range(20)), np.arange(20, dtype=float) / 100


class AdapterTests(unittest.TestCase):
    def test_official_fold_api_receives_integer(self):
        rng = np.random.default_rng(12)
        arrays = [(rng.normal(size=(80, 3)), np.arange(80).astype(str)),
                  (rng.normal(size=(20, 3)), np.array([f"test-{i}" for i in range(20)]))]
        with patch("p09.experiment.features_and_groups", side_effect=arrays), \
             patch("p09.experiment.version", return_value="test"):
            result = run(IntegerFoldTask(), [0], [17], Settings(budgets=(8, 12),
                         initial=8, trees=3, members=2), target_mae=0.6)
        self.assertEqual(result["split_audits"][0]["fold"], "fold_0")
        self.assertEqual(len(result["checkpoints"]), 12)


if __name__ == "__main__":
    unittest.main()
