import unittest

import numpy as np

from p09.experiment import exclude_test_compositions
from p09.report import analyze, render


class Phase02Tests(unittest.TestCase):
    def test_filter_keeps_outer_test_groups_out_of_training(self):
        x = np.arange(8).reshape(4, 2)
        y = np.array([2., 3., 4., 5.])
        groups = np.array(["A", "B", "A", "C"])
        filtered_x, filtered_y, filtered_groups, removed = exclude_test_compositions(
            x, y, groups, np.array(["A", "D"]))
        self.assertEqual(removed, 2)
        np.testing.assert_array_equal(filtered_groups, ["B", "C"])
        np.testing.assert_array_equal(filtered_y, [3., 5.])
        np.testing.assert_array_equal(filtered_x, x[[1, 3]])

    def test_folds_are_units_and_failed_crossings_stay_visible(self):
        rows = []
        for fold, active in [("fold_0", 0.4), ("fold_1", 0.6), ("fold_2", 0.3)]:
            for seed in (1, 2):
                for policy, mae in (("random", 0.5), ("uncertainty", active)):
                    rows.append(dict(fold=fold, seed=seed, budget=20, policy=policy,
                                     model="ensemble", mae_ev=mae, coverage_90=0.85))
        result = dict(task="matbench_expt_gap", split_mode="official",
                      settings={"target_mae_ev": 0.45}, checkpoints=rows,
                      threshold_summary=[{"first_budget": {"random": None, "uncertainty": 20}},
                                         {"first_budget": {"random": None, "uncertainty": None}}],
                      split_audits=[])
        summary = analyze(result)
        self.assertEqual(summary["budget_summary"][0]["folds"], 3)
        self.assertEqual(summary["crossing_categories"],
                         {"both": 0, "uncertainty_only": 1, "random_only": 0, "neither": 1})
        self.assertIsNotNone(summary["budget_summary"][0]["fold_bootstrap_ci95_ev"])
        self.assertIn("retrospective", render(result, summary))

    def test_unpaired_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, "unmatched"):
            analyze({"checkpoints": [{"fold": "fold_0", "seed": 1, "budget": 10,
                                       "policy": "random", "model": "ensemble"}]})

    def test_phase01_audit_can_be_rendered(self):
        result = {"task": "matbench_expt_gap", "settings": {"target_mae_ev": 0.6},
                  "split_audits": [{"fold": "fold_0", "train_count": 80, "test_count": 20,
                                    "overlapping_train_test_compositions": 2,
                                    "overlapping_test_records": 3}]}
        summary = {"n_folds": 1, "n_seeds": 1, "budget_summary": [],
                   "crossing_categories": {"both": 0, "uncertainty_only": 0,
                                           "random_only": 0, "neither": 1},
                   "labels_saved_where_both_cross": []}
        self.assertIn("| fold_0 | 80 | 20 | 2 | 3 | 0 |", render(result, summary))


if __name__ == "__main__":
    unittest.main()
