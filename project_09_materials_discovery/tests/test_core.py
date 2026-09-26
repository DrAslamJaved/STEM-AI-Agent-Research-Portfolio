import unittest

import numpy as np

from p09.core import Settings, conformal_radius, crossing_summary, make_partition, simulate


class ProtocolTests(unittest.TestCase):
    def test_calibration_groups_never_enter_pool(self):
        groups = np.repeat(np.arange(30), 2).astype(str)
        pool, cal = make_partition(groups, len(groups), 7, 0.2)
        self.assertFalse(set(groups[pool]) & set(groups[cal]))
        self.assertEqual(len(pool) + len(cal), len(groups))

    def test_conformal_finite_sample_quantile(self):
        self.assertEqual(conformal_radius(np.arange(1, 20), 0.1), 18)
        self.assertEqual(conformal_radius(np.arange(1, 8), 0.1), float("inf"))

    def test_paired_budget_and_locked_calibration(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(100, 6))
        y = x[:, 0] * 2 + rng.normal(size=100) * 0.1
        settings = Settings(budgets=(8, 12), initial=8, trees=3, members=2)
        result = simulate(x[:80], y[:80], np.arange(80).astype(str), x[80:], y[80:], seed=13, settings=settings)
        ens = [r for r in result if r["model"] == "ensemble"]
        self.assertEqual(len(ens), 4)
        for row in ens:
            self.assertEqual(len(row["selected_train_positions"]), row["budget"])
            self.assertFalse(set(row["selected_train_positions"]) & set(row["calibration_train_positions"]))
        self.assertEqual(ens[0]["selected_train_positions"], ens[2]["selected_train_positions"])
        self.assertEqual(ens[0]["mae_ev"], ens[2]["mae_ev"])
        self.assertTrue(set(ens[0]["selected_train_positions"]).issubset(ens[1]["selected_train_positions"]))
        self.assertTrue(set(ens[2]["selected_train_positions"]).issubset(ens[3]["selected_train_positions"]))
        changed_targets = simulate(x[:80], y[:80], np.arange(80).astype(str),
                                   x[80:], y[80:] + 50, seed=13, settings=settings)
        other_ens = [r for r in changed_targets if r["model"] == "ensemble"]
        self.assertEqual([r["selected_train_positions"] for r in ens],
                         [r["selected_train_positions"] for r in other_ens])

    def test_crossing_reports_failed_threshold(self):
        rows = [dict(fold="fold_0", seed=1, policy=p, budget=b, model="ensemble", mae_ev=mae)
                for p, b, mae in [("random", 8, 0.7), ("random", 12, 0.6),
                                  ("uncertainty", 8, 0.8), ("uncertainty", 12, 0.7)]]
        self.assertIsNone(crossing_summary(rows, 0.65)[0]["labels_saved_vs_random"])


if __name__ == "__main__":
    unittest.main()
