import copy
import unittest

import numpy as np

from p09.conditional_report import analyze, render
from p09.core import Settings, simulate


class Phase05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        x = rng.random((110, 123))
        y = x[:, 0] * 2 + rng.normal(scale=.2, size=110)
        records = []
        rows = simulate(x[:80], y[:80], np.arange(80).astype(str), x[80:], y[80:],
                        seed=17, settings=Settings(budgets=(8, 12), initial=8, trees=3, members=2),
                        include_diversity=True, prediction_sink=records)
        cls.checkpoints = dict(task="matbench_expt_gap", split_mode="official",
                               settings={"folds": [0], "seeds": [17], "budgets": [8, 12],
                                         "include_diversity": True},
                               split_audits=[{"fold": "fold_0", "test_count": 30}],
                               checkpoints=[{"fold": "fold_0", **r} for r in rows])
        cls.predictions = dict(task="matbench_expt_gap", split_mode="official",
                               checkpoint_sha256="known-digest",
                               records=[{"fold": "fold_0", **r} for r in records])

    def test_prediction_aggregates_reproduce_checkpoint_metrics(self):
        report = analyze(self.predictions, self.checkpoints, "known-digest")
        self.assertEqual(report["prediction_records"], 180)
        self.assertEqual(report["folds"], 1)
        self.assertEqual(len(report["budget_summary"]), 6)
        self.assertIsNone(next(r for r in report["element_strata"] if r["elements"] == "one")["coverage_90"])
        self.assertIn("exploratory", render(report))

    def test_tampering_with_test_evaluation_is_detected(self):
        altered = copy.deepcopy(self.predictions)
        altered["records"][0]["absolute_error_ev"] += 1
        with self.assertRaisesRegex(ValueError, "disagree with checkpoint"):
            analyze(altered, self.checkpoints, "known-digest")
        with self.assertRaisesRegex(ValueError, "digest"):
            analyze(self.predictions, self.checkpoints, "different-digest")


if __name__ == "__main__":
    unittest.main()
