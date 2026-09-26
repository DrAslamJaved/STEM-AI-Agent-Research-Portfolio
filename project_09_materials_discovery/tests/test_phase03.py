import unittest

import numpy as np

from p09.core import Settings, crossing_summary, diversity_scores, simulate
from p09.diversity_report import analyze, render
from p09.report import analyze as legacy_analyze


class Phase03Tests(unittest.TestCase):
    def test_novel_compositions_can_beat_largest_disagreement(self):
        x = np.zeros((5, 123))
        x[1, 0], x[2, 0], x[3, 0], x[4, 0] = 0.1, 3, 2, 1
        scores = diversity_scores(x, np.array([0]), np.arange(1, 5), np.array([10, 8, 7, 1]))
        self.assertEqual(int(np.argmax(scores)), 1)  # row 2, further from row 0

    def test_paired_start_and_test_target_independence(self):
        rng = np.random.default_rng(18)
        x = rng.random((72, 123))
        y = x[:, 0] + rng.normal(scale=0.01, size=72)
        config = Settings(budgets=(8, 12), initial=8, trees=3, members=2)
        kwargs = dict(seed=5, settings=config, include_diversity=True)
        a = simulate(x[:60], y[:60], np.arange(60).astype(str), x[60:], y[60:], **kwargs)
        b = simulate(x[:60], y[:60], np.arange(60).astype(str), x[60:], y[60:] + 99, **kwargs)
        ensemble = [r for r in a if r["model"] == "ensemble"]
        other = [r for r in b if r["model"] == "ensemble"]
        self.assertEqual(len(ensemble), 6)
        self.assertEqual([r["selected_train_positions"] for r in ensemble],
                         [r["selected_train_positions"] for r in other])
        first = [r for r in ensemble if r["budget"] == 8]
        self.assertEqual(len({tuple(r["selected_train_positions"]) for r in first}), 1)
        self.assertEqual(len({r["mae_ev"] for r in first}), 1)
        for row in ensemble:
            self.assertEqual(len(row["selected_train_positions"]), row["budget"])
            self.assertFalse(set(row["selected_train_positions"]) &
                             set(row["calibration_train_positions"]))
        for policy in ("random", "uncertainty", "uncertainty_diversity"):
            selection = [r for r in ensemble if r["policy"] == policy]
            self.assertTrue(set(selection[0]["selected_train_positions"]).issubset(
                selection[1]["selected_train_positions"]))

    def test_three_policy_report_preserves_failed_crossings(self):
        rows = []
        for fold in range(3):
            for seed in (17, 23):
                for budget in (8, 12):
                    for policy, mae in (("random", .7), ("uncertainty", .6),
                                        ("uncertainty_diversity", .55)):
                        rows.append(dict(fold=f"fold_{fold}", seed=seed, budget=budget,
                                         policy=policy, model="ensemble", mae_ev=mae,
                                         coverage_90=.9))
        thresholds = crossing_summary(rows, .58,
                                      ("random", "uncertainty", "uncertainty_diversity"))
        self.assertIsNone(thresholds[0]["labels_saved_diversity_vs_random"])
        result = dict(task="matbench_expt_gap", split_mode="official",
                      settings={"target_mae_ev": .58}, checkpoints=rows,
                      threshold_summary=thresholds)
        summary = analyze(result)
        self.assertEqual(summary["crossing_counts"],
                         {"random": 0, "uncertainty": 0, "uncertainty_diversity": 6})
        self.assertEqual(summary["paired_savings_where_both_cross"]["random"], [])
        self.assertAlmostEqual(summary["budget_summary"][0]["hybrid_gain_vs"]["random"]["gain_ev"], .15)
        self.assertIn("label saving undefined", render(result, summary))
        self.assertEqual(legacy_analyze(result)["budget_summary"][0]["folds"], 3)
        bad = dict(result, checkpoints=rows[:-1])
        with self.assertRaisesRegex(ValueError, "unmatched"):
            analyze(bad)


if __name__ == "__main__":
    unittest.main()
