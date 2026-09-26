import copy
import unittest

import numpy as np

from p09.core import Settings, normalized_conformal_half_width, simulate
from p09.normalized_report import analyze, render


class Phase06Tests(unittest.TestCase):
    def test_scaled_half_width_uses_calibration_floor(self):
        half_width, score, floor = normalized_conformal_half_width(
            np.array([1., 2., 3., 4., 5.]), np.array([0., .1, .2, .3, .4]),
            np.array([0., .2, .8]), .2)
        self.assertGreater(floor, 0)
        self.assertGreater(score, 0)
        self.assertGreater(half_width[2], half_width[1])
        self.assertEqual(half_width[0], score * floor)

    def test_normalized_prediction_report_matches_checkpoints(self):
        rng = np.random.default_rng(41)
        x = rng.random((110, 123))
        y = x[:, 0] + rng.normal(scale=.25, size=110)
        records = []
        rows = simulate(x[:80], y[:80], np.arange(80).astype(str), x[80:], y[80:],
                        seed=17, settings=Settings(budgets=(8, 12), initial=8, trees=3, members=2),
                        include_diversity=True, include_normalized_conformal=True,
                        prediction_sink=records)
        checkpoints = {"task": "matbench_expt_gap",
                       "settings": {"folds": [0], "seeds": [17], "budgets": [8, 12],
                                    "include_diversity": True, "normalized_conformal": True,
                                    "normalized_spread_floor_quantile": .10},
                       "split_audits": [{"fold": "fold_0", "test_count": 30}],
                       "checkpoints": [{"fold": "fold_0", **row} for row in rows]}
        predictions = {"checkpoint_sha256": "digest", "records": [{"fold": "fold_0", **row}
                                                                        for row in records]}
        report = analyze(predictions, checkpoints, "digest")
        self.assertEqual(report["prediction_records"], 180)
        self.assertEqual(len(report["summary"]), 6)
        self.assertIn("Scaled coverage", render(report))
        tampered = copy.deepcopy(predictions)
        tampered["records"][0]["normalized_covered_90"] = not tampered["records"][0]["normalized_covered_90"]
        with self.assertRaisesRegex(ValueError, "disagree"):
            analyze(tampered, checkpoints, "digest")


if __name__ == "__main__":
    unittest.main()
