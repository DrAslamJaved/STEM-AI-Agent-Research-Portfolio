import copy
import unittest

from p09.diagnostics import analyze, render


def fixture(mode="official"):
    rows = []
    for policy in ("random", "uncertainty", "uncertainty_diversity"):
        for budget in (2, 3):
            for model in ("mean", "forest", "ensemble"):
                row = dict(fold="fold_0", seed=17, budget=budget, policy=policy,
                           model=model, mae_ev=.7 if budget == 2 else .5,
                           rmse_ev=.8 if budget == 2 else .6,
                           test_count=2, calibration_count=1)
                if model == "ensemble":
                    row.update(selected_train_positions=[0, 1] if budget == 2 else [0, 1, 2],
                               calibration_train_positions=[5], coverage_90=.9,
                               interval_width_ev=2., mean_disagreement_ev=.2)
                rows.append(row)
    return dict(task="matbench_expt_gap", split_mode=mode, versions={"test": "1"},
                settings=dict(folds=[0], seeds=[17], budgets=[2, 3], initial=2,
                              include_diversity=True, target_mae_ev=.6),
                split_audits=[dict(fold="fold_0", train_count=6, test_count=2,
                                   original_overlapping_train_test_compositions=0,
                                   removed_training_records=0)], checkpoints=rows,
                threshold_summary=[dict(fold="fold_0", seed=17,
                                        first_budget={p: 3 for p in
                                                      ("random", "uncertainty", "uncertainty_diversity")})])


class Phase04Tests(unittest.TestCase):
    def test_model_specific_crossings_and_redundant_robustness(self):
        official = fixture()
        grouped = fixture("group_exclusive")
        report = analyze(official, grouped)
        self.assertEqual(report["crossing_counts"]["forest"]["random"], 1)
        self.assertIn("Not an independent robustness test", report["robustness"])
        self.assertIn("exploratory", render(report))

    def test_mismatched_initial_labels_are_rejected(self):
        data = fixture()
        row = next(r for r in data["checkpoints"] if r["policy"] == "uncertainty_diversity"
                   and r["budget"] == 2 and r["model"] == "ensemble")
        row["selected_train_positions"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "different initial labels"):
            analyze(data)

    def test_crossing_and_unfiltered_metric_mismatch_are_rejected(self):
        data = fixture()
        bad = copy.deepcopy(data)
        bad["threshold_summary"][0]["first_budget"]["random"] = None
        with self.assertRaisesRegex(ValueError, "Stored ensemble crossing"):
            analyze(bad)
        grouped = fixture("group_exclusive")
        grouped["checkpoints"][0]["mae_ev"] = 100
        with self.assertRaisesRegex(ValueError, "different metrics"):
            analyze(data, grouped)


if __name__ == "__main__":
    unittest.main()
