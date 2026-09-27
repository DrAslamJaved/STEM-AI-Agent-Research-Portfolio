import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p09.synthesis import render, summarize_phase08, synthesize


def phase08_fixture() -> dict:
    return {
        "phase": "08", "task": "matbench_dielectric", "benchmark": "Matbench v0.1",
        "target": "n", "unit": "unitless",
        "protocol": {"seed": 17, "graph_model": {"cpu_threads": 1}},
        "runs": [
            {"fold": fold, "seed": 17, "train_count": 10,
             "models": {"descriptor_random_forest": {"mae": 0.4 + fold / 100, "rmse": 1.0, "test_count": 5},
                        "periodic_message_passing": {"mae": 0.5 + fold / 100, "rmse": 1.2, "test_count": 5}}}
            for fold in range(5)
        ],
    }


class Phase09Tests(unittest.TestCase):
    def test_phase08_summary_preserves_negative_comparison(self):
        summary = summarize_phase08(phase08_fixture())
        self.assertEqual(summary["folds"], 5)
        self.assertEqual(summary["models"]["descriptor_random_forest"]["test_records"], 25)
        self.assertGreater(summary["graph_minus_descriptor"]["mean_fold_mae"], 0)
        self.assertEqual(summary["conclusion"], "graph model did not outperform the descriptor baseline")

    def test_phase08_rejects_unlocked_cpu_protocol(self):
        fixture = phase08_fixture()
        fixture["protocol"]["graph_model"]["cpu_threads"] = 2
        with self.assertRaisesRegex(ValueError, "deterministic protocol"):
            summarize_phase08(fixture)

    @patch("p09.synthesis.analyze_phase03")
    def test_synthesis_hashes_evidence_and_renders_claim_boundary(self, mock_phase03):
        mock_phase03.return_value = {
            "task": "matbench_expt_gap", "target_mae_ev": 0.6, "n_folds": 5, "n_seeds": 2,
            "budgets": [{"budget": 200}, {"budget": 400}],
            "crossing_counts": {"ensemble": {"random": 1, "uncertainty": 1, "uncertainty_diversity": 0}},
        }
        phase03 = {"threshold_summary": [
            {"labels_saved_vs_random": 0, "labels_saved_diversity_vs_random": None}
        ]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p03, p08 = root / "phase03.json", root / "phase08.json"
            p03.write_text(json.dumps(phase03), encoding="utf-8")
            p08.write_text(json.dumps(phase08_fixture()), encoding="utf-8")
            summary = synthesize(p03, p08)
        self.assertEqual(summary["phase03"]["conclusion"], "no observed label savings at the locked ensemble target")
        self.assertIn("sha256", summary["evidence_files"]["phase03"])
        self.assertIn("Future work should preregister", render(summary))


if __name__ == "__main__":
    unittest.main()
