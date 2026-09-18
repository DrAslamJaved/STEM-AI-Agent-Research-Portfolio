from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Phase01ContractTests(unittest.TestCase):
    def test_research_contract_defines_all_three_arms(self) -> None:
        text = (PROJECT_ROOT / "docs" / "phase_01_research_contract.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("LLM-only", text)
        self.assertIn("LLM + SymPy", text)
        self.assertIn("LLM + Lean repair", text)

    def test_validity_policy_prohibits_shortcuts(self) -> None:
        text = (PROJECT_ROOT / "docs" / "validity_policy.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("`sorry`", text)
        self.assertIn("`admit`", text)
        self.assertIn("held-out test split", text)

    def test_config_locks_the_test_split(self) -> None:
        text = (PROJECT_ROOT / "configs" / "phase_01_contract.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("test_split_locked: true", text)


if __name__ == "__main__":
    unittest.main()