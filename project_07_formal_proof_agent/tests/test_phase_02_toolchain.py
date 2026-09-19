"""Contract tests for frozen miniF2F benchmark and toolchain evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase_02_benchmark_toolchain.yaml"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "phase_02" / "toolchain_validation.json"
PROVENANCE_PATH = PROJECT_ROOT / "benchmarks" / "minif2f" / "provenance.json"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"

PINNED_COMMIT = "f0dcc8b59e630fba00ba9569ca6714700e0a8801"
MATHLIB_REVISION = "cb2b02fff213ed6f65bebd64446baac64137dcda"
LEAN_TOOLCHAIN = "leanprover-community/lean:3.42.1"


class Phase02ToolchainTests(unittest.TestCase):
    def test_config_locks_original_minif2f_environment(self) -> None:
        config = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("https://github.com/openai/miniF2F.git", config)
        self.assertIn(f"commit: {PINNED_COMMIT}", config)
        self.assertIn(f"lean: {LEAN_TOOLCHAIN}", config)
        self.assertIn(f"mathlib_revision: {MATHLIB_REVISION}", config)
        self.assertIn("test_split_locked: true", config)

    def test_generated_manifest_records_successful_build(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(PINNED_COMMIT, manifest["benchmark_commit"])
        self.assertEqual(LEAN_TOOLCHAIN, manifest["lean_toolchain"])
        self.assertEqual(MATHLIB_REVISION, manifest["mathlib_revision"])
        self.assertEqual("passed", manifest["configure_status"])
        self.assertEqual("passed", manifest["build_status"])
        self.assertEqual(0, manifest["build_exit_code"])

    def test_provenance_manifest_contains_all_locks(self) -> None:
        provenance = PROVENANCE_PATH.read_text(encoding="utf-8")
        for required in (PINNED_COMMIT, "v1", LEAN_TOOLCHAIN, MATHLIB_REVISION):
            self.assertIn(required, provenance)

    def test_gitignore_excludes_local_clone_and_logs(self) -> None:
        ignored = GITIGNORE_PATH.read_text(encoding="utf-8")
        self.assertIn("benchmarks/minif2f/upstream/", ignored)
        self.assertIn("benchmarks/minif2f/upstream/_target/", ignored)
        self.assertIn("reports/phase_02/*.log", ignored)


if __name__ == "__main__":
    unittest.main()
