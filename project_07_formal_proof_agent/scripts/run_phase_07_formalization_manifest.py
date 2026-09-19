"""Create the compact Phase 07 Lean-source manifest.

This command records source provenance only.  It does not run Lean and cannot
be treated as formal proof evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFAULT_PROJECT_ROOT / "src"))

from formal_math.fuzzy_formalization import write_formalization_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project 07 root containing formalizations/lean3/FuzzySimilarity.lean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional manifest path; defaults to results/phase_07_formalization_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = args.output or project_root / "results" / "phase_07_formalization_manifest.json"
    manifest = write_formalization_manifest(output, project_root)
    print(f"Wrote Phase 07 formalization manifest: {output}")
    print(f"Lean source SHA-256: {manifest['source']['sha256']}")
    print("This is source provenance, not Lean compilation evidence.")


if __name__ == "__main__":
    main()
