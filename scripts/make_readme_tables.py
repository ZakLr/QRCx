#!/usr/bin/env python3
"""Regenerate the README results block from results/*.json.

Every number in README.md's results section must be traceable to a results
file so a judge re-running the pipeline reproduces exactly what's printed.
This script is the only thing allowed to write between the
<!-- RESULTS:BEGIN --> / <!-- RESULTS:END --> markers in README.md.

Usage:
    python scripts/make_readme_tables.py [--check]

    --check   Exit nonzero if the README block is stale instead of rewriting it
              (used as a pre-commit / CI gate).
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
README_PATH = REPO_ROOT / "README.md"
BEGIN_MARKER = "<!-- RESULTS:BEGIN -->"
END_MARKER = "<!-- RESULTS:END -->"

PENDING_BLOCK = (
    "_Phase 3 results pending. Run the reproduce command in this README to "
    "generate `results/*.json`, then re-run this script to populate this "
    "section. No performance numbers are reported until they exist in "
    "`results/*.json`._"
)


def load_results() -> dict[str, dict]:
    results = {}
    if not RESULTS_DIR.is_dir():
        return results
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            results[path.stem] = json.load(f)
    return results


def render_table(results: dict[str, dict]) -> str:
    if not results:
        return PENDING_BLOCK

    lines = ["| Run | Horizon | RMSE | MAE | Skill | FSDH (h) |",
             "|---|---|---|---|---|---|"]
    for name, data in results.items():
        metrics = data.get("metrics", data)
        for horizon, row in sorted(metrics.items()) if isinstance(metrics, dict) else []:
            lines.append(
                f"| {name} | {horizon} | {row.get('rmse', '-')} | "
                f"{row.get('mae', '-')} | {row.get('skill', '-')} | "
                f"{row.get('fsdh', '-')} |"
            )
    if len(lines) == 2:
        return PENDING_BLOCK
    return "\n".join(lines)


def build_new_readme(readme_text: str, block: str) -> str:
    start = readme_text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = readme_text.index(END_MARKER)
    return readme_text[:start] + "\n" + block + "\n" + readme_text[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                         help="Fail if README.md results block is stale")
    args = parser.parse_args()

    readme_text = README_PATH.read_text(encoding="utf-8")
    if BEGIN_MARKER not in readme_text or END_MARKER not in readme_text:
        print(f"ERROR: {README_PATH} is missing {BEGIN_MARKER}/{END_MARKER} markers")
        return 2

    results = load_results()
    block = render_table(results)
    new_text = build_new_readme(readme_text, block)

    if args.check:
        if new_text != readme_text:
            print("README results block is stale. Run: python scripts/make_readme_tables.py")
            return 1
        print("README results block is up to date.")
        return 0

    README_PATH.write_text(new_text, encoding="utf-8")
    print(f"Updated {README_PATH} from {len(results)} results file(s) in {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
