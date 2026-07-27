#!/usr/bin/env python3
"""Build the submission zip: `TeamName_Challenge_Phase3.zip` containing the
write-up PDF, source folder, README.md. Team name is "QRCx".

Uses `git ls-files` (respects .gitignore, never includes venv/, raw ISD
downloads, __pycache__, or large regeneratable .npy feature arrays --
only what's tracked in version control), then drops a further explicit
exclude-list of internal planning/scratch artifacts that are not part of
the deliverable (superseded pre-package demo scripts, sprint-log dev
narrative, superprompt/planning docs, dropped Dirac-3 results, the
single-column paper variant now that the submission is twocolumn-only,
and duplicate/superseded figure sets) so reviewers see only the
write-up, the canonical source, and the results that back it. The paper
is twocolumn-only (the single-column variant was dropped).
"""
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "QRCx_Challenge_Phase3.zip"
OUT_PATH = REPO_ROOT / ZIP_NAME

# Exact tracked file paths to drop from the reviewer-facing zip.
EXCLUDE_FILES = {
    "Qbraid.py",
    "used_baselines.py",
    "pipeline_demo.py",
    "momo_reports.md",
    "QRCx_FINAL_24H_SUPERPROMPT.md",
    "QRCx_v5_sprint_plan.md",
    "docs/organizer_question.md",
    "results/baselines_test_HISTORICAL.json",
    "results/esn_diagnosis_test_HISTORICAL.json",
}

# Tracked path prefixes (whole directories) to drop.
EXCLUDE_DIR_PREFIXES = (
    "docs/archive/",
    "docs/sprint_log/",
    "qrc_figures/",
    "results/dirac3/",
)


def is_excluded(f: str) -> bool:
    return f in EXCLUDE_FILES or f.startswith(EXCLUDE_DIR_PREFIXES)


def main():
    result = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    all_files = [f for f in result.stdout.splitlines() if f.strip()]
    files = [f for f in all_files if not is_excluded(f)]
    print(f"{len(all_files)} tracked files; {len(all_files) - len(files)} excluded as internal/superseded; "
          f"{len(files)} to include.")

    pdf_path = REPO_ROOT / "docs" / "paper" / "main.pdf"
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} does not exist -- compile the paper first (pdflatex).", file=sys.stderr)
        sys.exit(1)

    if OUT_PATH.exists():
        OUT_PATH.unlink()

    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write-up PDF first, at archive root, so it's the first thing a judge sees.
        zf.write(pdf_path, arcname="QRCx_writeup.pdf")
        for f in files:
            full = REPO_ROOT / f
            if not full.exists():
                print(f"  WARNING: tracked file missing on disk, skipping: {f}")
                continue
            zf.write(full, arcname=f)

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Wrote {OUT_PATH} ({size_mb:.2f} MB, {len(files) + 1} files)")


if __name__ == "__main__":
    main()
