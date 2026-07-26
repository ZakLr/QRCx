#!/usr/bin/env python3
"""FINAL SPRINT Phase 7: build the submission zip.

Per QRCx_FINAL_24H_SUPERPROMPT.md Phase 7 item 5: `TeamName_Challenge_Phase3.zip`
containing the write-up PDF, source folder, README.md. Team name is
"QRCx" (README.md / docs/paper/main.tex's \\author). Uses `git ls-files`
(same methodology as Sprint 9) so the archive respects .gitignore and
never includes venv/, raw ISD downloads, __pycache__, or large
regeneratable .npy feature arrays -- only what's actually tracked in
version control.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "QRCx_Challenge_Phase3.zip"
OUT_PATH = REPO_ROOT / ZIP_NAME


def main():
    result = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    files = [f for f in result.stdout.splitlines() if f.strip()]
    print(f"{len(files)} tracked files to include.")

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
