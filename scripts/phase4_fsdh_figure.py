#!/usr/bin/env python3
"""FINAL SPRINT Phase 4: FSDH-vs-horizon figure, all models, one panel,
persistence zero-line. Uses results/full_benchmark_val.json's fsdh_curves
(Sprint 6, full 2011-2024 record) as a stand-in -- this script is
re-pointed at the canonical-split results in Phase 5 once available
(null_ridge/null_krr/residual_ridge's FSDH curves were not saved by the
Sprint 6 script -- a real gap, fixed for Phase 5's canonical rerun, not
worth a separate multi-hour recompute just for this supplementary figure
now)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.experiment.figstyle import apply_style, PALETTE

RESULTS_PATH = REPO_ROOT / "results" / "full_benchmark_val.json"
FIG_DIR = REPO_ROOT / "figures"

DISPLAY_NAMES = {
    "arima": "ARIMA(2,1,2)", "arima_auto": "ARIMA (auto)",
    "esn_dim_matched": "ESN dim-matched", "esn_500": "ESN-500", "residual_esn": "Residual-ESN",
    "null_ridge": "null-control Ridge", "residual_ridge": "Residual-Ridge",
}


def main():
    with open(RESULTS_PATH) as f:
        d = json.load(f)
    horizons = d["eval_horizons"]
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1.0, label="persistence")
    for i, (name, disp) in enumerate(DISPLAY_NAMES.items()):
        rows = d["metrics"].get(name)
        if rows is None:
            continue
        hs = [h for h in horizons if str(h) in rows]
        skills = [rows[str(h)]["skill"] * 100 for h in hs]
        ax.plot(hs, skills, color=PALETTE[(i + 1) % len(PALETTE)], label=disp, linewidth=1.3)
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("skill vs. persistence (%)")
    ax.set_ylim(-100, 40)  # ARIMA's extreme negative values would otherwise swamp the informative range
    ax.legend(fontsize=7, loc="lower right")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fsdh_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out} (null_ridge/residual_ridge included since sprint6_full_benchmark.py DOES "
          f"compute their per-horizon skill in `metrics`, even though it wasn't separately saved "
          f"in fsdh_curves)")


if __name__ == "__main__":
    main()
