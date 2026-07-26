#!/usr/bin/env python3
"""FINAL SPRINT Phase 5: FSDH-vs-horizon figure, all models, one panel,
persistence zero-line. Uses results/baselines_test.json -- the real
canonical-split (train 2019-2022/val 2023/test 2024) classical
baselines, evaluated at all 48 horizons -- consistent with every other
number in the paper. (Superseded an earlier version of this script that
sourced the stale Sprint-6 full-2011-2024-record file, a real
inconsistency caught during the final QA pass.) QRC/concat models are
not included here since only their scalar FSDH value, not a full
per-horizon skill curve, was computed (see results/qrc_fsdh_test.json)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.experiment.figstyle import apply_style, PALETTE

RESULTS_PATH = REPO_ROOT / "results" / "baselines_test.json"
FIG_DIR = REPO_ROOT / "figures"

DISPLAY_NAMES = {
    "arima": "ARIMA(2,1,2)", "arima_auto": "ARIMA (auto)",
    "esn_dim_matched": "ESN dim-matched", "esn_500": "ESN-500", "residual_esn": "Residual-ESN",
}
# null_ridge/null_krr/residual_ridge are NOT plotted here: generate_baselines.py
# only ever evaluates them at EVAL_HORIZONS=[1,6], not the full ALL_HORIZONS=1..48
# sweep (a real, documented scripting gap) -- their h=1/6 numbers are in
# Table 4 instead, not a partial/misleading 2-point line here.


def main():
    with open(RESULTS_PATH) as f:
        d = json.load(f)
    apply_style()
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    ax.axhline(0, color="black", linewidth=1.0, label="persistence")
    for i, (name, disp) in enumerate(DISPLAY_NAMES.items()):
        curve = d["skill_curves"].get(name)
        if curve is None:
            continue
        hs = sorted(int(h) for h in curve.keys())
        skills = [curve[str(h)] * 100 for h in hs]
        ax.plot(hs, skills, color=PALETTE[(i + 1) % len(PALETTE)], label=disp, linewidth=1.3)
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("skill vs. persistence (%)")
    ax.set_ylim(-100, 40)  # ARIMA's extreme negative values would otherwise swamp the informative range
    ax.legend(fontsize=6, loc="lower right")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fsdh_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    out_pdf = FIG_DIR / "fsdh_curve.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out} and {out_pdf}")


if __name__ == "__main__":
    main()
