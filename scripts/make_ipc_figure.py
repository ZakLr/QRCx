#!/usr/bin/env python3
"""Sec 5 (IPC-matched tuning) figure for the paper: 3 panels --
(a) task demand D(delay, degree) at h=6 (the named target horizon),
(b) reservoir supply C(lag, degree) at the matched config,
(c) captured-capacity comparison across the (gamma1, a) grid.
Vector PDF output, from results/ipc_matching.json (no new computation --
purely a repackaging of scripts/sprint5_figures.py's existing panels
into the single 3-panel layout the paper spec asks for).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.experiment.figstyle import apply_style, PALETTE, label_panel

RESULTS_PATH = REPO_ROOT / "results" / "ipc_matching.json"
FIG_DIR = REPO_ROOT / "figures"


def main():
    with open(RESULTS_PATH) as f:
        result = json.load(f)
    apply_style()

    demand = result["demand"]
    best = result["best_matched_config"]
    best_entry = next(e for e in result["supply_grid"]
                       if e["gamma1"] == best["gamma1"] and e["a"] == best["a"])
    degrees = demand["6"]["degrees"]
    delays = demand["6"]["delays"]
    lags = best_entry["lags"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    D = np.array(demand["6"]["D"])
    im0 = axes[0].imshow(D, aspect="auto", cmap="viridis", vmin=0)
    axes[0].set_yticks(range(len(degrees))); axes[0].set_yticklabels(degrees)
    axes[0].set_xlabel("delay (h)"); axes[0].set_ylabel("Legendre degree")
    axes[0].set_xticks(range(0, len(delays), 4))
    axes[0].set_xticklabels([delays[i] for i in range(0, len(delays), 4)])
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    label_panel(axes[0], "a")

    C = np.array(best_entry["C"])
    im1 = axes[1].imshow(C, aspect="auto", cmap="magma", vmin=0)
    axes[1].set_yticks(range(len(degrees))); axes[1].set_yticklabels(degrees)
    axes[1].set_xlabel("lag (steps)"); axes[1].set_ylabel("Hermite degree")
    axes[1].set_xticks(range(0, len(lags), 4))
    axes[1].set_xticklabels([lags[i] for i in range(0, len(lags), 4)])
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    label_panel(axes[1], "b")

    scores = result["matching_scores"]
    labels = [f"$\\gamma_1$={s['gamma1']:.2f}\n$a$={s['a']:.1f}" for s in scores]
    values = [s["captured_total"] for s in scores]
    best_idx = int(np.argmax(values))
    colors = [PALETTE[1] if i == best_idx else PALETTE[5] for i in range(len(scores))]
    axes[2].bar(range(len(scores)), values, color=colors)
    axes[2].set_xticks(range(len(scores)))
    axes[2].set_xticklabels(labels, fontsize=6, rotation=90)
    axes[2].set_ylabel(r"captured capacity $\sum \min(C,D)$")
    label_panel(axes[2], "c")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "ipc_matching_3panel.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
