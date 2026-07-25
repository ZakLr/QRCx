#!/usr/bin/env python3
"""Sprint 5 figures: demand vs. supply heatmaps + captured-capacity bar
chart, from results/ipc_matching.json (scripts/sprint5_ipc_matching.py).
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


def plot_demand_supply_heatmaps(result):
    apply_style()
    demand = result["demand"]
    best = result["best_matched_config"]
    best_entry = next(e for e in result["supply_grid"]
                       if e["gamma1"] == best["gamma1"] and e["a"] == best["a"])
    ref = result["reference_config"]
    ref_entry = next(e for e in result["supply_grid"]
                      if e["gamma1"] == ref["gamma1"] and e["a"] == ref["a"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    degrees = demand["1"]["degrees"]
    delays = demand["1"]["delays"]

    for col, h in enumerate(["1", "6"]):
        ax = axes[0, col]
        D = np.array(demand[h]["D"])
        im = ax.imshow(D, aspect="auto", cmap="viridis", vmin=0)
        ax.set_yticks(range(len(degrees)))
        ax.set_yticklabels(degrees)
        ax.set_xlabel("delay (h)")
        ax.set_ylabel("Legendre degree")
        ax.set_xticks(range(0, len(delays), 4))
        ax.set_xticklabels([delays[i] for i in range(0, len(delays), 4)])
        fig.colorbar(im, ax=ax, fraction=0.046)
        label_panel(ax, "a" if col == 0 else "b")

    for col, (entry, name) in enumerate([(best_entry, "matched"), (ref_entry, "reference")]):
        ax = axes[1, col]
        C = np.array(entry["C"])
        im = ax.imshow(C, aspect="auto", cmap="magma", vmin=0)
        ax.set_yticks(range(len(entry["lags"])) if False else range(len(degrees)))
        ax.set_yticklabels(degrees)
        ax.set_xlabel("lag (steps)")
        ax.set_ylabel("Hermite degree")
        lags = entry["lags"]
        ax.set_xticks(range(0, len(lags), 4))
        ax.set_xticklabels([lags[i] for i in range(0, len(lags), 4)])
        fig.colorbar(im, ax=ax, fraction=0.046)
        label_panel(ax, "c" if col == 0 else "d")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "sprint5_demand_supply_heatmaps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_captured_capacity_bar(result):
    apply_style()
    scores = result["matching_scores"]
    labels = [f"g1={s['gamma1']:.2f}\na={s['a']:.1f}" for s in scores]
    values = [s["captured_total"] for s in scores]
    best_idx = int(np.argmax(values))

    fig, ax = plt.subplots(figsize=(max(6, len(scores) * 0.6), 4))
    colors = [PALETTE[1] if i == best_idx else PALETTE[5] for i in range(len(scores))]
    ax.bar(range(len(scores)), values, color=colors)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_ylabel("captured capacity  Σ min(C, D)  (h=1 + h=6)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "sprint5_captured_capacity_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found -- run scripts/sprint5_ipc_matching.py first.")
    with open(RESULTS_PATH) as f:
        result = json.load(f)
    if "best_matched_config" not in result:
        raise RuntimeError(f"{RESULTS_PATH} looks like a partial/in-progress checkpoint "
                            "(no best_matched_config yet) -- wait for the run to finish.")
    plot_demand_supply_heatmaps(result)
    plot_captured_capacity_bar(result)


if __name__ == "__main__":
    main()
