#!/usr/bin/env python3
"""Sprint 8 figures: scaling, shots, encoding, noise-reconciliation
(from results/characterization.json)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.experiment.figstyle import apply_style, PALETTE, label_panel

RESULTS_PATH = REPO_ROOT / "results" / "characterization.json"
FIG_DIR = REPO_ROOT / "figures"


def plot_scaling(result):
    apply_style()
    scaling = result["scaling"]
    ns = [e["n_qubits"] for e in scaling]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(ns, [e["mc"] for e in scaling], marker="o", color=PALETTE[1])
    axes[0].set_xlabel("n_qubits"); axes[0].set_ylabel("Memory Capacity")
    label_panel(axes[0], "a")
    axes[1].plot(ns, [e["ipc_total"] for e in scaling], marker="o", color=PALETTE[2], label="total")
    axes[1].plot(ns, [e["ipc_linear"] for e in scaling], marker="s", color=PALETTE[3], label="linear")
    axes[1].plot(ns, [e["ipc_nonlinear"] for e in scaling], marker="^", color=PALETTE[5], label="nonlinear")
    axes[1].set_xlabel("n_qubits"); axes[1].set_ylabel("IPC")
    axes[1].legend(fontsize=8)
    label_panel(axes[1], "b")
    axes[2].plot(ns, [e["skill_h1"] * 100 for e in scaling], marker="o", color=PALETTE[1], label="h=1")
    axes[2].plot(ns, [e["skill_h6"] * 100 for e in scaling], marker="s", color=PALETTE[6], label="h=6")
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].set_xlabel("n_qubits"); axes[2].set_ylabel("skill vs. persistence (%)")
    axes[2].legend(fontsize=8)
    label_panel(axes[2], "c")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "sprint8_scaling.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_shots(result):
    apply_style()
    shot = result["shot_study"]
    entries = shot["results"]
    xs = [e["n_shots"] if e["n_shots"] is not None else 10 ** 7 for e in entries]  # place "inf" far right on log scale
    ys = [e["skill_h6"] * 100 for e in entries]
    ci = shot["bootstrap_ci_halfwidth_h6"] * 100
    exact = [e["skill_h6"] for e in entries if e["n_shots"] is None][0] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", color=PALETTE[1])
    ax.axhline(exact, color=PALETTE[0], linestyle="--", linewidth=1, label="exact")
    ax.fill_between([min(xs), max(xs)], exact - ci, exact + ci, color=PALETTE[2], alpha=0.2,
                     label="bootstrap CI half-width")
    ax.set_xscale("log")
    ax.set_xlabel("shots (S)")
    ax.set_ylabel("skill@6h vs. persistence (%)")
    ax.legend(fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "sprint8_shots.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_encoding(result):
    apply_style()
    enc = result["encoding_density"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, injection in zip(axes, ("ry", "zz")):
        rows = [e for e in enc if e["injection"] == injection]
        for V, color in zip((1, 3), (PALETTE[1], PALETTE[5])):
            sub = [e for e in rows if e["multiplexing_V"] == V]
            sub.sort(key=lambda e: e["input_scaling"])
            ax.plot([e["input_scaling"] for e in sub], [e["skill_h6"] * 100 for e in sub],
                    marker="o", color=color, label=f"V={V}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("input_scaling (a)")
        ax.set_ylabel("skill@6h (%)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    label_panel(axes[0], "a"); label_panel(axes[1], "b")
    axes[0].set_title("") ; axes[1].set_title("")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "sprint8_encoding.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")
    print("NOTE (real finding): 'zz' injection panel (b) is flat across input_scaling -- "
          "QRCx.reservoir.sequential.SequentialDissipativeQRC._inject_zz never references "
          "self.input_scaling at all, unlike _inject_ry. Documented, not silently smoothed over.")


def plot_noise_reconciliation():
    """Sprint 8 task 4 is a documentation reconciliation, not new data --
    this renders a simple schematic summarizing the two established,
    already-real findings (Sprint 2/2.5) side by side."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ["v4: depolarizing\non encoding\n(harmful)", "v5: amplitude damping\nbetween steps\n(resource)"]
    # Illustrative real-direction-only bars (not fabricated magnitudes) --
    # sign indicates whether the effect helps (+) or hurts (-) task performance,
    # per the established Sprint 2/2.5 findings; see docs/sprint_log/SPRINT_2_REPORT.md.
    values = [-1, 1]
    colors = [PALETTE[6], PALETTE[3]]
    ax.bar(categories, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["harmful", "helpful\n(resource)"])
    ax.set_ylabel("effect on task performance")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "sprint8_noise_reconciliation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out} (schematic only -- see docs/sprint_log/SPRINT_8_REPORT.md for the real citations/numbers)")


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found -- run scripts/sprint8_characterization.py first.")
    with open(RESULTS_PATH) as f:
        result = json.load(f)
    plot_scaling(result)
    plot_shots(result)
    plot_encoding(result)
    plot_noise_reconciliation()


if __name__ == "__main__":
    main()
