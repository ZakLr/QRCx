#!/usr/bin/env python3
"""Sprint 2 Phase 2.2: IPC/MC characterization + the paper's key
memory-nonlinearity trade-off figure (total IPC and its linear/nonlinear
split vs gamma1, for input_scaling a in {0.3, 1.0, 3.0}).

Uses QRCx/experiment/figstyle.py (serif, no in-figure titles, panel labels,
300 dpi, colorblind-safe Okabe-Ito palette) throughout.

Every experiment script accepts FAST_MODE: FAST_MODE=True uses fewer
steps/lags for the IPC/MC estimate (fast but noisier); FAST_MODE=False
uses the paper-grade settings.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from QRCx.experiment.figstyle import apply_style, PALETTE, label_panel
from QRCx.metrics.reservoir_sequential import (
    drive_iid_gaussian, measure_memory_capacity_sequential, measure_ipc_sequential,
)
from QRCx.reservoir.sequential import SequentialDissipativeQRC

N_QUBITS = 10
W_IN_SEED = 123
GAMMA1_GRID = [0.0, 0.01, 0.03, 0.1, 0.3]
A_GRID = [0.3, 1.0, 3.0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false")
    parser.add_argument("--gamma2", type=float, default=0.03, help="Fixed gamma2 for this characterization")
    args = parser.parse_args()
    FAST_MODE = args.fast_mode

    n_steps = 200 if FAST_MODE else 500
    max_lag_ipc = 8 if FAST_MODE else 15
    max_lag_mc = 12 if FAST_MODE else 20

    w_in = np.random.default_rng(W_IN_SEED).uniform(0.5, 1.5, size=N_QUBITS)

    records = []
    for a in A_GRID:
        for gamma1 in GAMMA1_GRID:
            qrc = SequentialDissipativeQRC(
                n_qubits=N_QUBITS, trotter_steps=10, gamma1=gamma1, gamma2=args.gamma2,
                injection="ry", washout=0, seed=42, input_scaling=a,
                propagator="exact", multiplexing=1, w_in=w_in,
            )
            u, features = drive_iid_gaussian(qrc, n_steps=n_steps, seed=0)
            # Sprint 2.5 Phase C: Dambre et al. 2012 shuffle-surrogate
            # thresholding -- capacities not distinguishable from an
            # input-shuffled null (95th percentile) are zeroed.
            mc = measure_memory_capacity_sequential(
                max_lag=max_lag_mc, u=u, features=features,
                threshold_surrogates=True, n_surrogates=20,
            )
            ipc = measure_ipc_sequential(
                max_lag=max_lag_ipc, u=u, features=features,
                threshold_surrogates=True, n_surrogates=20,
            )
            record = {
                "a": a, "gamma1": gamma1, "gamma2": args.gamma2,
                "MC": mc["MC"], "MC_raw": mc["MC_raw"],
                "linear_ipc": ipc["linear_ipc"], "nonlinear_ipc": ipc["nonlinear_ipc"],
                "total_ipc": ipc["total_ipc"], "total_ipc_raw": ipc["total_ipc_raw"],
            }
            records.append(record)
            print(json.dumps(record))

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "ipc_mc_characterization.json"
    out_path.write_text(json.dumps({
        "FAST_MODE": FAST_MODE, "n_qubits": N_QUBITS, "n_steps": n_steps,
        "max_lag_ipc": max_lag_ipc, "max_lag_mc": max_lag_mc,
        "w_in_seed": W_IN_SEED, "records": records,
    }, indent=2))
    print(f"\nWritten to {out_path}")

    # -- the trade-off figure -----------------------------------------
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for i, a in enumerate(A_GRID):
        sub = [r for r in records if r["a"] == a]
        g1 = [r["gamma1"] for r in sub]
        total = [r["total_ipc"] for r in sub]
        lin = [r["linear_ipc"] for r in sub]
        nonlin = [r["nonlinear_ipc"] for r in sub]
        axes[0].plot(g1, total, marker="o", color=PALETTE[i], label=f"a={a}")
        axes[1].plot(g1, nonlin, marker="o", color=PALETTE[i], label=f"a={a} (nonlinear)")
        axes[1].plot(g1, lin, marker="s", linestyle="--", color=PALETTE[i], alpha=0.6, label=f"a={a} (linear)")

    axes[0].set_xlabel(r"Amplitude damping rate $\gamma_1$")
    axes[0].set_ylabel("Total IPC (shuffle-surrogate thresholded, p95)")
    axes[0].legend(fontsize=8)
    label_panel(axes[0], "a")

    axes[1].set_xlabel(r"Amplitude damping rate $\gamma_1$")
    axes[1].set_ylabel("IPC (linear vs. nonlinear)")
    axes[1].legend(fontsize=7)
    label_panel(axes[1], "b")

    fig.tight_layout()
    fig_dir = REPO_ROOT / "qrc_figures"
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / "fig_ipc_tradeoff.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure written to {fig_path}")

    # Report honestly whether the interior optimum expected by the sprint
    # spec (Cindrak-style reset-length peak) actually appears, per gamma1=0
    # instruction: "If the optimum is at gamma1=0, report it honestly."
    for a in A_GRID:
        sub = [r for r in records if r["a"] == a]
        best = max(sub, key=lambda r: r["total_ipc"])
        print(f"a={a}: best gamma1={best['gamma1']} (total_ipc={best['total_ipc']:.4f}, "
              f"raw={best['total_ipc_raw']:.4f}); "
              f"{'INTERIOR OPTIMUM' if 0 < best['gamma1'] < GAMMA1_GRID[-1] else 'BOUNDARY (gamma1=0 or max) -- reporting honestly, not forcing an interior claim'}")

    # Does the monotonic-in-gamma1 trend (found without thresholding in
    # Sprint 2) survive thresholding? Compare thresholded vs raw ordering.
    for a in A_GRID:
        sub = sorted([r for r in records if r["a"] == a], key=lambda r: r["gamma1"])
        thresholded_monotonic = all(
            sub[i]["total_ipc"] <= sub[i + 1]["total_ipc"] for i in range(len(sub) - 1)
        )
        raw_monotonic = all(
            sub[i]["total_ipc_raw"] <= sub[i + 1]["total_ipc_raw"] for i in range(len(sub) - 1)
        )
        print(f"a={a}: monotonic increase in gamma1 -- thresholded={thresholded_monotonic}, raw={raw_monotonic}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
