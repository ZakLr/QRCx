#!/usr/bin/env python3
"""Sprint 9: architecture schematic for the paper (Sec. 2) -- the
sequential dissipative TFIM reservoir pipeline: input -> RY/ZZ injection
-> TFIM coherent evolution -> dissipation (amplitude damping + dephasing)
-> time-multiplexed Pauli-correlator readout -> Ridge (residual
decomposition) -> forecast."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.experiment.figstyle import apply_style, PALETTE

FIG_DIR = REPO_ROOT / "figures"


def box(ax, xy, w, h, text, color, text_color="black"):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                    linewidth=1.2, edgecolor="black", facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9, color=text_color)


def arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", lw=1.2))


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 3.6))

    stages = [
        ("real weather\nfeature x_k\n(13-dim, hourly)", PALETTE[4], "black"),
        ("RY/ZZ injection\n(input_scaling a,\nw_in per qubit)", PALETTE[2], "black"),
        ("TFIM coherent\nevolution\n" + r"$e^{-iH\tau}$", PALETTE[5], "white"),
        ("dissipation D\n(amp. damping " + r"$\gamma_1$" + ",\ndephasing " + r"$\gamma_2$" + ")", PALETTE[6], "white"),
        ("time-multiplexed\nreadout (V steps)\nPauli correlators", PALETTE[3], "white"),
        ("Ridge readout\n(residual\ndecomposition)", PALETTE[1], "black"),
        ("forecast\n" + r"$\hat{y}_{k+h}$", "white", "black"),
    ]
    w, h, gap = 1.5, 1.3, 0.3
    n = len(stages)
    total_w = n * w + (n - 1) * gap
    x0 = 0.0
    y = 0.9
    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    x = x0
    centers = []
    for text, color, text_color in stages:
        box(ax, (x, y), w, h, text, color, text_color)
        centers.append((x, x + w))
        x += w + gap
    for i in range(len(centers) - 1):
        arrow(ax, (centers[i][1], y + h / 2), (centers[i + 1][0], y + h / 2))
    # feedback loop: rho carries state to the next timestep k+1
    left = centers[1][0] + 0.1
    right = centers[3][1] - 0.1
    ax.annotate("", xy=(left, y + h + 0.05), xytext=(right, y + h + 0.05),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=PALETTE[0], linestyle="--",
                                 connectionstyle="arc3,rad=-0.35"))
    ax.text((left + right) / 2, y + h + 0.85,
            r"$\rho_k$ carries forward to step $k{+}1$ (recurrent state)",
            ha="center", fontsize=9, color=PALETTE[0])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "architecture_diagram.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    out_pdf = FIG_DIR / "architecture_diagram.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
