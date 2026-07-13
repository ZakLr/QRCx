"""Academic figure style (Sprint 2 Phase 2.2): matplotlib, serif fonts, no
titles-in-figure, labeled panels (a)/(b)/..., 300 dpi, colorblind-safe
palette (Okabe & Ito 2008, the standard colorblind-safe qualitative
palette used widely in physics/CS papers).

Usage:
    from QRCx.experiment.figstyle import apply_style, PALETTE, label_panel

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax.plot(x, y, color=PALETTE[0], label="...")
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.savefig(path, dpi=300, bbox_inches="tight")

Caption text (including any "Figure N" label) belongs in the paper/README,
not baked into the figure via `ax.set_title`/`fig.suptitle` -- this module
does not provide a title helper on purpose.
"""
import matplotlib
import matplotlib.pyplot as plt

# Okabe & Ito (2008) colorblind-safe qualitative palette.
PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]


def apply_style() -> None:
    """Apply the project-wide academic figure style. Call once before
    plotting; safe to call multiple times."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "axes.titlesize": 0,  # discourage in-figure titles (see module docstring)
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "legend.frameon": False,
    })


def label_panel(ax, letter: str, x: float = -0.12, y: float = 1.05) -> None:
    """Add a (a)/(b)/... panel label in the upper-left corner of `ax`,
    the academic-figure convention used instead of a per-axes title."""
    ax.text(
        x, y, f"({letter})", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="right",
    )
