from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_architecture(output_dir: str = "./figures") -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Direct QRC", "Residual QRC", "Parallel QRC"]
    for ax, title in zip(axes, titles):
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle("Fig 1: QRC Architectures", fontsize=16)
    fig.tight_layout()
    path = _ensure_output(Path(output_dir)) / "fig1_architecture.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_esp_convergence(distances: np.ndarray, threshold: float = 1e-2, output_dir: str = "./figures") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(distances, color="b", linewidth=2)
    ax.axhline(threshold, color="r", linestyle="--", label=f"threshold={threshold}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean pairwise L2 distance")
    ax.set_title("Fig 2: ESP Convergence")
    ax.legend()
    ax.grid(True)
    path = _ensure_output(Path(output_dir)) / "fig2_esp_convergence.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_ipc_and_vpt(ipc_results: dict, vpt_results: dict, output_dir: str = "./figures") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if ipc_results:
        lags = list(ipc_results.keys())
        values = list(ipc_results.values())
        ax1.bar(lags, values, color="steelblue")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("IPC")
        ax1.set_title("Fig 3a: IPC per Lag")

    if vpt_results:
        models = list(vpt_results.keys())
        scores = list(vpt_results.values())
        ax2.bar(models, scores, color=["green", "orange", "red"])
        ax2.set_xlabel("Model")
        ax2.set_ylabel("VPT")
        ax2.set_title("Fig 3b: Lorenz-63 VPT Comparison")

    fig.suptitle("Fig 3: IPC and VPT", fontsize=16)
    fig.tight_layout()
    path = _ensure_output(Path(output_dir)) / "fig3_ipc_vpt.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_noise_sweep(p_values: list, rmse_values: list, output_dir: str = "./figures") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, rmse_values, "o-", color="darkred", linewidth=2)
    ax.set_xlabel("Depolarizing probability p")
    ax.set_ylabel("RMSE")
    ax.set_title("Fig 4: Noise Sweep")
    ax.grid(True)
    path = _ensure_output(Path(output_dir)) / "fig4_noise_sweep.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_results_table(results: dict, output_dir: str = "./figures") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    rows = []
    for model, metrics in results.items():
        if isinstance(metrics, dict):
            row = [model]
            for h in [1, 6]:
                h_metrics = metrics.get(h, {})
                row.append(f'{h_metrics.get("rmse", "N/A"):.3f}' if isinstance(h_metrics.get("rmse"), (int, float)) else "N/A")
                row.append(f'{h_metrics.get("mae", "N/A"):.3f}' if isinstance(h_metrics.get("mae"), (int, float)) else "N/A")
                row.append(f'{h_metrics.get("skill", "N/A"):.3f}' if isinstance(h_metrics.get("skill"), (int, float)) else "N/A")
            rows.append(row)

    if rows:
        columns = ["Model", "RMSE 1h", "MAE 1h", "Skill 1h", "RMSE 6h", "MAE 6h", "Skill 6h"]
        table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(columns))))

    ax.set_title("Fig 5: Results Summary", fontsize=14, pad=20)
    path = _ensure_output(Path(output_dir)) / "fig5_results_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    return fig


def plot_fsdh_bar(fsdh_results: dict, output_dir: str = "./figures") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(fsdh_results.keys())
    hours = [fsdh_results[m] if isinstance(fsdh_results[m], (int, float)) else 0 for m in models]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    ax.bar(models, hours, color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("FSDH (hours)")
    ax.set_title("Fig 6: FSDH Comparison")
    ax.tick_params(axis="x", rotation=45)
    for i, h in enumerate(hours):
        ax.text(i, h + 0.5, str(h), ha="center", fontsize=10)
    fig.tight_layout()
    path = _ensure_output(Path(output_dir)) / "fig6_fsdh_bar.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig
