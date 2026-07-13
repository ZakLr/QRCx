#!/usr/bin/env python3
"""
Qbraid.py — QRC weather forecasting using the QRCx package API.

Demonstrates a full pipeline: ISD data loading → preprocessing →
TFIM quantum reservoir → feature extraction → DirectQRC / ResidualQRC
readouts → baselines (Persistence, ARIMA, ESN) → evaluation.

All settings are controlled by ``ExperimentConfig`` at the bottom;
override any field to customise.

Usage:
    python Qbraid.py
"""

import sys, os, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from QRCx import (
    ExperimentConfig,
    QRCPipeline,
    AtmosphericQRC,
    KRRReadout,
    ResidualQRC,
    rmse, mae, skill_score,
    compute_fsdh_curve, compute_vpt_curve,
    verify_esp, measure_memory_capacity,
)
from sklearn.linear_model import Ridge


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

cfg = ExperimentConfig(
    # ── Data ──────────────────────────────────────────────────────────────
    train_years=(2019, 2019),    # (start_year, end_year) for training
    val_year=2020,               # single year for validation
    test_year=2021,              # single year for testing
    target_col="T_db",           # target variable name in the DataFrame

    # ── Quantum circuit ──────────────────────────────────────────────────
    n_qubits=12,                 # number of qubits
    n_layers=2,                  # encoding/evolution layers
    trotter_steps=4,             # Trotter steps for TFIM evolution

    # ── Readout ──────────────────────────────────────────────────────────
    readout="krr",               # readout type: "krr", "ridge", "quantum_ridge", "mlp", "lasso"

    # ── Architecture ─────────────────────────────────────────────────────
    architecture="residual",     # "residual", "direct", or "parallel"

    # ── Baselines ────────────────────────────────────────────────────────
    baselines=["persistence", "arima", "esn500"],

    # ── Evaluation ───────────────────────────────────────────────────────
    horizons=[1, 6],             # forecast horizons (hours)
    metrics=["rmse", "mae", "skill", "fsdh", "vpt"],

    # ── Diagnostics ──────────────────────────────────────────────────────
    noise_sweep=False,

    # ── Paths ────────────────────────────────────────────────────────────
    cache_dir="./cache",
    figures_dir="./qrc_figures",

    # ── Reproducibility ──────────────────────────────────────────────────
    seed=42,
)


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RESERVOIR (when not using QRCPipeline directly)
# ══════════════════════════════════════════════════════════════════════════════

def make_windows(arr: np.ndarray, window: int = 24,
                 horizons: list = None, target_col: int = 0):
    """Create sliding windows from a (N, F) array for multi-horizon targets."""
    if horizons is None:
        horizons = [1, 6]
    mx = max(horizons)
    xl, yl = [], []
    for i in range(window, len(arr) - mx + 1):
        w = arr[i - window:i]
        t = np.array([arr[i + h - 1, target_col] for h in horizons])
        if not (np.isnan(w).any() or np.isnan(t).any()):
            xl.append(w); yl.append(t)
    return np.array(xl), np.array(yl)


def main():
    print("=" * 70)
    print("  Qbraid — QRC Weather Forecast (QRCx package API)")
    print(f"  Config: {cfg.n_qubits} qubits, {cfg.architecture} architecture, "
          f"{cfg.readout} readout")
    print(f"  Horizons: {cfg.horizons}h  Target: {cfg.target_col}")
    print("=" * 70)

    # ── 1. Data ──
    print(f"\n[1/6] Loading ISD KORD data ({cfg.train_years[0]}–{cfg.test_year})...")
    from QRCx.data.loader import load_isd_range
    from QRCx.data.preprocessor import preprocess

    df = load_isd_range(start_year=cfg.train_years[0], end_year=cfg.test_year)
    if df is None or len(df) < 1000:
        raise RuntimeError(f"ISD data insufficient: {len(df)} rows")

    # ── 2. Preprocess ──
    print("\n[2/6] Preprocessing...")
    data = preprocess(
        df,
        target_col=cfg.target_col,
        train_years=cfg.train_years,
        val_year=cfg.val_year,
        test_year=cfg.test_year,
        W=24,
        horizons=cfg.horizons,
    )
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"],   data["y_val"]
    X_te, y_te = data["X_test"],  data["y_test"]
    tci = data["target_col_idx"]
    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    # ── 3. Reservoir ──
    print(f"\n[3/6] Building {cfg.n_qubits}-qubit TFIM reservoir...")
    qrc = AtmosphericQRC(
        n_qubits=cfg.n_qubits,
        n_layers=cfg.n_layers,
        trotter_steps=cfg.trotter_steps,
        seed=cfg.seed,
    )
    print(f"  J/g ratio: {qrc.jg_ratio:.3f}")

    # ── 4. Extract features ──
    print("\n[4/6] Extracting reservoir features...")
    F_tr = qrc.transform(X_tr)
    F_te = qrc.transform(X_te)

    # ── 5. Models ──
    print("\n[5/6] Training models...")

    # ResidualQRC
    print("  ResidualQRC...")
    residual = ResidualQRC(qrc, KRRReadout(), target_col_idx=tci)
    residual.fit(X_tr, y_tr, X_va, y_va)
    y_residual = residual.predict(X_te)
    if y_residual.ndim == 1:
        y_residual = y_residual[:, np.newaxis]

    # DirectQRC (cached features, per-horizon KRR)
    print("  DirectQRC...")
    d_readouts = {}
    for h_idx in range(y_tr.shape[1]):
        krr = KRRReadout()
        krr.fit(F_tr, y_tr[:, h_idx])
        d_readouts[h_idx] = krr
    y_direct = np.column_stack(
        [d_readouts[h].predict(F_te) for h in range(y_tr.shape[1])]
    )

    # Persistence baseline
    print("  Persistence...")
    persist_val = X_te[:, -1, tci, np.newaxis]
    y_persist = np.tile(persist_val, (1, y_tr.shape[1]))

    # ── 6. Evaluation ──
    print("\n[6/6] Evaluation\n")

    for h_idx, h_name in enumerate([f"{h}h" for h in cfg.horizons]):
        print(f"  Horizon {h_name}:")
        print(f"  {'Model':<18} {'RMSE':>8} {'MAE':>8} {'Skill%':>8}")
        print(f"  {'-' * 46}")
        for name, preds in [("ResidualQRC", y_residual),
                            ("DirectQRC", y_direct),
                            ("Persistence", y_persist)]:
            gt = y_te[:, h_idx]
            r = rmse(gt, preds[:, h_idx])
            m = mae(gt, preds[:, h_idx])
            s = skill_score(gt, preds[:, h_idx], y_persist[:, h_idx])
            print(f"  {name:<18} {r:>8.4f} {m:>8.4f} {s * 100:>7.1f}%")

    print("\n  FSDH & VPT (NRMSE < 0.5):")
    for name, preds in [("ResidualQRC", y_residual),
                        ("DirectQRC", y_direct),
                        ("Persistence", y_persist)]:
        fsdh_h = compute_fsdh_curve(y_te, preds, y_persist)
        vpt_h = compute_vpt_curve(y_te, preds, threshold=0.5)
        print(f"  {name:<18} FSDH={fsdh_h}h  VPT={vpt_h}h")

    # Diagnostics
    print("\n  Diagnostics:")
    esp_in = np.random.default_rng(cfg.seed).normal(size=(50, 24, 13))
    _, esp_ok, _ = verify_esp(qrc, esp_in, n_initial_states=5,
                              n_steps=20, convergence_threshold=0.5)
    print(f"  ESP converged: {esp_ok}")

    mc = measure_memory_capacity(qrc, max_lag=10, n_samples=150)
    # measure_memory_capacity from the package returns a float (MC value)
    mc_val = mc if isinstance(mc, (int, float)) else mc.get("MC", 0.0)
    print(f"  Memory Capacity: {mc_val:.3f}  "
          f"(target > {cfg.n_qubits / 2:.0f})")

    print("\n" + "=" * 70)
    print("  V Qbraid demo complete.")
    print("  V To change parameters, edit ``cfg`` at the top of this file.")
    print("=" * 70)


if __name__ == "__main__":
    main()
