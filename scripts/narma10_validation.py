#!/usr/bin/env python3
"""Sprint 1 micro-validation: SequentialDissipativeQRC vs a linear AR
readout baseline on NARMA10 (Jaeger 2001 / Fujii & Nakajima 2017 standard
reservoir-computing sanity check).

Every experiment script accepts FAST_MODE (per project convention):
FAST_MODE=True uses a short sequence and the runtime-gate fallback of
10 qubits (see docs/sprint_log/SPRINT_1_REPORT.md for why: the reference
12-qubit numpy backend measured ~120 s/step, making even this short
micro-validation impractical within the sprint's compute budget).
FAST_MODE=False attempts the full run at 12 qubits (very slow; not
recommended without dedicated compute).

Driven features are cached to results/narma10_features.npz so that the
(cheap) readout/regularization step can be re-run without repeating the
(expensive) reservoir drive -- use --use-cache to skip driving.

Usage:
    python scripts/narma10_validation.py [--fast-mode / --no-fast-mode] [--use-cache]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

from QRCx.data.narma import generate_narma10
from QRCx.reservoir.sequential import SequentialDissipativeQRC

CACHE_PATH = REPO_ROOT / "results" / "narma10_features.npz"
ALPHA_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0]


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def ar_baseline_nmse(u: np.ndarray, y: np.ndarray, order: int, train_frac: float) -> float:
    """Linear AR readout: predict y[t] from y[t-1..t-order] and u[t-1..t-order]."""
    n = len(y)
    X = np.stack(
        [np.concatenate([y[t - order:t], u[t - order:t]]) for t in range(order, n)]
    )
    Y = y[order:]
    n_train = int(len(Y) * train_frac)
    model = Ridge(alpha=1e-3).fit(X[:n_train], Y[:n_train])
    pred = model.predict(X[n_train:])
    return nmse(Y[n_train:], pred)


def select_alpha_by_cv(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """TimeSeriesSplit CV over ALPHA_GRID (project convention: no shuffling)."""
    tscv = TimeSeriesSplit(n_splits=3)
    best_alpha, best_score = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            model = Ridge(alpha=alpha).fit(X_train[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train[val_idx])
            scores.append(nmse(y_train[val_idx], pred))
        mean_score = float(np.mean(scores))
        if mean_score < best_score:
            best_score, best_alpha = mean_score, alpha
    return best_alpha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false")
    parser.add_argument("--use-cache", action="store_true",
                         help="Reuse cached driven features instead of re-driving the reservoir")
    args = parser.parse_args()
    FAST_MODE = args.fast_mode

    if FAST_MODE:
        n_qubits = 10  # Sprint 1 runtime-gate fallback; see sprint log
        n_steps = 400
        washout = 10
    else:
        n_qubits = 12
        n_steps = 300
        washout = 24

    u, y = generate_narma10(n_steps=n_steps + washout, seed=42)

    if args.use_cache and CACHE_PATH.exists():
        cached = np.load(CACHE_PATH)
        features, elapsed = cached["features"], float(cached["elapsed"])
        print(f"Using cached features from {CACHE_PATH}")
    else:
        qrc = SequentialDissipativeQRC(
            n_qubits=n_qubits, trotter_steps=10, gamma1=0.05, gamma2=0.02,
            injection="ry", washout=washout, seed=42, input_scaling=3.0,
        )
        # NARMA10 is scalar-input. Broadcasting the identical scalar onto
        # every qubit's RY angle (tried first) makes the whole reservoir
        # permutation-symmetric under the uniform TFIM coupling: every
        # qubit is in an identical role, so the 165 (n=10) Pauli
        # correlators collapse to ~6 distinct repeated values with tiny
        # variance across time -- confirmed by inspecting
        # results/narma10_features.npz, std ~0.03 and many features
        # numerically identical -- which is why that version scored
        # NMSE > 1 (worse than predicting the mean). Fixed the standard
        # reservoir-computing way: a fixed random per-qubit input-weight
        # vector (analogous to an ESN's random W_in), which breaks the
        # qubit-exchange symmetry so each qubit sees a different scaled
        # copy of u.
        rng_w = np.random.default_rng(123)
        input_weights = rng_w.uniform(0.5, 1.5, size=n_qubits)
        seq_input = u[:, None] * input_weights[None, :]

        t0 = time.perf_counter()
        features = qrc.drive(seq_input)
        elapsed = time.perf_counter() - t0
        CACHE_PATH.parent.mkdir(exist_ok=True)
        np.savez(CACHE_PATH, features=features, elapsed=elapsed)

    steps_per_sec = len(u) / elapsed

    features = features[washout:]
    y_eval = y[washout:]
    n_train = int(len(y_eval) * 0.7)

    best_alpha = select_alpha_by_cv(features[:n_train], y_eval[:n_train])
    reservoir_model = Ridge(alpha=best_alpha).fit(features[:n_train], y_eval[:n_train])
    reservoir_pred = reservoir_model.predict(features[n_train:])
    reservoir_nmse = nmse(y_eval[n_train:], reservoir_pred)

    ar_nmse = ar_baseline_nmse(u, y, order=10, train_frac=0.7)

    result = {
        "FAST_MODE": FAST_MODE,
        "n_qubits": n_qubits,
        "n_steps": n_steps,
        "washout": washout,
        "n_train": n_train,
        "n_test": len(y_eval) - n_train,
        "n_features": features.shape[1],
        "ridge_alpha_selected_by_tscv": best_alpha,
        "drive_wall_clock_s": elapsed,
        "steps_per_sec": steps_per_sec,
        "reservoir_nmse": reservoir_nmse,
        "ar_baseline_nmse": ar_nmse,
        "beats_ar_baseline": reservoir_nmse < ar_nmse,
        "target_nmse": 0.4,
        "meets_target": reservoir_nmse < 0.4,
    }

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "narma10_validation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
