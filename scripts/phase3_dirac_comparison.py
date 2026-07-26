#!/usr/bin/env python3
"""FINAL SPRINT Phase 3: Dirac-3 readout-sparsification attempt, with a
real device-access attempt (error logged verbatim if it fails) and a
full RMSE-vs-K comparison via the local SA stand-in, benchmarked
against Lasso and greedy forward selection at equal K. Uses the real
driven v5 pilot features from Phase 2 (results/phase2_features/,
N=10 qubits, gamma1=0.03 "tuned") as the frozen feature set -- the
same features already used for Phase 2's A/B/C/C' experiment, so
Stage 2's input here is the exact Stage 1 output the pipeline framing
(GPU reservoir -> Dirac-3 sparsification -> classical ridge) requires."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.readout.dirac3_selector import build_problem, select

FEAT_DIR = REPO_ROOT / "results" / "phase2_features"
OUT_DIR = REPO_ROOT / "results" / "dirac3"
K_VALUES = [32, 64, 128]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def greedy_forward(F_train, y_train, K):
    n = F_train.shape[1]
    y = y_train - y_train.mean()
    remaining = list(range(n))
    selected = []
    resid = y.copy()
    for _ in range(K):
        Fc = F_train[:, remaining] - F_train[:, remaining].mean(axis=0, keepdims=True)
        norms = np.linalg.norm(Fc, axis=0) + 1e-12
        corr = np.abs(Fc.T @ resid) / norms
        best_local = int(np.argmax(corr))
        best_global = remaining.pop(best_local)
        selected.append(best_global)
        ridge = Ridge(alpha=1.0).fit(F_train[:, selected], y_train)
        resid = y_train - ridge.predict(F_train[:, selected]) + y_train.mean() - ridge.intercept_
        resid = y_train - ridge.predict(F_train[:, selected])
    return np.array(selected)


def lasso_topk(F_train, y_train, K):
    model = LassoCV(cv=5, max_iter=50000).fit(F_train, y_train)
    coefs = np.abs(model.coef_)
    return np.argsort(-coefs)[:K]


def eval_selection(idx, F_train, y_train, F_eval, y_eval):
    ridge = Ridge(alpha=1.0).fit(F_train[:, idx], y_train)
    pred = ridge.predict(F_eval[:, idx])
    return rmse(y_eval, pred)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    B = np.load(FEAT_DIR / "v5_features_tuned.npy").astype(np.float64)
    target = np.load(FEAT_DIR / "target_series.npy").astype(np.float64)
    meta = json.load(open(FEAT_DIR / "meta.json"))
    n_train = meta["n_train"]
    y = target[1:]  # 1-step-ahead target, aligned to features at t (predict t+1)
    F = B[:-1]
    n_fit = int(0.75 * len(F))
    F_train, y_train = F[:n_fit], y[:n_fit]
    F_eval, y_eval = F[n_fit:], y[n_fit:]

    print(f"Feature set: {F.shape}, fit={len(F_train)}, eval={len(F_eval)}")
    c, Q = build_problem(F_train, y_train)

    # Real Dirac-3 attempt -- genuine, not simulated. Logs the exact
    # error if credentials/device access are unavailable.
    dirac3_attempt = select("dirac3", c, Q, K=32, cache_dir=OUT_DIR)
    dirac3_log = {
        "attempted": True,
        "solver": dirac3_attempt.solver,
        "error": dirac3_attempt.error,
        "wall_clock_s": dirac3_attempt.wall_clock_s,
        "succeeded": dirac3_attempt.error is None,
    }
    print(f"Dirac-3 real attempt: {'SUCCEEDED' if dirac3_log['succeeded'] else 'FAILED'}")
    if dirac3_attempt.error:
        print(f"  error: {dirac3_attempt.error}")

    results = {"dirac3_attempt": dirac3_log, "by_K": {}}
    for K in K_VALUES:
        print(f"\n=== K={K} ===")
        row = {}

        t0 = time.perf_counter()
        sa_result = select("sa", c, Q, K=K, seed=42)
        idx_sa = sa_result.selected_idx
        row["sa"] = {
            "rmse": eval_selection(idx_sa, F_train, y_train, F_eval, y_eval),
            "energy": sa_result.energy,
            "wall_clock_s": sa_result.wall_clock_s,
            "device_params": sa_result.device_params,
        }
        print(f"  SA:      rmse={row['sa']['rmse']:.4f}  energy={sa_result.energy:.3f}  "
              f"({sa_result.wall_clock_s:.1f}s)")

        t0 = time.perf_counter()
        idx_lasso = lasso_topk(F_train, y_train, K)
        row["lasso"] = {"rmse": eval_selection(idx_lasso, F_train, y_train, F_eval, y_eval),
                         "wall_clock_s": time.perf_counter() - t0}
        print(f"  Lasso:   rmse={row['lasso']['rmse']:.4f}  ({row['lasso']['wall_clock_s']:.1f}s)")

        t0 = time.perf_counter()
        idx_greedy = greedy_forward(F_train, y_train, K)
        row["greedy"] = {"rmse": eval_selection(idx_greedy, F_train, y_train, F_eval, y_eval),
                          "wall_clock_s": time.perf_counter() - t0}
        print(f"  Greedy:  rmse={row['greedy']['rmse']:.4f}  ({row['greedy']['wall_clock_s']:.1f}s)")

        full_ridge_rmse = eval_selection(np.arange(F.shape[1]), F_train, y_train, F_eval, y_eval)
        row["full_no_selection"] = {"rmse": full_ridge_rmse}
        print(f"  Full (no selection, {F.shape[1]} feats): rmse={full_ridge_rmse:.4f}")

        results["by_K"][str(K)] = row

    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
