#!/usr/bin/env python3
"""Sprint 5 — IPC-Matched Reservoir Tuning (Novel Contribution).

Task-adapted tuning: measure the *demand profile* of the real weather
residual forecasting task (how much of the h-step-ahead target is
explained by low-order Legendre-polynomial functions of its own lagged
history, at each (delay, degree) cell — QRCx.metrics.task_demand), then
tune the sequential dissipative reservoir's (gamma1, input_scaling) so its
*supply* profile (the same decomposition applied to the reservoir's own
Information Processing Capacity under an iid Gaussian drive —
QRCx.metrics.reservoir_sequential.measure_ipc_by_degree) best matches it,
via the captured-capacity objective Σ min(C, D). No published work does
this for operational weather data (per the sprint's own framing).

Real compute budget (this is the classical part of Sprint 5; all CPU,
no GPU needed): reservoir driven at n_qubits=10 (this project's
documented CPU-feasible fallback, not the 12-qubit reference — the
reference is GPU-only-feasible, and this is a coarse tuning sweep, not
a final headline number) with real per-step cost measured at ~0.67-0.91
s/step (complex64/complex128) on this machine before committing to the
grid below.

Scope reductions from the literal spec (logged here, not silent):
- Coarse grid restricted to (gamma1, input_scaling) with tau fixed at the
  reference value (1.0) -- Sprint 2/2.5's own established tuning grid
  never swept tau either (see configs/v5_reference.yaml), so this matches
  precedent rather than inventing a new reduction.
- Supply-side IPC measured with n_steps=200 (not thousands) per grid
  point -- a coarse/exploratory sample size, sufficient for a comparative
  ranking across configs, not a final precision measurement.
- The final "matched vs Sprint 2 reference, on pilot forecast metrics"
  comparison uses a REDUCED pilot subsequence (1500 train + 500 test
  steps, still real KORD data from data/sprint4_pilot_seq.npz), not the
  full 3-year pilot -- a full-scale re-run at 10 qubits would take many
  hours per config; this is an exploratory/comparative check, consistent
  with Sprint 2.6's SEARCH_PROTOCOL/FULL_PROTOCOL two-tier convention.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

from QRCx.reservoir.sequential import SequentialDissipativeQRC
from QRCx.metrics.reservoir_sequential import drive_iid_gaussian, measure_ipc_by_degree
from QRCx.metrics.task_demand import compute_demand_profile
from sklearn.linear_model import Ridge

N_QUBITS = 10  # documented CPU-feasible fallback (see README/SPRINT_2_5_REPORT.md); 12q reference is GPU-only-feasible
DTYPE = "complex64"
DELAYS = list(range(0, 25))
DEGREES = (1, 2, 3)
GAMMA1_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 0.5]  # reusing Sprint 2.5's ipc_mc_characterization grid points
A_GRID = [0.1, 0.3, 1.0]
TAU = 1.0  # fixed at reference value -- not swept, matching Sprint 2/2.5 precedent (see module docstring)
SUPPLY_N_STEPS = 200
FORECAST_TRAIN_STEPS = 1500
FORECAST_TEST_STEPS = 500
REFERENCE_GAMMA1 = 0.03  # configs/v5_reference.yaml
REFERENCE_A = 0.3
HORIZONS_FOR_DEMAND = [1, 6]
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

OUT_PATH = REPO_ROOT / "results" / "ipc_matching.json"


def _r_squared_skill_rmse(y_true, y_pred, y_persist):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rmse_p = np.sqrt(np.mean((y_true - y_persist) ** 2))
    return float(1.0 - rmse / rmse_p) if rmse_p > 0 else float("nan")


def fit_eval_ridge(X_fit, y_fit, X_ev, y_ev):
    n_val = max(1, int(0.2 * len(X_fit)))
    Xf, Xv = X_fit[:-n_val], X_fit[-n_val:]
    yf, yv = y_fit[:-n_val], y_fit[-n_val:]
    best_alpha, best_score = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha).fit(Xf, yf)
        score = np.mean((yv - m.predict(Xv)) ** 2)
        if score < best_score:
            best_score, best_alpha = score, alpha
    model = Ridge(alpha=best_alpha).fit(X_fit, y_fit)
    return model.predict(X_ev), best_alpha


def evaluate_forecast_metrics(gamma1, a, train_seq_sub, target_col_idx, horizons=(1, 3, 6, 12)):
    """Drives a reduced real pilot subsequence at the given (gamma1, a)
    and reports skill vs. persistence at each horizon (residual target,
    same convention as Sprint 4). Uses N_QUBITS/TAU/DTYPE module constants."""
    n = train_seq_sub.shape[0]
    n_train = FORECAST_TRAIN_STEPS
    n_test = FORECAST_TEST_STEPS
    assert n >= n_train + n_test
    w_in = np.ones(N_QUBITS)  # neutral w_in (no NARMA-style symmetry-breaking needed with real multi-feature input)
    qrc = SequentialDissipativeQRC(
        n_qubits=N_QUBITS, tau=TAU, trotter_steps=10, gamma1=gamma1, gamma2=0.1,
        input_scaling=a, w_in=w_in, washout=0, seed=42, dtype=DTYPE,
    )
    t0 = time.perf_counter()
    feats = qrc.drive(train_seq_sub[:n_train + n_test])
    elapsed = time.perf_counter() - t0

    y_full = train_seq_sub[:n_train + n_test, target_col_idx]
    results = {}
    for h in horizons:
        X_fit = feats[:n_train - h]
        y_fit = y_full[h:n_train] - y_full[:n_train - h]  # residual target
        X_ev = feats[n_train:n_train + n_test - h]
        y_persist = y_full[n_train:n_train + n_test - h]
        y_true = y_full[n_train + h:n_train + n_test]
        pred_reg, alpha = fit_eval_ridge(X_fit, y_fit, X_ev, y_true - y_persist)
        pred_true = pred_reg + y_persist
        results[str(h)] = {
            "skill_vs_persistence": _r_squared_skill_rmse(y_true, pred_true, y_persist),
            "best_alpha": alpha,
        }
    return {"gamma1": gamma1, "a": a, "wall_clock_s": elapsed, "n_train": n_train, "n_test": n_test,
            "metrics": results}


def main():
    print(f"N_QUBITS={N_QUBITS} DTYPE={DTYPE} (CPU-feasible fallback; see module docstring)")

    npz_path = REPO_ROOT / "data" / "sprint4_pilot_seq.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found -- run scripts/sprint4_export_pilot_seq.py first.")
    npz = np.load(npz_path)
    train_seq = npz["train_seq"]
    target_col_idx = int(npz["target_col_idx"])
    target_series = train_seq[:, target_col_idx]

    print("\n=== Step 1: task demand profile (real KORD residual target) ===")
    demand = {}
    for h in HORIZONS_FOR_DEMAND:
        t0 = time.perf_counter()
        d = compute_demand_profile(target_series, horizon=h, delays=DELAYS, degrees=DEGREES)
        print(f"  h={h}: total_demand={d['total_demand']:.4f} ({time.perf_counter() - t0:.1f}s)")
        demand[str(h)] = d

    print(f"\n=== Step 2: supply profile grid ({len(GAMMA1_GRID)}x{len(A_GRID)}"
          f"={len(GAMMA1_GRID) * len(A_GRID)} configs, n_steps={SUPPLY_N_STEPS}) ===")
    supply_grid = []
    for gamma1 in GAMMA1_GRID:
        for a in A_GRID:
            t0 = time.perf_counter()
            qrc = SequentialDissipativeQRC(
                n_qubits=N_QUBITS, tau=TAU, trotter_steps=10, gamma1=gamma1, gamma2=0.1,
                input_scaling=a, washout=0, seed=42, dtype=DTYPE,
            )
            u, feats = drive_iid_gaussian(qrc, n_steps=SUPPLY_N_STEPS, seed=0)
            supply = measure_ipc_by_degree(u=u, features=feats, max_lag=24, degrees=DEGREES)
            elapsed = time.perf_counter() - t0
            entry = {"gamma1": gamma1, "a": a, "total_capacity": supply["total_capacity"],
                      "C": supply["C"].tolist(), "lags": supply["lags"], "wall_clock_s": elapsed}
            supply_grid.append(entry)
            print(f"  gamma1={gamma1:.3f} a={a:.2f}: total_capacity={supply['total_capacity']:.4f} "
                  f"({elapsed:.1f}s)")
            with open(OUT_PATH, "w") as f:
                json.dump({"demand": {h: {**demand[h], "D": demand[h]["D"].tolist()} for h in demand},
                            "supply_grid_partial": supply_grid}, f, indent=2)

    print("\n=== Step 3: matching (maximize sum over h in {1,6} of Sigma min(C, D)) ===")
    matching_scores = []
    for entry in supply_grid:
        C = np.array(entry["C"])
        score_total = 0.0
        per_h = {}
        for h in HORIZONS_FOR_DEMAND:
            D = demand[str(h)]["D"]
            assert C.shape == D.shape, f"shape mismatch C{C.shape} vs D{D.shape}"
            captured = float(np.minimum(C, D).sum())
            per_h[str(h)] = captured
            score_total += captured
        matching_scores.append({"gamma1": entry["gamma1"], "a": entry["a"],
                                  "captured_total": score_total, "captured_per_h": per_h})
        print(f"  gamma1={entry['gamma1']:.3f} a={entry['a']:.2f}: captured_total={score_total:.4f}")

    best = max(matching_scores, key=lambda m: m["captured_total"])
    print(f"\nBest matched config: gamma1={best['gamma1']}, a={best['a']} "
          f"(captured_total={best['captured_total']:.4f})")

    print(f"\n=== Step 4: matched config vs Sprint 2 reference (gamma1={REFERENCE_GAMMA1}, "
          f"a={REFERENCE_A}) on reduced real pilot forecast metrics ===")
    matched_result = evaluate_forecast_metrics(best["gamma1"], best["a"], train_seq, target_col_idx)
    print(f"  matched: {json.dumps(matched_result['metrics'])} ({matched_result['wall_clock_s']:.1f}s)")
    reference_result = evaluate_forecast_metrics(REFERENCE_GAMMA1, REFERENCE_A, train_seq, target_col_idx)
    print(f"  reference: {json.dumps(reference_result['metrics'])} ({reference_result['wall_clock_s']:.1f}s)")

    result = {
        "n_qubits": N_QUBITS, "dtype": DTYPE, "tau": TAU,
        "gamma1_grid": GAMMA1_GRID, "a_grid": A_GRID,
        "demand": {h: {**demand[h], "D": demand[h]["D"].tolist()} for h in demand},
        "supply_grid": supply_grid,
        "matching_scores": matching_scores,
        "best_matched_config": best,
        "matched_forecast_result": matched_result,
        "reference_forecast_result": reference_result,
        "reference_config": {"gamma1": REFERENCE_GAMMA1, "a": REFERENCE_A},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    main()
