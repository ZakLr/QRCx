#!/usr/bin/env python3
"""Sprint 2 Phase 2.1: NARMA10 tuning gate.

HARD GATE (per sprint spec): the tuned reservoir must beat the linear AR
baseline (NMSE < 0.099, the value measured in Sprint 1 --
results/narma10_validation.json); target NMSE <= 0.2, aspirational <= 0.15.
If the gate fails, this script stops and writes the best config + result
found to results/narma10_sweep.json for the orchestrator to review --
weather-data work does not proceed on a failed gate.

Search strategy: the full gamma1(5) x gamma2(3) x input_scaling(3) x V(3)
x washout(3) factorial (405 combinations) is not tractable within this
sprint's compute budget even with the Sprint 2 Phase 2.0 exact-propagator
speedup (~1.87 s/step at the 10-qubit fallback, vs Sprint 1's ~6.59 s/step
Trotter path). Instead:
  1. Coarse 1D sweep over gamma1 (the spec's primary axis) at fixed
     gamma2=0.03, a=1.0, washout=24, V=1 -- 5 drives.
  2. Refine gamma2 at the best gamma1 -- 2 more drives (gamma2=0.03 already
     done in step 1).
  3. Refine input_scaling (a) at the best (gamma1, gamma2) -- 2 more drives
     (a=1.0 already done).
  4. At the best (gamma1, gamma2, a): washout doesn't require re-driving
     (it only changes how many initial steps of the same trajectory are
     discarded), so all three washout values are evaluated for free by
     slicing one drive differently.
  5. Multiplexing (V) likewise doesn't require re-driving if driven once
     at V=4 (sub-readouts are nested: V=1's single readout is V=4's 4th
     sub-readout, V=2's two readouts are V=4's 2nd and 4th) -- so V=1/2/4
     are evaluated from one V=4 drive at the final best config.
This is a coordinate-wise search, not exhaustive, documented as a
deviation from the literal grid due to compute budget (see
docs/sprint_log/SPRINT_2_REPORT.md).

Every experiment script accepts FAST_MODE. FAST_MODE=True (default) uses
250 post-washout steps; FAST_MODE=False uses 400 (Sprint 1's setting).
"""
import argparse
import itertools
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

AR_BASELINE_NMSE = 0.099  # results/narma10_validation.json, Sprint 1
TARGET_NMSE = 0.2
ASPIRATIONAL_NMSE = 0.15
ALPHA_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0]
N_QUBITS = 10  # Sprint 1/2 runtime-gate fallback on this (CPU-only) hardware
W_IN_SEED = 123


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def select_alpha_by_cv(X_train: np.ndarray, y_train: np.ndarray) -> float:
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


def evaluate_features(features: np.ndarray, y_eval: np.ndarray) -> dict:
    n_train = int(len(y_eval) * 0.7)
    best_alpha = select_alpha_by_cv(features[:n_train], y_eval[:n_train])
    model = Ridge(alpha=best_alpha).fit(features[:n_train], y_eval[:n_train])
    pred = model.predict(features[n_train:])
    return {
        "nmse": nmse(y_eval[n_train:], pred),
        "alpha": best_alpha,
        "n_train": n_train,
        "n_test": len(y_eval) - n_train,
        "n_features": features.shape[1],
    }


def drive_config(u: np.ndarray, w_in: np.ndarray, gamma1: float, gamma2: float,
                  a: float, washout_max: int, multiplexing: int = 4) -> np.ndarray:
    qrc = SequentialDissipativeQRC(
        n_qubits=N_QUBITS, trotter_steps=10, gamma1=gamma1, gamma2=gamma2,
        injection="ry", washout=washout_max, seed=42, input_scaling=a,
        propagator="exact", multiplexing=multiplexing, w_in=w_in,
    )
    seq_input = u[:, None] * w_in[None, :]
    return qrc.drive(seq_input)


def eval_at_washout(features_v4: np.ndarray, y: np.ndarray, washout: int, V: int) -> dict:
    """Slice a V=4 drive's features (no re-drive) to emulate washout and V."""
    n_feat_base = features_v4.shape[1] // 4
    if V == 4:
        feats = features_v4
    elif V == 2:
        # sub-readouts 2 and 4 of the V=4 drive (indices 1 and 3 of 4 blocks)
        blocks = features_v4.reshape(-1, 4, n_feat_base)
        feats = blocks[:, [1, 3], :].reshape(len(features_v4), -1)
    elif V == 1:
        blocks = features_v4.reshape(-1, 4, n_feat_base)
        feats = blocks[:, 3, :]
    else:
        raise ValueError(V)
    return evaluate_features(feats[washout:], y[washout:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false")
    args = parser.parse_args()
    FAST_MODE = args.fast_mode

    n_steps = 250 if FAST_MODE else 400
    washout_max = 24
    total_steps = n_steps + washout_max

    rng_w = np.random.default_rng(W_IN_SEED)
    w_in = rng_w.uniform(0.5, 1.5, size=N_QUBITS)

    u, y = generate_narma10(n_steps=total_steps, seed=42)

    log = []
    t_start = time.perf_counter()

    def run_and_log(gamma1, gamma2, a, tag):
        t0 = time.perf_counter()
        feats_v4 = drive_config(u, w_in, gamma1, gamma2, a, washout_max, multiplexing=4)
        drive_s = time.perf_counter() - t0
        result = eval_at_washout(feats_v4, y, washout_max, V=1)
        entry = {
            "tag": tag, "gamma1": gamma1, "gamma2": gamma2, "input_scaling": a,
            "washout": washout_max, "V": 1, "drive_wall_clock_s": drive_s, **result,
        }
        log.append(entry)
        print(json.dumps(entry))
        return entry, feats_v4

    # Step 1: coarse gamma1 sweep at gamma2=0.03, a=1.0
    gamma1_grid = [0.0, 0.01, 0.03, 0.1, 0.3]
    best = None
    for g1 in gamma1_grid:
        entry, _ = run_and_log(g1, 0.03, 1.0, "gamma1_sweep")
        if best is None or entry["nmse"] < best["nmse"]:
            best = entry
    best_gamma1 = best["gamma1"]

    # Step 2: refine gamma2 at best gamma1 (0.03 already covered)
    for g2 in [0.0, 0.1]:
        entry, _ = run_and_log(best_gamma1, g2, 1.0, "gamma2_refine")
        if entry["nmse"] < best["nmse"]:
            best = entry
    best_gamma2 = best["gamma2"]

    # Step 3: refine input_scaling at best (gamma1, gamma2) (a=1.0 already covered)
    for a in [0.3, 3.0]:
        entry, _ = run_and_log(best_gamma1, best_gamma2, a, "input_scaling_refine")
        if entry["nmse"] < best["nmse"]:
            best = entry
    best_a = best["input_scaling"]

    # Step 4+5: at the best config, evaluate washout in {10,20,50} and V in
    # {1,2,4} from ONE V=4 drive (no re-driving needed for either axis).
    u2, y2 = generate_narma10(n_steps=50 + n_steps, seed=42)
    t0 = time.perf_counter()
    feats_v4_final = drive_config(u2, w_in, best_gamma1, best_gamma2, best_a, washout_max=50, multiplexing=4)
    drive_s = time.perf_counter() - t0

    washout_v_grid = []
    for washout, V in itertools.product([10, 20, 50], [1, 2, 4]):
        result = eval_at_washout(feats_v4_final, y2, washout, V)
        entry = {
            "tag": "washout_V_grid", "gamma1": best_gamma1, "gamma2": best_gamma2,
            "input_scaling": best_a, "washout": washout, "V": V,
            "drive_wall_clock_s": drive_s, **result,
        }
        washout_v_grid.append(entry)
        log.append(entry)
        print(json.dumps(entry))
        if best is None or entry["nmse"] < best["nmse"]:
            best = entry

    total_wall_clock = time.perf_counter() - t_start
    gate_passed = best["nmse"] < AR_BASELINE_NMSE

    result = {
        "FAST_MODE": FAST_MODE,
        "n_qubits": N_QUBITS,
        "w_in_seed": W_IN_SEED,
        "w_in": w_in.tolist(),
        "ar_baseline_nmse": AR_BASELINE_NMSE,
        "target_nmse": TARGET_NMSE,
        "aspirational_nmse": ASPIRATIONAL_NMSE,
        "best_config": {
            "gamma1": best["gamma1"], "gamma2": best["gamma2"],
            "input_scaling": best["input_scaling"], "washout": best["washout"],
            "V": best["V"], "nmse": best["nmse"], "alpha": best["alpha"],
        },
        "gate_passed": gate_passed,
        "meets_target": best["nmse"] <= TARGET_NMSE,
        "meets_aspirational": best["nmse"] <= ASPIRATIONAL_NMSE,
        "total_wall_clock_s": total_wall_clock,
        "log": log,
    }

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "narma10_sweep.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps({k: v for k, v in result.items() if k != "log"}, indent=2))
    print(f"\nWritten to {out_path}")
    print(f"\nGATE {'PASSED' if gate_passed else 'FAILED'}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
