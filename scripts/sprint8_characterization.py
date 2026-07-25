#!/usr/bin/env python3
"""Sprint 8 — Scaling, Shots & Noise Characterization: qubit-count
scaling (MC/IPC/forecast skill vs N), finite-shot noise degradation, and
encoding-density sweep for the sequential dissipative reservoir.

All at the reference config (gamma1=0.03, input_scaling=0.3, tau=1.0,
injection=ry) except where a parameter is itself the thing being swept.
N=12 stays the reference/anchor point (per the sprint spec); N<12 are
characterization-only, run at genuinely smaller cost.

Scope notes (logged, not silent):
- N=12's MC/IPC sample size is deliberately smaller (150 iid-Gaussian
  steps) than N<12's (300 steps) -- real per-step cost at 12 qubits
  (measured ~10-18 s/step CPU) makes an equal-size sweep impractical;
  documented, not silently reduced.
- Forecast-skill-vs-N uses a small, IDENTICAL-size real pilot subsequence
  across all N (not Sprint 4/5's larger, N-specific samples) so the
  scaling comparison is apples-to-apples.
- Shot-noise reuses the EXACT (infinite-shot) correlator features already
  driven for the N=10 forecast-skill point -- finite-shot estimates are
  simulated by binomial-sampling each Pauli expectation post-hoc (a valid
  simulation of measurement shot noise: for a Pauli P with true
  expectation c in [-1,1], an S-shot estimate is 2*Binomial(S,(1+c)/2)/S-1),
  NOT by re-driving the reservoir once per shot budget -- avoids S x N
  redundant re-drives.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

from QRCx.reservoir.sequential import SequentialDissipativeQRC
from QRCx.metrics.reservoir_sequential import (
    drive_iid_gaussian, measure_memory_capacity_sequential, measure_ipc_sequential,
)
from QRCx.metrics.significance import skill_difference_ci
from sklearn.linear_model import Ridge

REFERENCE = dict(gamma1=0.03, gamma2=0.1, input_scaling=0.3, tau=1.0, injection="ry")
N_VALUES = [4, 6, 8, 10, 12]
MC_IPC_N_STEPS = {4: 300, 6: 300, 8: 300, 10: 300, 12: 150}  # N=12 reduced -- see module docstring
FORECAST_TRAIN = 200  # kept small and uniform across all N (see docstring) -- real N=12 cost (~10s/step CPU) would make a larger sample take hours
FORECAST_TEST = 80
SHOT_BUDGETS = [1_000, 10_000, 100_000, None]  # None = infinite/exact
SHOT_STUDY_N = 10  # reference-scale but CPU-fast
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

OUT_PATH = REPO_ROOT / "results" / "characterization.json"


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


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def skill(y, p, y_persist):
    rp, rpe = rmse(y, p), rmse(y, y_persist)
    return float(1.0 - rp / rpe) if rpe > 0 else float("nan")


def forecast_skill(features, target_series, n_train, n_test, horizon):
    y_full = target_series
    X_fit = features[:n_train - horizon]
    y_fit = y_full[horizon:n_train] - y_full[:n_train - horizon]
    X_ev = features[n_train:n_train + n_test - horizon]
    y_persist = y_full[n_train:n_train + n_test - horizon]
    y_true = y_full[n_train + horizon:n_train + n_test]
    pred_reg, alpha = fit_eval_ridge(X_fit, y_fit, X_ev, y_true - y_persist)
    pred_true = pred_reg + y_persist
    return skill(y_true, pred_true, y_persist), alpha


def apply_shot_noise(features_exact, n_shots, rng):
    """features_exact: (T, n_feat) array of exact Pauli expectations in
    [-1, 1]. Returns a shot-noisy copy via per-entry binomial sampling."""
    if n_shots is None:
        return features_exact.copy()
    c = np.clip(features_exact, -1.0, 1.0)
    p = (1.0 + c) / 2.0
    k = rng.binomial(n_shots, p)
    return 2.0 * k / n_shots - 1.0


def main():
    npz_path = REPO_ROOT / "data" / "sprint4_pilot_seq.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found -- run scripts/sprint4_export_pilot_seq.py first.")
    npz = np.load(npz_path)
    train_seq = npz["train_seq"]
    target_col_idx = int(npz["target_col_idx"])
    target_series = train_seq[:FORECAST_TRAIN + FORECAST_TEST + 1, target_col_idx]

    print(f"=== Step 1: qubit scaling N={N_VALUES} (MC, IPC, forecast skill) ===")
    scaling = []
    shot_study_features = None
    for n in N_VALUES:
        t0 = time.perf_counter()
        qrc_mc = SequentialDissipativeQRC(n_qubits=n, washout=0, seed=42, dtype="complex64", **REFERENCE)
        n_steps = MC_IPC_N_STEPS[n]
        u, feats_iid = drive_iid_gaussian(qrc_mc, n_steps=n_steps, seed=0)
        mc = measure_memory_capacity_sequential(u=u, features=feats_iid, max_lag=min(20, n_steps - 1))
        ipc = measure_ipc_sequential(u=u, features=feats_iid, max_lag=min(10, n_steps - 1))
        t_mcipc = time.perf_counter() - t0

        t0 = time.perf_counter()
        qrc_fc = SequentialDissipativeQRC(n_qubits=n, washout=0, seed=42, dtype="complex64", **REFERENCE)
        feats_fc = qrc_fc.drive(train_seq[:FORECAST_TRAIN + FORECAST_TEST])
        t_drive = time.perf_counter() - t0
        skill_h1, alpha1 = forecast_skill(feats_fc, target_series, FORECAST_TRAIN, FORECAST_TEST, 1)
        skill_h6, alpha6 = forecast_skill(feats_fc, target_series, FORECAST_TRAIN, FORECAST_TEST, 6)

        entry = {
            "n_qubits": n, "mc": mc["MC"], "ipc_total": ipc["total_ipc"],
            "ipc_linear": ipc["linear_ipc"], "ipc_nonlinear": ipc["nonlinear_ipc"],
            "skill_h1": skill_h1, "skill_h6": skill_h6,
            "mc_ipc_n_steps": n_steps, "wall_clock_mc_ipc_s": t_mcipc, "wall_clock_drive_s": t_drive,
        }
        scaling.append(entry)
        print(f"  N={n}: MC={mc['MC']:.3f} IPC={ipc['total_ipc']:.3f} "
              f"skill@1h={skill_h1*100:.2f}% skill@6h={skill_h6*100:.2f}% "
              f"({t_mcipc+t_drive:.1f}s)")
        if n == SHOT_STUDY_N:
            shot_study_features = feats_fc.copy()

    print(f"\n=== Step 2: finite-shot noise study (N={SHOT_STUDY_N}, reusing its exact drive) ===")
    rng = np.random.default_rng(7)
    shot_results = []
    exact_skill, _ = forecast_skill(shot_study_features, target_series, FORECAST_TRAIN, FORECAST_TEST, 6)
    for s in SHOT_BUDGETS:
        feats_noisy = apply_shot_noise(shot_study_features, s, rng)
        sk, alpha = forecast_skill(feats_noisy, target_series, FORECAST_TRAIN, FORECAST_TEST, 6)
        shot_results.append({"n_shots": s, "skill_h6": sk, "degradation_vs_exact": exact_skill - sk})
        print(f"  S={s if s else 'inf'}: skill@6h={sk*100:.2f}% (exact={exact_skill*100:.2f}%)")

    # Bootstrap CI on the exact-shot skill, to define "degradation exceeds CI" threshold
    y_full = target_series
    n_train, n_test, h = FORECAST_TRAIN, FORECAST_TEST, 6
    X_ev = shot_study_features[n_train:n_train + n_test - h]
    y_persist = y_full[n_train:n_train + n_test - h]
    y_true = y_full[n_train + h:n_train + n_test]
    X_fit = shot_study_features[:n_train - h]
    y_fit = y_full[h:n_train] - y_full[:n_train - h]
    pred_reg, _ = fit_eval_ridge(X_fit, y_fit, X_ev, y_true - y_persist)
    pred_exact = pred_reg + y_persist
    ci = skill_difference_ci(y_true, pred_exact, y_persist, y_persist, n_boot=1000, seed=42)
    ci_halfwidth = (ci["upper"] - ci["lower"]) / 2.0
    threshold_shots = None
    for entry in shot_results:
        if entry["n_shots"] is not None and entry["degradation_vs_exact"] > ci_halfwidth:
            threshold_shots = entry["n_shots"]
            break
    if threshold_shots is not None:
        print(f"  bootstrap CI half-width on skill@6h: {ci_halfwidth*100:.3f}%  "
              f"-> degradation first exceeds CI at S={threshold_shots}")
    else:
        print(f"  bootstrap CI half-width on skill@6h: {ci_halfwidth*100:.3f}%  "
              f"-> no tested finite shot budget's degradation exceeded the CI "
              f"(even the smallest, S={min(s for s in SHOT_BUDGETS if s is not None)}, "
              f"is statistically indistinguishable from exact)")

    print(f"\n=== Step 3: encoding density (injection x multiplexing x input_scaling) ===")
    encoding_results = []
    for injection in ("ry", "zz"):
        for V in (1, 3):
            for a in (0.1, 0.3, 1.0):
                cfg = dict(REFERENCE)
                cfg["injection"] = injection
                cfg["input_scaling"] = a
                t0 = time.perf_counter()
                qrc = SequentialDissipativeQRC(n_qubits=8, washout=0, seed=42, dtype="complex64",
                                                multiplexing=V, **cfg)
                feats = qrc.drive(train_seq[:FORECAST_TRAIN + FORECAST_TEST])
                sk, alpha = forecast_skill(feats, target_series, FORECAST_TRAIN, FORECAST_TEST, 6)
                elapsed = time.perf_counter() - t0
                encoding_results.append({"injection": injection, "multiplexing_V": V, "input_scaling": a,
                                          "skill_h6": sk, "wall_clock_s": elapsed})
                print(f"  injection={injection} V={V} a={a}: skill@6h={sk*100:.2f}% ({elapsed:.1f}s)")

    result = {
        "reference_config": REFERENCE,
        "n_values": N_VALUES,
        "scaling": scaling,
        "shot_study": {"n_qubits": SHOT_STUDY_N, "shot_budgets": SHOT_BUDGETS, "results": shot_results,
                        "bootstrap_ci_halfwidth_h6": ci_halfwidth, "threshold_shots_exceeding_ci": threshold_shots},
        "encoding_density": encoding_results,
        "forecast_train": FORECAST_TRAIN, "forecast_test": FORECAST_TEST,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    main()
