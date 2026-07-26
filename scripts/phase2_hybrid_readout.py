#!/usr/bin/env python3
"""FINAL SPRINT Phase 2 -- THE key experiment: does the quantum reservoir
add forecast-relevant information beyond what's linearly present in the
raw observation history?

Four design matrices, aligned to the SAME underlying continuous sequence
the v5 features were driven on (see scripts/phase2_drive_pilot_features.py):
  A  = flattened raw 24h window (312 features)      [current champion]
  B  = v5 reservoir features (already driven)
  C  = [A, B] concatenated                          [quantum increment]
  C' = [A, ESN features] concatenated, ESN sized to match B's dimension
       [classical control -- mandatory]

Ridge readout on each (alpha selected on a held-out validation TAIL of
the driven sequence, never on the eval segment), then:
  - DM test of C vs A (squared errors) at h in {1,3,6,12,24}, HAC-corrected,
    plus moving-block bootstrap CI on the skill difference.
  - Same for C' vs A (fairness control).
  - B alone, reported too.

Interpretation rule fires automatically based on the DM p-values and
which of C/C' actually beats A (see INTERPRETATION_RULES below) --
written into the output JSON, not decided by eye.

Diagnostics:
  - Effective rank (participation ratio) of B vs gamma1 (tuned vs off).
  - Feature variance of B vs gamma1 (mechanism check for Table 3's collapse).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.metrics.significance import diebold_mariano, skill_difference_ci

FEAT_DIR = REPO_ROOT / "results" / "phase2_features"
OUT_PATH = REPO_ROOT / "results" / "hybrid_readout.json"
FIG_DIR = REPO_ROOT / "figures"
HORIZONS = [1, 3, 6, 12, 24]
W = 24  # window length for A
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
VAL_FRAC = 0.2  # tail fraction of the fit segment used for alpha selection


def build_A(full_seq, start, end):
    """Flattened raw W-hour window ending at each t in [start, end)."""
    return np.stack([full_seq[t - W:t].reshape(-1) for t in range(start, end)])


def standardize_fit(X_fit):
    mean = X_fit.mean(axis=0, keepdims=True)
    std = X_fit.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std


def fit_eval_ridge(X_fit, y_fit, X_ev, y_ev):
    n_val = max(1, int(VAL_FRAC * len(X_fit)))
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


def participation_ratio(X):
    """Effective rank via participation ratio: (sum(s^2))^2 / sum(s^4),
    s = singular values of the (mean-centered) feature matrix."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def build_esn_control(full_seq, dim):
    from reservoirpy.nodes import Reservoir
    reservoir = Reservoir(units=dim, sr=0.9, lr=0.3, input_scaling=0.5, seed=7, rc_connectivity=0.1)
    states = reservoir.run(full_seq)
    return states


def main():
    meta = json.load(open(FEAT_DIR / "meta.json"))
    target_series = np.load(FEAT_DIR / "target_series.npy")
    n_train, n_test = meta["n_train"], meta["n_test"]
    full_seq = np.load(REPO_ROOT / "data" / "sprint4_pilot_seq.npz")
    # Reconstruct the exact full_seq the features were driven on
    train_seq_full = full_seq["train_seq"]
    val_seq_full = full_seq["val_seq"]
    driven_seq = np.concatenate([train_seq_full[-n_train:], val_seq_full[:n_test]], axis=0)
    target_col_idx = meta["target_col_idx"]

    B_tuned = np.load(FEAT_DIR / "v5_features_tuned.npy").astype(np.float64)
    B_off = np.load(FEAT_DIR / "v5_features_diagnostic_off.npy").astype(np.float64)
    n_total = len(driven_seq)
    assert B_tuned.shape[0] == n_total

    print(f"Loaded: {n_total} total steps, B dim={B_tuned.shape[1]}")

    print("Building ESN control (dimension-matched to B)...")
    esn_states = build_esn_control(driven_seq, dim=B_tuned.shape[1])

    max_h = max(HORIZONS)
    start = W
    end = n_total - max_h
    A_full = build_A(driven_seq, start, end)
    B_full = B_tuned[start:end]
    Bctrl_full = esn_states[start:end]
    y_full = driven_seq[start:end, target_col_idx]
    y_persist_full = driven_seq[start - 1:end - 1, target_col_idx]  # not used per-h, computed per-h below

    n_fit = int(0.75 * len(A_full))  # 75/25 fit/eval split of the driven-sequence sample

    results = {"horizons": HORIZONS, "n_fit": n_fit, "n_eval": len(A_full) - n_fit, "per_horizon": {}}

    for h in HORIZONS:
        idx_end = end - h  # need y[t+h] to exist
        n_h = idx_end - start
        if n_h <= n_fit:
            continue
        y_target = driven_seq[start + h:idx_end + h, target_col_idx]
        y_pers = driven_seq[start:idx_end, target_col_idx]
        A = build_A(driven_seq, start, idx_end)
        B = B_tuned[start:idx_end]
        Bc = esn_states[start:idx_end]

        def split(X):
            return X[:n_fit], X[n_fit:]

        A_fit, A_ev = split(A)
        B_fit, B_ev = split(B)
        Bc_fit, Bc_ev = split(Bc)
        y_fit, y_ev = y_target[:n_fit], y_target[n_fit:]
        pers_fit, pers_ev = y_pers[:n_fit], y_pers[n_fit:]

        # standardize per block using FIT statistics only
        def std_apply(fit, ev):
            mean, std = standardize_fit(fit)
            return (fit - mean) / std, (ev - mean) / std

        A_fit_s, A_ev_s = std_apply(A_fit, A_ev)
        B_fit_s, B_ev_s = std_apply(B_fit, B_ev)
        Bc_fit_s, Bc_ev_s = std_apply(Bc_fit, Bc_ev)

        C_fit_s = np.concatenate([A_fit_s, B_fit_s], axis=1)
        C_ev_s = np.concatenate([A_ev_s, B_ev_s], axis=1)
        Cc_fit_s = np.concatenate([A_fit_s, Bc_fit_s], axis=1)
        Cc_ev_s = np.concatenate([A_ev_s, Bc_ev_s], axis=1)

        row = {}
        for name, X_fit, X_ev in [("A", A_fit_s, A_ev_s), ("B", B_fit_s, B_ev_s),
                                    ("C", C_fit_s, C_ev_s), ("Cprime", Cc_fit_s, Cc_ev_s)]:
            pred, alpha = fit_eval_ridge(X_fit, y_fit - pers_fit, X_ev, y_ev - pers_ev)
            pred_true = pred + pers_ev
            row[name] = {"skill": skill(y_ev, pred_true, pers_ev), "rmse": rmse(y_ev, pred_true),
                         "alpha": alpha, "_pred": pred_true}

        dm_C_vs_A = diebold_mariano(y_ev, row["C"]["_pred"], row["A"]["_pred"], h=h)
        ci_C_vs_A = skill_difference_ci(y_ev, row["C"]["_pred"], row["A"]["_pred"], pers_ev, n_boot=1000, seed=42)
        dm_Cp_vs_A = diebold_mariano(y_ev, row["Cprime"]["_pred"], row["A"]["_pred"], h=h)
        ci_Cp_vs_A = skill_difference_ci(y_ev, row["Cprime"]["_pred"], row["A"]["_pred"], pers_ev, n_boot=1000, seed=42)

        for name in row:
            del row[name]["_pred"]

        results["per_horizon"][str(h)] = {
            "A": row["A"], "B": row["B"], "C": row["C"], "Cprime": row["Cprime"],
            "dm_C_vs_A": dm_C_vs_A, "dm_Cprime_vs_A": dm_Cp_vs_A,
            "skill_diff_ci_C_vs_A": {k: v for k, v in ci_C_vs_A.items() if k != "replicates"},
            "skill_diff_ci_Cprime_vs_A": {k: v for k, v in ci_Cp_vs_A.items() if k != "replicates"},
        }
        print(f"  h={h}: A={row['A']['skill']*100:.2f}% B={row['B']['skill']*100:.2f}% "
              f"C={row['C']['skill']*100:.2f}% (p={dm_C_vs_A['p_value']:.4f}) "
              f"C'={row['Cprime']['skill']*100:.2f}% (p={dm_Cp_vs_A['p_value']:.4f})")

    # Interpretation rule
    sig_C = [h for h, r in results["per_horizon"].items() if r["dm_C_vs_A"]["p_value"] < 0.05
             and r["C"]["skill"] > r["A"]["skill"]]
    sig_Cp = [h for h, r in results["per_horizon"].items() if r["dm_Cprime_vs_A"]["p_value"] < 0.05
              and r["Cprime"]["skill"] > r["A"]["skill"]]
    c_gain = np.mean([results["per_horizon"][h]["C"]["skill"] - results["per_horizon"][h]["A"]["skill"]
                       for h in results["per_horizon"]])
    cp_gain = np.mean([results["per_horizon"][h]["Cprime"]["skill"] - results["per_horizon"][h]["A"]["skill"]
                        for h in results["per_horizon"]])

    if sig_C and c_gain > cp_gain:
        rule = "quantum_adds_information"
        conclusion = ("The quantum reservoir contributes forecast-relevant information not "
                       "linearly present in the observation history, beyond what a "
                       "dimension-matched classical reservoir contributes.")
    elif sig_C:
        rule = "reservoir_adds_information_no_quantum_advantage"
        conclusion = ("Reservoir features (quantum or classical) add real information over the "
                       "raw window, but the quantum reservoir offers no advantage over a "
                       "dimension-matched classical reservoir on this task.")
    else:
        rule = "redundant"
        conclusion = ("The reservoir's signal is redundant with the raw window's linear history "
                       "-- concatenating it does not significantly improve on A at any tested horizon.")

    results["interpretation_rule"] = rule
    results["conclusion"] = conclusion
    results["significant_horizons_C_vs_A"] = sig_C
    results["significant_horizons_Cprime_vs_A"] = sig_Cp
    results["mean_C_minus_A_skill"] = float(c_gain)
    results["mean_Cprime_minus_A_skill"] = float(cp_gain)
    print(f"\nINTERPRETATION RULE: {rule}\n{conclusion}")

    # Diagnostics
    print("\nDiagnostics: effective rank + variance of B, tuned vs diagnostic_off")
    pr_tuned = participation_ratio(B_tuned[start:end])
    pr_off = participation_ratio(B_off[start:end])
    var_tuned = float(np.mean(np.var(B_tuned[start:end], axis=0)))
    var_off = float(np.mean(np.var(B_off[start:end], axis=0)))
    results["diagnostics"] = {
        "effective_rank_tuned": pr_tuned, "effective_rank_diagnostic_off": pr_off,
        "n_features_B": int(B_tuned.shape[1]),
        "mean_feature_variance_tuned": var_tuned, "mean_feature_variance_diagnostic_off": var_off,
    }
    print(f"  effective rank: tuned={pr_tuned:.2f}  off={pr_off:.2f}  (of {B_tuned.shape[1]} raw features)")
    print(f"  mean feature variance: tuned={var_tuned:.4f}  off={var_off:.6f}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
