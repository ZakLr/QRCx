#!/usr/bin/env python3
"""FINAL SPRINT Phase 5 (v2): the final matched benchmark on the CANONICAL
split (train 2019-2022 / val 2023 / test 2024) -- every model scored on
IDENTICAL rows, in ONE consolidated output file. Combines:
  - classical baselines + GBM ceiling probe (results/baselines_{split}.json,
    already run by scripts/generate_baselines.py on this same canonical split)
  - v5 QRC (12 qubits, density matrix, sequential/recurrent): residual
    ridge on the driven reservoir features from results/phase1_v5_canonical/,
    aligned to each windowed sample via canonical_seq.npz's *_valid_idx
    (window start position) + a split offset (v5 was driven on
    train_seq+val_seq+test_seq concatenated, in that order)
  - v4 QRC (20 qubits, statevector, windowed/non-recurrent): residual
    ridge on results/phase5_v4_features_{split}.npy (1:1 aligned to
    X_train/X_val/X_test, no windowing indirection needed)
  - Phase 2's concatenated-readout models C=[A,B] and C'=[A,B_esn],
    re-run here on the full canonical-split features

Every QRC/concat row reports: RMSE (scaled AND degC), MAE (degC), skill,
VPT, DM p vs persistence, DM p vs the strongest classical baseline
(null-control Ridge -- tied with Residual-Ridge as the strongest
classical model on this data, and already computed in this same script),
bootstrap CI vs persistence.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.metrics.forecast import rmse, mae, skill_score, vpt
from QRCx.metrics.significance import diebold_mariano, skill_difference_ci
from QRCx.data.split_guard import assert_test_unlocked, UNLOCK_TOKEN

EVAL_HORIZONS = [1, 6]
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

DATA_NPZ = REPO_ROOT / "data" / "canonical_seq.npz"
V5_DIR = REPO_ROOT / "results" / "phase1_v5_canonical"
V4_DIR = REPO_ROOT
UNITS_PATH = REPO_ROOT / "results" / "canonical_units.json"

# Classical rows pulled in verbatim from results/baselines_{split}.json --
# generated separately by scripts/generate_baselines.py on the identical
# canonical split, so no need to recompute here.
CLASSICAL_ROWS = [
    "persistence", "arima", "arima_auto", "esn_dim_matched", "esn_500",
    "residual_esn", "null_ridge", "null_krr", "residual_ridge",
]


def fit_eval_ridge(F_train, y_train_res, F_val, y_val_res):
    """Grid-search alpha via a held-out 20% tail of train, then refit on
    all of train -- same protocol as scripts/phase2_hybrid_readout.py."""
    n_tune = int(0.2 * len(F_train))
    F_fit, F_tune = F_train[:-n_tune], F_train[-n_tune:]
    y_fit, y_tune = y_train_res[:-n_tune], y_train_res[-n_tune:]
    best_alpha, best_mse = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha).fit(F_fit, y_fit)
        mse = float(np.mean((y_tune - m.predict(F_tune)) ** 2))
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    final = Ridge(alpha=best_alpha).fit(F_train, y_train_res)
    return final.predict(F_val), best_alpha


def evaluate(name, y_true, y_pred, y_persist, h, y_strongest_classical=None):
    row = {
        "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
        "skill": skill_score(y_true, y_pred, y_persist), "vpt": vpt(y_true, y_pred),
    }
    if name != "persistence":
        row["dm_vs_persistence"] = diebold_mariano(y_true, y_pred, y_persist, h=h)
        ci = skill_difference_ci(y_true, y_pred, y_persist, y_persist, n_boot=1000, seed=42)
        row["skill_diff_ci_vs_persistence"] = {k: v for k, v in ci.items() if k != "replicates"}
    if y_strongest_classical is not None:
        row["dm_vs_strongest_classical"] = diebold_mariano(y_true, y_pred, y_strongest_classical, h=h)
        ci2 = skill_difference_ci(y_true, y_pred, y_strongest_classical, y_persist, n_boot=1000, seed=42)
        row["skill_diff_ci_vs_strongest_classical"] = {k: v for k, v in ci2.items() if k != "replicates"}
    return row


def build_v5_aligned(v5_features, valid_idx, split_offset, W=24):
    """v5_features: (T_total, 234) per-timestep recurrent reservoir state,
    driven over train_seq+val_seq+test_seq concatenated. For each windowed
    sample with start position `idx` (into its own split's *_seq), the
    aligned feature is the reservoir state right after processing the
    window's LAST input (position idx+W-1 within that split, offset by
    split_offset into the global concatenated sequence)."""
    global_pos = split_offset + valid_idx + (W - 1)
    return v5_features[global_pos]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    eval_split = args.split
    if eval_split == "test":
        assert_test_unlocked(unlocked_by=UNLOCK_TOKEN,
                              reason="scripts/phase5_final_benchmark.py --split test "
                                     "(final one-time canonical-split confirmatory report)")

    npz = np.load(DATA_NPZ)
    horizons_available = [int(h) for h in npz["horizons"]]
    target_col_idx = int(npz["target_col_idx"])

    n_train_seq, n_val_seq = len(npz["train_seq"]), len(npz["val_seq"])
    split_offsets = {"train": 0, "val": n_train_seq, "test": n_train_seq + n_val_seq}

    X_train, y_train_all = npz["X_train"], npz["y_train"]
    X_eval, y_eval_all = npz[f"X_{eval_split}"], npz[f"y_{eval_split}"]
    train_valid_idx, eval_valid_idx = npz["train_valid_idx"], npz[f"{eval_split}_valid_idx"]

    y_train_persist = X_train[:, -1, target_col_idx]
    y_eval_persist = X_eval[:, -1, target_col_idx]

    units = json.load(open(UNITS_PATH))
    std_degc = units["std_scaled_to_anomaly"]

    # --- v5 (12q, density matrix, recurrent) ---
    v5_tuned = np.load(V5_DIR / "phase1_features_tuned.npy")
    F5_train = build_v5_aligned(v5_tuned, train_valid_idx, split_offsets["train"])
    F5_eval = build_v5_aligned(v5_tuned, eval_valid_idx, split_offsets[eval_split])

    # --- v4 (20q, statevector, windowed) ---
    v4_train_path = V4_DIR / "results" / "phase5_v4_features_train.npy"
    v4_eval_path = V4_DIR / "results" / f"phase5_v4_features_{eval_split}.npy"
    have_v4 = v4_train_path.exists() and v4_eval_path.exists()
    if have_v4:
        F4_train = np.load(v4_train_path)
        F4_eval = np.load(v4_eval_path)
        assert len(F4_train) == len(X_train), f"v4 train features {len(F4_train)} != X_train {len(X_train)}"
        assert len(F4_eval) == len(X_eval), f"v4 {eval_split} features {len(F4_eval)} != X_{eval_split} {len(X_eval)}"
    else:
        print(f"WARNING: v4 canonical {eval_split} features not yet available -- skipping v4 rows this run.")

    # --- Raw-window features (A) for the concatenation experiment ---
    A_train = X_train.reshape(len(X_train), -1)
    A_eval = X_eval.reshape(len(X_eval), -1)
    a_mean, a_std = A_train.mean(axis=0), A_train.std(axis=0) + 1e-12
    A_train_z = (A_train - a_mean) / a_std
    A_eval_z = (A_eval - a_mean) / a_std

    # --- Classical baselines + GBM ceiling probe, pulled in verbatim ---
    baselines_path = REPO_ROOT / "results" / f"baselines_{eval_split}.json"
    baselines = json.load(open(baselines_path))
    assert baselines["eval_split"] == eval_split

    results = {"eval_split": eval_split, "eval_horizons": EVAL_HORIZONS,
               "n_train": len(X_train), f"n_{eval_split}": len(X_eval),
               "std_scaled_to_anomaly_degC": std_degc, "models": {}}

    # Copy classical rows (already have rmse/mae/skill/vpt/dm_vs_persistence/CI at h=1,6)
    for name in CLASSICAL_ROWS:
        rows = {}
        for h in EVAL_HORIZONS:
            m = dict(baselines["metrics"][name][str(h)])
            m["rmse_degC"] = m["rmse"] * std_degc
            m["mae_degC"] = m["mae"] * std_degc
            rows[str(h)] = m
        results["models"][name] = rows

    # GBM ceiling probe: only rmse/mae/nrmse/skill available (no VPT/DM --
    # it's a predictability-ceiling probe, not an RC-comparison baseline,
    # per generate_baselines.py's own documented convention)
    gbm = baselines["ceiling_probes"]["gbm"]["metrics"]
    gbm_rows = {}
    for h in EVAL_HORIZONS:
        if str(h) in gbm:
            m = dict(gbm[str(h)])
            m["rmse_degC"] = m["rmse"] * std_degc
            m["mae_degC"] = m["mae"] * std_degc
            gbm_rows[str(h)] = m
    results["models"]["gbm_ceiling_probe"] = gbm_rows

    for h in EVAL_HORIZONS:
        h_idx = horizons_available.index(h)
        y_train = y_train_all[:, h_idx]
        y_eval = y_eval_all[:, h_idx]
        y_train_res = y_train - y_train_persist
        y_eval_res = y_eval - y_eval_persist

        # Strongest classical baseline's prediction, for DM comparison --
        # null-control Ridge, computed here (not reused from
        # generate_baselines.py, whose null_control_forecast has a
        # slightly different alpha-tuning protocol) so the DM test
        # compares against a prediction array actually held in memory.
        pred_null_res, null_alpha = fit_eval_ridge(A_train_z, y_train_res, A_eval_z, y_eval_res)
        pred_null = pred_null_res + y_eval_persist

        def add_row(model_name, F_train, F_eval, standardize=False, vs_classical=None):
            if standardize:
                m, s = F_train.mean(axis=0), F_train.std(axis=0) + 1e-12
                F_train_use, F_eval_use = (F_train - m) / s, (F_eval - m) / s
            else:
                F_train_use, F_eval_use = F_train, F_eval
            pred_res, alpha = fit_eval_ridge(F_train_use, y_train_res, F_eval_use, y_eval_res)
            pred = pred_res + y_eval_persist
            row = evaluate(model_name, y_eval, pred, y_eval_persist, h, y_strongest_classical=vs_classical)
            row["rmse_degC"] = row["rmse"] * std_degc
            row["mae_degC"] = row["mae"] * std_degc
            row["best_alpha"] = alpha
            results["models"].setdefault(model_name, {})[str(h)] = row
            dmc = row.get("dm_vs_strongest_classical", {}).get("p_value")
            print(f"  h={h} {model_name}: skill={row['skill']*100:+.2f}%  "
                  f"rmse_degC={row['rmse_degC']:.3f}  p_persist={row['dm_vs_persistence']['p_value']:.4f}"
                  + (f"  p_classical={dmc:.4f}" if dmc is not None else ""))

        print(f"\n=== h={h} ===")
        add_row("v5_qrc_residual_12q", F5_train, F5_eval, standardize=True, vs_classical=pred_null)
        if have_v4:
            add_row("v4_qrc_residual_20q", F4_train, F4_eval, standardize=True, vs_classical=pred_null)
        add_row("null_ridge_raw_window", A_train_z, A_eval_z, standardize=False)

        # Concatenation experiment: C = [A, B(v5)]
        C_train = np.concatenate([A_train_z, (F5_train - F5_train.mean(0)) / (F5_train.std(0) + 1e-12)], axis=1)
        C_eval = np.concatenate([A_eval_z, (F5_eval - F5_train.mean(0)) / (F5_train.std(0) + 1e-12)], axis=1)
        add_row("concat_C_raw_plus_v5", C_train, C_eval, standardize=False, vs_classical=pred_null)

        if have_v4:
            C4_train = np.concatenate([A_train_z, (F4_train - F4_train.mean(0)) / (F4_train.std(0) + 1e-12)], axis=1)
            C4_eval = np.concatenate([A_eval_z, (F4_eval - F4_train.mean(0)) / (F4_train.std(0) + 1e-12)], axis=1)
            add_row("concat_C_raw_plus_v4", C4_train, C4_eval, standardize=False, vs_classical=pred_null)

    out_path = REPO_ROOT / "results" / f"full_matched_benchmark_{eval_split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    if not have_v4:
        print(f"\nRe-run this script once results/phase5_v4_features_{{train,{eval_split}}}.npy exist "
              "to add the v4 rows and the v4 concatenation row.")


if __name__ == "__main__":
    main()
