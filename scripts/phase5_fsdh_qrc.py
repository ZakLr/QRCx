#!/usr/bin/env python3
"""FINAL SPRINT: real FSDH (Forecast Skill Duration Horizon, all 48 hours)
for the QRC/concat models -- a required headline-table column
(Sec. 6.3 of the paper spec) that phase5_final_benchmark.py never
computed (it only evaluated h in {1,6}). Classical baselines already
have FSDH from generate_baselines.py's ALL_HORIZONS=1..48 sweep on the
raw sequence; QRC/concat models need their own sweep since their
features are fixed arrays aligned to specific windowed samples, not a
raw sequence generate_baselines.py can re-window at arbitrary horizons.

Restricts to the subset of samples valid at every horizon 1..48
(valid_idx + W - 1 + 48 < len(seq)) so compute_fsdh_curve's fixed-
sample-count requirement is met, refits Ridge per horizon (cheap: the
features are already driven, this is just 48 more Ridge fits per
model) using the SAME fit protocol as phase5_final_benchmark.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.metrics.fsdh import compute_fsdh_curve
from QRCx.metrics.forecast import compute_vpt_curve
from QRCx.data.split_guard import assert_test_unlocked, UNLOCK_TOKEN

MAX_H = 48
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
DATA_NPZ = REPO_ROOT / "data" / "canonical_seq.npz"
V5_DIR = REPO_ROOT / "results" / "phase1_v5_canonical"
V4_DIR = REPO_ROOT


def fit_eval_ridge(F_train, y_train_res, F_val, y_val_res):
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


def build_v5_aligned(v5_features, valid_idx, split_offset, W=24):
    return v5_features[split_offset + valid_idx + (W - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    eval_split = args.split
    if eval_split == "test":
        assert_test_unlocked(unlocked_by=UNLOCK_TOKEN,
                              reason="scripts/phase5_fsdh_qrc.py --split test")

    npz = np.load(DATA_NPZ)
    target_col_idx = int(npz["target_col_idx"])
    n_train_seq, n_val_seq = len(npz["train_seq"]), len(npz["val_seq"])
    split_offsets = {"train": 0, "val": n_train_seq, "test": n_train_seq + n_val_seq}
    seqs = {"train": npz["train_seq"], "val": npz["val_seq"], "test": npz["test_seq"]}

    X_train = npz["X_train"]
    X_eval = npz[f"X_{eval_split}"]
    train_valid_idx = npz["train_valid_idx"]
    eval_valid_idx = npz[f"{eval_split}_valid_idx"]
    W = X_train.shape[1]

    train_seq = seqs["train"]
    eval_seq = seqs[eval_split]

    # Restrict to samples valid at EVERY horizon 1..MAX_H (fixed sample
    # count across the sweep, required by compute_fsdh_curve).
    train_keep = train_valid_idx + (W - 1) + MAX_H < len(train_seq)
    eval_keep = eval_valid_idx + (W - 1) + MAX_H < len(eval_seq)
    train_valid_idx = train_valid_idx[train_keep]
    eval_valid_idx = eval_valid_idx[eval_keep]
    print(f"Samples valid at all h<= {MAX_H}: train={len(train_valid_idx)}, "
          f"{eval_split}={len(eval_valid_idx)} (dropped {(~train_keep).sum()}/{(~eval_keep).sum()})")

    def target_at_h(seq, valid_idx, h):
        return seq[valid_idx + (W - 1) + h, target_col_idx]

    y_train_persist = train_seq[train_valid_idx + (W - 1), target_col_idx]
    y_eval_persist = eval_seq[eval_valid_idx + (W - 1), target_col_idx]

    v5_tuned = np.load(V5_DIR / "phase1_features_tuned.npy")
    F5_train = build_v5_aligned(v5_tuned, train_valid_idx, split_offsets["train"], W)
    F5_eval = build_v5_aligned(v5_tuned, eval_valid_idx, split_offsets[eval_split], W)

    v4_train_path = V4_DIR / "results" / "phase5_v4_features_train.npy"
    v4_eval_path = V4_DIR / "results" / f"phase5_v4_features_{eval_split}.npy"
    have_v4 = v4_train_path.exists() and v4_eval_path.exists()
    F4_train_full = np.load(v4_train_path) if have_v4 else None
    F4_eval_full = np.load(v4_eval_path) if have_v4 else None
    # v4 features are 1:1 aligned to X_train/X_eval rows (pre-drop), so
    # subselect by the same boolean masks used above on the *_valid_idx arrays.
    if have_v4:
        F4_train = F4_train_full[train_keep]
        F4_eval = F4_eval_full[eval_keep]

    A_train = X_train[train_keep].reshape(train_keep.sum(), -1)
    A_eval = X_eval[eval_keep].reshape(eval_keep.sum(), -1)
    a_mean, a_std = A_train.mean(axis=0), A_train.std(axis=0) + 1e-12
    A_train_z = (A_train - a_mean) / a_std
    A_eval_z = (A_eval - a_mean) / a_std

    models = {}

    def standardized(F_train, F_eval):
        m, s = F_train.mean(axis=0), F_train.std(axis=0) + 1e-12
        return (F_train - m) / s, (F_eval - m) / s

    F5_train_z, F5_eval_z = standardized(F5_train, F5_eval)
    models["v5_qrc_residual_12q"] = (F5_train_z, F5_eval_z)
    models["null_ridge_raw_window"] = (A_train_z, A_eval_z)
    C5_train = np.concatenate([A_train_z, F5_train_z], axis=1)
    C5_eval = np.concatenate([A_eval_z, F5_eval_z], axis=1)
    models["concat_C_raw_plus_v5"] = (C5_train, C5_eval)
    if have_v4:
        F4_train_z, F4_eval_z = standardized(F4_train, F4_eval)
        models["v4_qrc_residual_20q"] = (F4_train_z, F4_eval_z)
        C4_train = np.concatenate([A_train_z, F4_train_z], axis=1)
        C4_eval = np.concatenate([A_eval_z, F4_eval_z], axis=1)
        models["concat_C_raw_plus_v4"] = (C4_train, C4_eval)

    n_eval = eval_keep.sum()
    y_true_h = np.zeros((n_eval, MAX_H))
    y_persist_h = np.zeros((n_eval, MAX_H))
    y_model_h = {name: np.zeros((n_eval, MAX_H)) for name in models}

    for h in range(1, MAX_H + 1):
        y_train_h = target_at_h(train_seq, train_valid_idx, h)
        y_eval_h_arr = target_at_h(eval_seq, eval_valid_idx, h)
        y_true_h[:, h - 1] = y_eval_h_arr
        y_persist_h[:, h - 1] = y_eval_persist

        y_train_res = y_train_h - y_train_persist
        y_eval_res = y_eval_h_arr - y_eval_persist
        for name, (F_train, F_eval) in models.items():
            pred_res, _ = fit_eval_ridge(F_train, y_train_res, F_eval, y_eval_res)
            y_model_h[name][:, h - 1] = pred_res + y_eval_persist
        if h % 12 == 0:
            print(f"  h={h}/{MAX_H} done")

    fsdh_results = {}
    vpt_results = {}
    for name in models:
        fsdh_val = compute_fsdh_curve(y_true_h, y_model_h[name], y_persist_h)
        fsdh_results[name] = fsdh_val
        # Real VPT (max consecutive horizon with NRMSE < threshold) --
        # distinct from the per-sample 0/1 flag `metrics[name]['vpt']`
        # computed elsewhere, which is NOT the official challenge doc's
        # VPT definition (a real bug this fixes).
        vpt_val = compute_vpt_curve(y_true_h, y_model_h[name])
        vpt_results[name] = vpt_val
        print(f"FSDH[{name}] = {fsdh_val}   VPT[{name}] = {vpt_val}")

    out_path = REPO_ROOT / "results" / f"qrc_fsdh_{eval_split}.json"
    with open(out_path, "w") as f:
        json.dump({"eval_split": eval_split, "max_horizon": MAX_H,
                    "n_samples": int(n_eval), "fsdh": fsdh_results,
                    "vpt": vpt_results}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
