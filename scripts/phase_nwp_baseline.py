#!/usr/bin/env python3
"""FINAL SPRINT (post-review): real NWP-style baseline comparison.

The official challenge doc explicitly permits comparing against publicly
available NWP forecasts (NOAA GFS, ECMWF IFS) "rather than running their
own" model. We use Open-Meteo's free, no-auth "Previous Runs API"
(https://open-meteo.com/en/docs/previous-runs-api), which archives real
GFS forecast output realigned to fixed day-ahead lead times
(temperature_2m_previous_dayN = the value forecast N*24h before valid
time). This only offers DAILY lead-time offsets (24h, 48h, ...), not our
full hourly horizon set -- but 24h and 48h happen to be two of our own
six canonical horizons, so a real, apples-to-apples comparison is
possible there without running our own NWP pipeline (data/nwp/
openmeteo_kord_2024.json, fetched 2026-07-26, lat/lon 41.9786/-87.9048,
KORD).

Compares GFS (via Open-Meteo) against persistence and our own
null-control Ridge / v5 QRC at h=24,48, all on the identical canonical
test-2024 period, in real raw-Celsius space (skill is scale-invariant,
so no anomaly-transform inversion is needed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.data.loader import load_isd_range, validate_dataframe
from QRCx.metrics.significance import diebold_mariano

NWP_PATH = REPO_ROOT / "data" / "nwp" / "openmeteo_kord_2024.json"
DATA_NPZ = REPO_ROOT / "data" / "canonical_seq.npz"
V5_DIR = REPO_ROOT / "results" / "phase1_v5_canonical"
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def skill(y, p, y_persist):
    rp, rb = rmse(y, p), rmse(y, y_persist)
    return float(1.0 - rp / rb) if rb > 0 else float("nan")


def fit_eval_ridge(F_train, y_train, F_eval, y_eval_target_unused=None):
    n_tune = int(0.2 * len(F_train))
    F_fit, F_tune = F_train[:-n_tune], F_train[-n_tune:]
    y_fit, y_tune = y_train[:-n_tune], y_train[-n_tune:]
    best_alpha, best_mse = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha).fit(F_fit, y_fit)
        mse = float(np.mean((y_tune - m.predict(F_tune)) ** 2))
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    final = Ridge(alpha=best_alpha).fit(F_train, y_train)
    return final.predict(F_eval)


def main():
    # --- Real GFS forecast (Open-Meteo Previous Runs API archive) ---
    nwp = json.load(open(NWP_PATH))
    h = nwp["hourly"]
    nwp_df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "actual_om": h["temperature_2m"],
        "gfs_24h": h["temperature_2m_previous_day1"],
        "gfs_48h": h["temperature_2m_previous_day2"],
    }).set_index("time")

    # --- Real ISD ground truth (our own pipeline's raw T_db, Celsius) ---
    df = load_isd_range(2019, 2024, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)
    test_df = df.loc["2024-01-01":"2024-12-31", ["T_db"]].copy()

    joined = test_df.join(nwp_df, how="inner")
    print(f"Joined {len(joined)} hours (of {len(test_df)} real ISD hours, "
          f"{len(nwp_df)} Open-Meteo hours).")

    results = {"station": "KORD", "year": 2024, "source": "open-meteo.com previous-runs-api "
               "(real archived GFS forecast, day-ahead lead times only)", "horizons": {}}

    npz = np.load(DATA_NPZ)
    horizons_available = [int(x) for x in npz["horizons"]]
    target_col_idx = int(npz["target_col_idx"])
    X_train, y_train_all = npz["X_train"], npz["y_train"]
    X_test, y_test_all = npz["X_test"], npz["y_test"]
    train_valid_idx, test_valid_idx = npz["train_valid_idx"], npz["test_valid_idx"]
    n_train_seq, n_val_seq = len(npz["train_seq"]), len(npz["val_seq"])
    split_offset_test = n_train_seq + n_val_seq
    W = X_train.shape[1]

    y_train_persist = X_train[:, -1, target_col_idx]
    y_test_persist = X_test[:, -1, target_col_idx]
    A_train = X_train.reshape(len(X_train), -1)
    A_test = X_test.reshape(len(X_test), -1)
    a_mean, a_std = A_train.mean(axis=0), A_train.std(axis=0) + 1e-12
    A_train_z, A_test_z = (A_train - a_mean) / a_std, (A_test - a_mean) / a_std

    v5_tuned = np.load(V5_DIR / "phase1_features_tuned.npy")
    F5_train = v5_tuned[train_valid_idx + (W - 1)]
    F5_test = v5_tuned[split_offset_test + test_valid_idx + (W - 1)]
    f5_mean, f5_std = F5_train.mean(axis=0), F5_train.std(axis=0) + 1e-12
    F5_train_z, F5_test_z = (F5_train - f5_mean) / f5_std, (F5_test - f5_mean) / f5_std

    for h_lead, gfs_col in [(24, "gfs_24h"), (48, "gfs_48h")]:
        sub = joined.dropna(subset=[gfs_col, "T_db"])
        y_actual = sub["T_db"].to_numpy()
        y_gfs = sub[gfs_col].to_numpy()
        y_persist_om = sub["T_db"].shift(h_lead).reindex(sub.index).to_numpy()
        valid = ~(np.isnan(y_persist_om) | np.isnan(y_actual) | np.isnan(y_gfs))
        y_actual_v, y_gfs_v, y_persist_v = y_actual[valid], y_gfs[valid], y_persist_om[valid]

        skill_gfs = skill(y_actual_v, y_gfs_v, y_persist_v)
        dm_gfs = diebold_mariano(y_actual_v, y_gfs_v, y_persist_v, h=h_lead)

        # Our own models at the same horizon, on the canonical split (real, not estimated)
        h_idx = horizons_available.index(h_lead)
        y_train_res = y_train_all[:, h_idx] - y_train_persist
        y_test_res = y_test_all[:, h_idx] - y_test_persist

        pred_null_res = fit_eval_ridge(A_train_z, y_train_res, A_test_z)
        pred_null = pred_null_res + y_test_persist
        skill_null = skill(y_test_all[:, h_idx], pred_null, y_test_persist)

        pred_v5_res = fit_eval_ridge(F5_train_z, y_train_res, F5_test_z)
        pred_v5 = pred_v5_res + y_test_persist
        skill_v5 = skill(y_test_all[:, h_idx], pred_v5, y_test_persist)

        results["horizons"][str(h_lead)] = {
            "n_hours_compared": int(valid.sum()),
            "gfs_skill_vs_persistence": skill_gfs,
            "gfs_dm_vs_persistence": dm_gfs,
            "null_control_ridge_skill_vs_persistence": skill_null,
            "v5_qrc_skill_vs_persistence": skill_v5,
        }
        print(f"h={h_lead}h  (n={valid.sum()}): GFS skill={skill_gfs*100:+.1f}%  "
              f"(p={dm_gfs['p_value']:.4f})  null_ridge={skill_null*100:+.1f}%  "
              f"v5={skill_v5*100:+.1f}%")

    out_path = REPO_ROOT / "results" / "nwp_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
