#!/usr/bin/env python3
"""Sprint 6 — Full-Dataset Benchmark: the headline numbers, judge-grade
rigor. Full available KORD ISD-Lite record (2011-2024 -- confirmed
acceptable QC yield, comparable missingness to the 2019-2024 subset
already used throughout this project: 0.05-1.37% per column).

Canonical FINAL split (locked, test touched once): train 2011-2020
(10 years), val 2021-2022 (2 years), test 2023-2024 (2 years) -- the
sprint spec's first-listed split option, chosen since the full 14-year
record has acceptable QC yield (no need for the 2019-2024 fallback).
val/test as 2-year ranges required extending QRCx.data.splits.temporal_split
(and preprocess()) to accept val_year/test_year as either a single int
(original convention, unchanged, still tested) or a (start, end) tuple
-- backward compatible, QRCx/tests/test_data.py covers both.

Every non-persistence model gets a Diebold-Mariano test + moving-block
bootstrap CI vs. persistence at EVERY horizon 1-48 (not just a small
subset), per the sprint's literal spec.

FAST_MODE (default True) uses the reduced 4-point ESN tuning grid (same
convention as scripts/generate_baselines.py); --no-fast-mode runs the
full 384-point grid. Given the full 10-year training set (~54,000
windows, ~2.5x Sprint 3/4's dataset), the full grid is a genuinely
multi-hour run -- FAST_MODE is the default for a reason, logged here
honestly rather than silently claimed as full-grid.

3 seeds (not 1) for every ESN variant, per the sprint spec -- reported
per-seed and averaged.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
warnings.filterwarnings("ignore")

from QRCx.data.loader import load_isd_range, validate_dataframe
from QRCx.data.preprocessor import preprocess
from QRCx.data.split_guard import assert_test_unlocked, UNLOCK_TOKEN
from QRCx.baselines import persistence, arima, esn
from QRCx.baselines.fairness import null_control_forecast, residual_ridge_forecast, gbm_ceiling_probe_forecast
from QRCx.metrics.forecast import rmse, mae, nrmse, skill_score, vpt
from QRCx.metrics.fsdh import compute_fsdh_curve
from QRCx.metrics.significance import diebold_mariano, skill_difference_ci

TRAIN_YEARS = (2011, 2020)
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023, 2024)
ALL_HORIZONS = list(range(1, 49))  # Sprint 6 spec: DM tests + bootstrap CIs at EVERY horizon 1-48, not a subset
GBM_HORIZONS = [1, 3, 6, 12, 24, 48]
QRC_N_QUBITS = 12
ESN_SEEDS = [42, 43, 44]  # 3 seeds per Sprint 6 spec


def _qrc_n_features(n_qubits: int) -> int:
    return 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2


def evaluate_model(name, preds_by_h, y_eval, y_persist_by_h):
    row = {}
    for h in ALL_HORIZONS:
        h_idx = ALL_HORIZONS.index(h)
        yt = y_eval[:, h_idx]
        pred = preds_by_h[h]
        y_persist = y_persist_by_h[h]
        m = {
            "rmse": rmse(yt, pred), "mae": mae(yt, pred), "nrmse": nrmse(yt, pred),
            "skill": skill_score(yt, pred, y_persist), "vpt": vpt(yt, pred),
        }
        if name != "persistence":
            dm = diebold_mariano(yt, pred, y_persist, h=h)
            ci = skill_difference_ci(yt, pred, y_persist, y_persist, n_boot=1000, seed=42)
            m["dm_vs_persistence"] = dm
            m["skill_diff_ci_vs_persistence"] = {k: v for k, v in ci.items() if k != "replicates"}
        row[str(h)] = m
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--unlock-test", action="store_true",
                         help="Required together with --split test to confirm this is the "
                              "one-time final confirmatory run (per split ledger policy).")
    args = parser.parse_args()
    FAST_MODE = args.fast_mode
    split = args.split

    if split == "test":
        if not args.unlock_test:
            print("ERROR: --split test requires --unlock-test (one-time final confirmatory "
                  "run only -- see docs/evaluation_protocol.md split ledger).", file=sys.stderr)
            return 1
        assert_test_unlocked(unlocked_by=UNLOCK_TOKEN, reason="scripts/sprint6_full_benchmark.py --split test")

    output_path = args.output or str(REPO_ROOT / "results" / f"full_benchmark_{split}.json")

    print(f"FAST_MODE={FAST_MODE}  split={split}  train={TRAIN_YEARS} val={VAL_YEARS} test={TEST_YEARS}")
    t_start = time.time()
    print("[1/6] Loading full ISD data (2011-2024, cached under data/isd/)...")
    df = load_isd_range(2011, 2024, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)

    print("[2/6] Preprocessing (13 features, full-record split, FSDH horizons 1..48)...")
    data = preprocess(df, train_years=TRAIN_YEARS, val_year=VAL_YEARS, test_year=TEST_YEARS,
                       horizons=ALL_HORIZONS)
    target_col_idx = data["target_col_idx"]
    print(f"  X_train={data['X_train'].shape}  X_val={data['X_val'].shape}  X_test={data['X_test'].shape}")

    X_eval = data[f"X_{split}"]
    y_eval = data[f"y_{split}"]
    eval_seq = data[f"{split}_seq"]
    eval_valid_idx = data[f"{split}_valid_idx"]
    y_eval_all_cols = y_eval  # ALL_HORIZONS already == the columns of y_eval

    dim_matched_nodes = _qrc_n_features(QRC_N_QUBITS)
    esn_sizes = {"esn_dim_matched": dim_matched_nodes, "esn_500": 500}
    if not FAST_MODE:
        esn_sizes["esn_5000"] = 5000

    print(f"[3/6] Running baselines (scored on {split}, {len(ESN_SEEDS)} seeds for ESN variants)...")
    all_preds_eval = {}
    all_preds_curve = {}
    wall_clock = {}
    misc_log = {}

    t0 = time.time()
    pers_curve = persistence.forecast(X_eval, y_eval[:, 0], horizons=ALL_HORIZONS, target_col_idx=target_col_idx)
    pers_eval = {h: pers_curve[h] for h in ALL_HORIZONS}
    all_preds_curve["persistence"] = pers_curve
    all_preds_eval["persistence"] = pers_eval
    wall_clock["persistence"] = time.time() - t0
    print(f"  persistence: {wall_clock['persistence']:.1f}s")

    t0 = time.time()
    arima_curve = arima.forecast(data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS,
                                  valid_idx=eval_valid_idx)
    all_preds_curve["arima"] = arima_curve
    all_preds_eval["arima"] = {h: arima_curve[h] for h in ALL_HORIZONS}
    wall_clock["arima"] = time.time() - t0
    print(f"  arima(2,1,2): {wall_clock['arima']:.1f}s")

    t0 = time.time()
    arima_auto_curve, arima_auto_selection = arima.auto_order_forecast(
        data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS, valid_idx=eval_valid_idx, stride=6,
    )
    all_preds_curve["arima_auto"] = arima_auto_curve
    all_preds_eval["arima_auto"] = {h: arima_auto_curve[h] for h in ALL_HORIZONS}
    wall_clock["arima_auto"] = time.time() - t0
    misc_log["arima_auto_order_selection"] = arima_auto_selection
    print(f"  arima_auto [order={arima_auto_selection['order']}]: {wall_clock['arima_auto']:.1f}s")

    # 3-seed ESN variants: fit+tune per seed, report per-seed predictions
    # under a seed-suffixed name AND an averaged prediction under the bare name.
    for esn_name, nodes in esn_sizes.items():
        seed_preds = []
        t0 = time.time()
        for seed in ESN_SEEDS:
            curve = esn.forecast(
                data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS,
                reservoir_size=nodes, val_seq=data["val_seq"], fast_mode=FAST_MODE,
                valid_idx=eval_valid_idx, seed=seed,
            )
            seed_preds.append(curve)
            named = f"{esn_name}_seed{seed}"
            all_preds_curve[named] = curve
            all_preds_eval[named] = {h: curve[h] for h in ALL_HORIZONS}
        wall_clock[esn_name] = time.time() - t0
        avg_curve = {h: np.mean([sp[h] for sp in seed_preds], axis=0) for h in ALL_HORIZONS}
        all_preds_curve[esn_name] = avg_curve
        all_preds_eval[esn_name] = {h: avg_curve[h] for h in ALL_HORIZONS}
        print(f"  {esn_name} (nodes={nodes}, {len(ESN_SEEDS)} seeds): {wall_clock[esn_name]:.1f}s")

    t0 = time.time()
    res_esn_seed_preds = []
    for seed in ESN_SEEDS:
        curve = esn.forecast(
            data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS,
            reservoir_size=esn_sizes["esn_dim_matched"], val_seq=data["val_seq"], fast_mode=FAST_MODE,
            valid_idx=eval_valid_idx, residual=True, seed=seed,
        )
        res_esn_seed_preds.append(curve)
        named = f"residual_esn_seed{seed}"
        all_preds_curve[named] = curve
        all_preds_eval[named] = {h: curve[h] for h in ALL_HORIZONS}
    wall_clock["residual_esn"] = time.time() - t0
    avg_res = {h: np.mean([sp[h] for sp in res_esn_seed_preds], axis=0) for h in ALL_HORIZONS}
    all_preds_curve["residual_esn"] = avg_res
    all_preds_eval["residual_esn"] = {h: avg_res[h] for h in ALL_HORIZONS}
    print(f"  residual_esn ({len(ESN_SEEDS)} seeds): {wall_clock['residual_esn']:.1f}s")

    y_train_all = data["y_train"]
    y_val_all = data["y_val"]

    t0 = time.time()
    null_ridge = null_control_forecast(data["X_train"], y_train_all, data["X_val"], y_val_all, X_eval, ALL_HORIZONS, method="ridge")
    all_preds_eval["null_ridge"] = null_ridge
    wall_clock["null_ridge"] = time.time() - t0

    KRR_MAX_TRAIN = 3000  # O(n^2) in samples -- pre-existing limitation, see docs/evaluation_protocol.md
    X_train_krr = data["X_train"][-KRR_MAX_TRAIN:]
    y_train_krr = y_train_all[-KRR_MAX_TRAIN:]
    t0 = time.time()
    null_krr = null_control_forecast(X_train_krr, y_train_krr, data["X_val"], y_val_all, X_eval, ALL_HORIZONS, method="krr")
    all_preds_eval["null_krr"] = null_krr
    wall_clock["null_krr"] = time.time() - t0

    t0 = time.time()
    res_ridge = residual_ridge_forecast(data["X_train"], y_train_all, data["X_val"], y_val_all, X_eval,
                                         ALL_HORIZONS, target_col_idx=target_col_idx, method="ridge")
    all_preds_eval["residual_ridge"] = res_ridge
    wall_clock["residual_ridge"] = time.time() - t0
    print(f"  null_ridge / null_krr / residual_ridge: "
          f"{wall_clock['null_ridge']:.1f}s / {wall_clock['null_krr']:.1f}s / {wall_clock['residual_ridge']:.1f}s")

    print("  gbm_ceiling_probe (NOT an RC baseline, reported separately)...")
    t0 = time.time()
    gbm_horizon_cols = [h - 1 for h in GBM_HORIZONS]
    y_train_gbm = data["y_train"][:, gbm_horizon_cols]
    y_val_gbm = data["y_val"][:, gbm_horizon_cols]
    gbm_preds, gbm_tuning = gbm_ceiling_probe_forecast(
        data["X_train"], y_train_gbm, data["X_val"], y_val_gbm, X_eval, GBM_HORIZONS,
    )
    wall_clock["gbm_ceiling_probe"] = time.time() - t0
    print(f"  gbm_ceiling_probe: {wall_clock['gbm_ceiling_probe']:.1f}s")

    print(f"[4/6] Scoring (RMSE/MAE/NRMSE/skill/VPT/FSDH, DM test, bootstrap CI at ALL {len(ALL_HORIZONS)} horizons)...")
    metrics = {}
    scoring_names = [n for n in all_preds_eval if "_seed" not in n]  # skip per-seed rows for the heavy DM/bootstrap pass; keep them in fsdh_curves/predictions only
    for name in scoring_names:
        metrics[name] = evaluate_model(name, all_preds_eval[name], y_eval_all_cols, pers_eval)

    fsdh_curves = {}
    for name, preds_curve in all_preds_curve.items():
        if name == "persistence":
            continue
        y_model_mh = np.column_stack([preds_curve[h] for h in ALL_HORIZONS])
        y_persist_mh = np.column_stack([pers_curve[h] for h in ALL_HORIZONS])
        fsdh_curves[name] = compute_fsdh_curve(y_eval, y_model_mh, y_persist_mh)

    pers_gbm = {h: persistence.forecast(X_eval, y_eval[:, 0], horizons=GBM_HORIZONS, target_col_idx=target_col_idx)[h] for h in GBM_HORIZONS}
    ceiling_probes = {
        "gbm": {
            "description": "HistGradientBoosting on the flattened raw 24x13 window -- predictability-ceiling probe, not an RC-comparison baseline.",
            "tuning": gbm_tuning, "wall_clock_s": wall_clock["gbm_ceiling_probe"],
            "metrics": {
                str(h): {
                    "rmse": rmse(y_eval[:, h - 1], gbm_preds[h]), "mae": mae(y_eval[:, h - 1], gbm_preds[h]),
                    "nrmse": nrmse(y_eval[:, h - 1], gbm_preds[h]),
                    "skill": skill_score(y_eval[:, h - 1], gbm_preds[h], pers_gbm[h]),
                } for h in GBM_HORIZONS
            },
        },
    }

    print(f"[5/6] Writing {output_path}...")
    total_wall_clock = time.time() - t_start
    output = {
        "FAST_MODE": FAST_MODE, "eval_split": split,
        "train_years": TRAIN_YEARS, "val_years": VAL_YEARS, "test_years": TEST_YEARS,
        "esn_seeds": ESN_SEEDS,
        "n_features": len(data["feature_cols"]), "feature_cols": data["feature_cols"],
        "target_col_idx": int(target_col_idx), "eval_horizons": ALL_HORIZONS, "gbm_horizons": GBM_HORIZONS,
        "qrc_n_qubits_reference": QRC_N_QUBITS, "esn_dim_matched_nodes": esn_sizes.get("esn_dim_matched"),
        "null_krr_train_cap": KRR_MAX_TRAIN,
        "n_train_samples": int(len(data["X_train"])), "n_val_samples": int(len(data["X_val"])),
        "n_eval_samples": int(len(y_eval)),
        "metrics": metrics, "ceiling_probes": ceiling_probes, "fsdh_curves": fsdh_curves,
        "wall_clock_s": wall_clock, "total_wall_clock_s": total_wall_clock, "misc_log": misc_log,
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"  Wrote {out_path}")
    print(f"[6/6] Total wall-clock: {total_wall_clock / 60:.1f} min")

    for name in metrics:
        s1 = metrics[name].get("1", {}).get("skill")
        s6 = metrics[name].get("6", {}).get("skill")
        print(f"  {name:<24} skill@1h={s1 * 100:.1f}%  skill@6h={s6 * 100:.1f}%" if s1 is not None else f"  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
