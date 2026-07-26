#!/usr/bin/env python3
"""Sprint 3: bulletproof classical-baseline results, written to
results/baselines.json — the single source of truth for every baseline
number reported in README/paper (per project policy: no number appears
anywhere that isn't traceable to a results/*.json file).

Baselines (all sharing identical 13 features, identical strict temporal
splits 2019-2022/2023/2024, identical StandardScaler -- see
data/preprocessor.py):
  - Persistence (floor)
  - ARIMA(2,1,2) on the raw chronological anomaly sequence (spec-fixed
    order), AND auto-selected ARIMA(p,0,q) (AIC on train) -- two lines,
    so the ARIMA row isn't left as a strawman after the ESN fix.
  - ESN (dimension-matched to the 12-qubit QRC: 234 units), ESN-500,
    ESN-5000 -- each grid-tuned on the validation split only
  - Residual-ESN (predicts the same anomaly-residual target ResidualQRC
    does), dimension-matched size
  - Null control: Ridge and KRR directly on the flattened raw 24x13
    window, no reservoir
  - Residual-Ridge: same residual target, flattened-window features
  - GBM predictability-ceiling probe (HistGradientBoosting on the
    flattened window, h in {1,3,6,12}) -- NOT an RC-comparison baseline,
    reported separately as "ceiling_probes" in the output JSON.

For every RC-comparison baseline vs. persistence at the eval horizons,
this script also runs a Diebold-Mariano test (HAC-corrected for h-step-
ahead forecasts) and a moving-block-bootstrap 95% CI for the skill
difference (metrics/significance.py).

Scored on val-2023 by default (Sprint 3 closeout addendum: all
development decisions and DoD/gate checks are made on val, never test;
test-2024 is reserved for a one-time final confirmatory report -- see
docs/evaluation_protocol.md's split ledger and QRCx.data.split_guard).
--split test requires an explicit unlock and is logged in the ledger.

Both --fast-mode and --no-fast-mode run on the full real 2019-2024 KORD
ISD data -- FAST_MODE controls only the ESN tuning grid's resolution and
which ESN sizes are tried: FAST_MODE=True (default) uses the reduced
4-point grid and skips ESN-5000 (each full-grid ESN fit takes minutes;
ESN-5000 at the full 384-point grid is a paper-grade, multi-hour run --
see docs/evaluation_protocol.md for measured wall-clock and why ESN-5000
was not run at the full grid this session).
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
from QRCx.data.split_guard import assert_test_unlocked
from QRCx.baselines import persistence, arima, esn
from QRCx.baselines.fairness import null_control_forecast, residual_ridge_forecast, gbm_ceiling_probe_forecast
from QRCx.metrics.forecast import rmse, mae, nrmse, skill_score, vpt, compute_vpt_curve
from QRCx.metrics.fsdh import compute_fsdh_curve
from QRCx.metrics.significance import diebold_mariano, skill_difference_ci

EVAL_HORIZONS = [1, 6]
GBM_HORIZONS = [1, 3, 6, 12]
FSDH_MAX = 48
ALL_HORIZONS = list(range(1, FSDH_MAX + 1))
QRC_N_QUBITS = 12  # reference config (global invariant)


def _qrc_n_features(n_qubits: int) -> int:
    """Matches AtmosphericQRC.n_features without paying for circuit
    construction (see reservoir/tfim.py: 3N single-body + 3*C(N,2) two-body)."""
    return 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2


def evaluate_model(name, preds_by_h, y_eval, y_persist_by_h):
    row = {}
    for h in EVAL_HORIZONS:
        h_idx = EVAL_HORIZONS.index(h)
        yt = y_eval[:, h_idx]
        pred = preds_by_h[h]
        y_persist = y_persist_by_h[h]
        m = {
            "rmse": rmse(yt, pred), "mae": mae(yt, pred), "nrmse": nrmse(yt, pred),
            "skill": skill_score(yt, pred, y_persist),
            "vpt": vpt(yt, pred),
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
    args = parser.parse_args()
    FAST_MODE = args.fast_mode
    split = args.split

    if split == "test":
        assert_test_unlocked(reason="scripts/generate_baselines.py --split test")

    output_path = args.output or str(REPO_ROOT / "results" / f"baselines_{split}.json")

    print(f"FAST_MODE={FAST_MODE}  split={split}")
    print("[1/5] Loading ISD data (2019-2024, cached under data/isd/)...")
    df = load_isd_range(2019, 2024, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)

    print("[2/5] Preprocessing (13 features, strict temporal split, FSDH horizons 1..48)...")
    data = preprocess(df, horizons=ALL_HORIZONS)
    target_col_idx = data["target_col_idx"]
    print(f"  X_train={data['X_train'].shape}  X_val={data['X_val'].shape}  X_test={data['X_test'].shape}")

    X_eval = data[f"X_{split}"]
    y_eval = data[f"y_{split}"]
    eval_seq = data[f"{split}_seq"]
    eval_valid_idx = data[f"{split}_valid_idx"]
    y_eval_eval_cols = y_eval[:, [h - 1 for h in EVAL_HORIZONS]]

    dim_matched_nodes = _qrc_n_features(QRC_N_QUBITS)
    if FAST_MODE:
        # Reduced ESN grid (4 points) and no ESN-5000 (a full-384-point
        # grid at 5000 units is a multi-hour paper-grade run — not
        # executed under FAST_MODE; see docs/evaluation_protocol.md).
        esn_sizes = {"esn_dim_matched": dim_matched_nodes, "esn_500": 500}
    else:
        esn_sizes = {"esn_dim_matched": dim_matched_nodes, "esn_500": 500, "esn_5000": 5000}

    print(f"[3/5] Running baselines (scored on {split})...")
    all_preds_eval = {}
    all_preds_curve = {}
    wall_clock = {}
    misc_log = {}

    t0 = time.time()
    pers_curve = persistence.forecast(X_eval, y_eval[:, 0], horizons=ALL_HORIZONS, target_col_idx=target_col_idx)
    pers_eval = {h: pers_curve[h] for h in EVAL_HORIZONS}
    all_preds_curve["persistence"] = pers_curve
    all_preds_eval["persistence"] = pers_eval
    wall_clock["persistence"] = time.time() - t0
    print(f"  persistence: {wall_clock['persistence']:.1f}s")

    t0 = time.time()
    arima_curve = arima.forecast(data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS,
                                  valid_idx=eval_valid_idx)
    all_preds_curve["arima"] = arima_curve
    all_preds_eval["arima"] = {h: arima_curve[h] for h in EVAL_HORIZONS}
    wall_clock["arima"] = time.time() - t0
    print(f"  arima(2,1,2) [spec-fixed]: {wall_clock['arima']:.1f}s")

    t0 = time.time()
    # stride is decoupled from max(ALL_HORIZONS)=48 (the FSDH curve's
    # range): rolling_origin_forecast's default stride=max(horizons) would
    # otherwise re-anchor only once every 48h, defeating the point of the
    # rolling-origin fix for the eval horizons that actually matter (1h,
    # 6h) -- see docs/evaluation_protocol.md Section 4b for the diagnosis
    # of this exact bug, caught during Sprint 3 closeout.
    arima_auto_curve, arima_auto_selection = arima.auto_order_forecast(
        data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS, valid_idx=eval_valid_idx,
        stride=max(EVAL_HORIZONS),
    )
    all_preds_curve["arima_auto"] = arima_auto_curve
    all_preds_eval["arima_auto"] = {h: arima_auto_curve[h] for h in EVAL_HORIZONS}
    wall_clock["arima_auto"] = time.time() - t0
    misc_log["arima_auto_order_selection"] = arima_auto_selection
    print(f"  arima_auto [AIC-selected order={arima_auto_selection['order']}]: {wall_clock['arima_auto']:.1f}s")

    for esn_name, nodes in esn_sizes.items():
        t0 = time.time()
        esn_curve = esn.forecast(
            data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS,
            reservoir_size=nodes, val_seq=data["val_seq"], fast_mode=FAST_MODE,
            valid_idx=eval_valid_idx,
        )
        all_preds_curve[esn_name] = esn_curve
        all_preds_eval[esn_name] = {h: esn_curve[h] for h in EVAL_HORIZONS}
        wall_clock[esn_name] = time.time() - t0
        print(f"  {esn_name} (nodes={nodes}): {wall_clock[esn_name]:.1f}s")

    t0 = time.time()
    res_esn_curve = esn.forecast(
        data["train_seq"], eval_seq, target_col_idx, ALL_HORIZONS,
        reservoir_size=esn_sizes["esn_dim_matched"], val_seq=data["val_seq"], fast_mode=FAST_MODE,
        valid_idx=eval_valid_idx, residual=True,
    )
    all_preds_curve["residual_esn"] = res_esn_curve
    all_preds_eval["residual_esn"] = {h: res_esn_curve[h] for h in EVAL_HORIZONS}
    wall_clock["residual_esn"] = time.time() - t0
    print(f"  residual_esn: {wall_clock['residual_esn']:.1f}s")

    eval_horizon_cols = [h - 1 for h in EVAL_HORIZONS]
    y_train_eval = data["y_train"][:, eval_horizon_cols]
    y_val_eval = data["y_val"][:, eval_horizon_cols]

    t0 = time.time()
    null_ridge = null_control_forecast(data["X_train"], y_train_eval, data["X_val"], y_val_eval, X_eval, EVAL_HORIZONS, method="ridge")
    all_preds_eval["null_ridge"] = null_ridge
    wall_clock["null_ridge"] = time.time() - t0

    # KRR is O(n^2) in training samples (a dense n x n kernel matrix per
    # gamma/alpha combo, refit across TimeSeriesSplit folds) -- at the full
    # ~22-35k training windows this is intractable (tens of GB). This is a
    # pre-existing limitation of KRRReadout used pipeline-wide (e.g.
    # DirectQRC/ResidualQRC's readout="krr"), not new to Sprint 3; not
    # fixed here (out of scope), but capped for this null-control baseline
    # specifically so the script completes. The eval set is NOT subsampled
    # -- only the KRR fit's training tail is capped.
    KRR_MAX_TRAIN = 3000
    X_train_krr = data["X_train"][-KRR_MAX_TRAIN:]
    y_train_krr = y_train_eval[-KRR_MAX_TRAIN:]
    t0 = time.time()
    null_krr = null_control_forecast(X_train_krr, y_train_krr, data["X_val"], y_val_eval, X_eval, EVAL_HORIZONS, method="krr")
    all_preds_eval["null_krr"] = null_krr
    wall_clock["null_krr"] = time.time() - t0

    t0 = time.time()
    res_ridge = residual_ridge_forecast(data["X_train"], y_train_eval, data["X_val"], y_val_eval, X_eval,
                                         EVAL_HORIZONS, target_col_idx=target_col_idx, method="ridge")
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

    print(f"[4/5] Scoring (RMSE/MAE/NRMSE/skill/VPT/FSDH, DM test, bootstrap CI) on {split}...")
    metrics = {}
    for name, preds_eval in all_preds_eval.items():
        metrics[name] = evaluate_model(name, preds_eval, y_eval_eval_cols, pers_eval)

    fsdh_curves = {}
    vpt_curves = {}
    skill_curves = {}
    for name, preds_curve in all_preds_curve.items():
        if name == "persistence":
            continue
        y_model_mh = np.column_stack([preds_curve[h] for h in ALL_HORIZONS])
        y_persist_mh = np.column_stack([pers_curve[h] for h in ALL_HORIZONS])
        fsdh_curves[name] = compute_fsdh_curve(y_eval, y_model_mh, y_persist_mh)
        # Real VPT (Valid Prediction Time): max consecutive horizon where
        # NRMSE < threshold -- NOT the same as metrics[name]['vpt'], which
        # is a per-sample 0/1 flag at a single horizon and was being
        # mis-reported as "VPT" in the paper's headline table (a real bug,
        # caught against the official challenge doc's VPT definition).
        vpt_curves[name] = compute_vpt_curve(y_eval, y_model_mh)
        # Real per-horizon skill, not just the scalar FSDH -- needed for the
        # skill-vs-horizon figure; previously computed in-memory above and
        # discarded, a real gap fixed here rather than worked around again.
        skill_curves[name] = {
            str(h): skill_score(y_eval[:, h_idx], preds_curve[h], pers_curve[h])
            for h_idx, h in enumerate(ALL_HORIZONS)
        }

    pers_gbm = {h: persistence.forecast(X_eval, y_eval[:, 0], horizons=GBM_HORIZONS, target_col_idx=target_col_idx)[h] for h in GBM_HORIZONS}
    ceiling_probes = {
        "gbm": {
            "description": "HistGradientBoosting on the flattened raw 24x13 window -- "
                            "predictability-ceiling probe, not an RC-comparison baseline.",
            "tuning": gbm_tuning,
            "wall_clock_s": wall_clock["gbm_ceiling_probe"],
            "metrics": {
                str(h): {
                    "rmse": rmse(y_eval[:, h - 1], gbm_preds[h]),
                    "mae": mae(y_eval[:, h - 1], gbm_preds[h]),
                    "nrmse": nrmse(y_eval[:, h - 1], gbm_preds[h]),
                    "skill": skill_score(y_eval[:, h - 1], gbm_preds[h], pers_gbm[h]),
                } for h in GBM_HORIZONS
            },
        },
    }

    print(f"[5/5] Writing {output_path}...")
    output = {
        "FAST_MODE": FAST_MODE,
        "eval_split": split,
        "n_features": len(data["feature_cols"]),
        "feature_cols": data["feature_cols"],
        "target_col_idx": int(target_col_idx),
        "eval_horizons": EVAL_HORIZONS,
        "gbm_horizons": GBM_HORIZONS,
        "fsdh_max": FSDH_MAX,
        "qrc_n_qubits_reference": QRC_N_QUBITS,
        "esn_dim_matched_nodes": esn_sizes.get("esn_dim_matched"),
        "null_krr_train_cap": KRR_MAX_TRAIN,
        "n_eval_samples": int(len(y_eval)),
        "metrics": metrics,
        "ceiling_probes": ceiling_probes,
        "fsdh_curves": fsdh_curves,
        "vpt_curves": vpt_curves,
        "skill_curves": skill_curves,
        "wall_clock_s": wall_clock,
        "misc_log": misc_log,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"  Wrote {out_path}")

    for name in metrics:
        s1 = metrics[name]["1"]["skill"] * 100 if "1" in metrics[name] else None
        print(f"  {name:<20} skill@1h={s1:.1f}%" if s1 is not None else f"  {name}")
    for h in GBM_HORIZONS:
        s = ceiling_probes["gbm"]["metrics"][str(h)]["skill"] * 100
        print(f"  gbm_ceiling_probe   skill@{h}h={s:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
