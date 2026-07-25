#!/usr/bin/env python3
"""Sprint 4 (REVISED) -- Pilot Evaluation: classical/v4 baselines.

A SEPARATE, nested pilot split carved only out of the canonical TRAIN
block (2019-2022): pilot-train = 2019-2021, pilot-eval = 2022. This is
NOT the canonical val/test split (val=2023, test=2024) and must never
touch or score against test_year=2024 data -- see
docs/evaluation_protocol.md and QRCx.data.split_guard. This script never
imports/calls split_guard and never reads an "X_test"/"y_test"/"test_seq"
key from preprocess()'s return dict; it only reads *_train/*_val keys.
(preprocess() requires a test_year argument -- we pass test_year=2022,
i.e. deliberately identical to val_year, so temporal_split's internal
"test" slice is just a redundant copy of the 2022 val slice, never the
real 2024 data, and load_isd_range is only asked to load 2019-2022
in the first place: the 2024 files are never touched.)

Horizons for the pilot are {1, 3, 6, 12} (NOT the canonical [1, 6] --
Sprint 4's spec fixes these four).

Baselines run (same fairness protocol as scripts/generate_baselines.py,
adapted to the pilot split/horizons):
  - Persistence (floor)
  - ARIMA(2,1,2) (spec-fixed order)
  - ESN dimension-matched to the 12-qubit QRC (234 units)
  - ESN-500
  - Residual-ESN (dimension-matched size)
  - Null control: Ridge and KRR on the flattened raw 24x13 window
    (KRR's training tail capped -- see KRR_MAX_TRAIN below; it is O(n^2)
    in training samples, a pre-existing limitation of KRRReadout, not
    fixed here, same as generate_baselines.py)
  - Residual-Ridge

For every non-persistence baseline, at each of the 4 pilot horizons:
RMSE/MAE/NRMSE/skill/VPT, a Diebold-Mariano test vs. persistence, and a
moving-block-bootstrap 95% CI for the skill difference vs. persistence.

FAST_MODE (default True) controls only the ESN tuning grid's resolution
(reduced 4-point grid vs. the full grid) -- matches the project-wide
convention in scripts/generate_baselines.py. Both modes run on the real
2019-2022 KORD ISD data (no synthetic/mocked data).
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
from QRCx.baselines import persistence, arima, esn
from QRCx.baselines.fairness import null_control_forecast, residual_ridge_forecast
from QRCx.metrics.forecast import rmse, mae, nrmse, skill_score, vpt
from QRCx.metrics.significance import diebold_mariano, skill_difference_ci

PILOT_TRAIN_YEARS = (2019, 2021)
PILOT_EVAL_YEAR = 2022
EVAL_HORIZONS = [1, 3, 6, 12]
QRC_N_QUBITS = 12  # reference config (global invariant) -- used only to
# compute the dimension-matched ESN size below, independent of whatever
# qubit count the actual v4 QRC pilot run (sprint4_pilot_v4.py) ends up
# using for wall-clock-feasibility reasons.


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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    FAST_MODE = args.fast_mode

    output_path = args.output or str(REPO_ROOT / "results" / "sprint4_pilot_baselines.json")

    print(f"FAST_MODE={FAST_MODE}  eval_split=pilot_2022  "
          f"pilot_train_years={PILOT_TRAIN_YEARS}  eval_horizons={EVAL_HORIZONS}")
    print("[1/5] Loading ISD data (2019-2022 only -- 2023/2024 never requested/touched)...")
    df = load_isd_range(PILOT_TRAIN_YEARS[0], PILOT_EVAL_YEAR, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)

    print("[2/5] Preprocessing (13 features, pilot temporal split "
          f"train={PILOT_TRAIN_YEARS} val=test={PILOT_EVAL_YEAR})...")
    # test_year is deliberately set equal to val_year (2022): preprocess()
    # requires a test_year arg, but this script never reads X_test/y_test/
    # test_seq/test_valid_idx from the returned dict below -- only the
    # *_train/*_val keys are used, so the redundant "test" slice (an exact
    # duplicate of the 2022 val slice) is simply ignored.
    data = preprocess(df, train_years=PILOT_TRAIN_YEARS, val_year=PILOT_EVAL_YEAR,
                       test_year=PILOT_EVAL_YEAR, horizons=EVAL_HORIZONS)
    target_col_idx = data["target_col_idx"]
    print(f"  X_train={data['X_train'].shape}  X_val(pilot-eval)={data['X_val'].shape}")

    X_eval = data["X_val"]
    y_eval = data["y_val"]
    eval_seq = data["val_seq"]
    eval_valid_idx = data["val_valid_idx"]

    dim_matched_nodes = _qrc_n_features(QRC_N_QUBITS)
    if FAST_MODE:
        esn_sizes = {"esn_dim_matched": dim_matched_nodes, "esn_500": 500}
    else:
        esn_sizes = {"esn_dim_matched": dim_matched_nodes, "esn_500": 500}

    print("[3/5] Running baselines (scored on pilot-eval 2022)...")
    all_preds_eval = {}
    wall_clock = {}
    misc_log = {}

    t0 = time.time()
    pers_eval = persistence.forecast(X_eval, y_eval[:, 0], horizons=EVAL_HORIZONS, target_col_idx=target_col_idx)
    all_preds_eval["persistence"] = pers_eval
    wall_clock["persistence"] = time.time() - t0
    print(f"  persistence: {wall_clock['persistence']:.1f}s")

    t0 = time.time()
    arima_curve = arima.forecast(data["train_seq"], eval_seq, target_col_idx, EVAL_HORIZONS,
                                  valid_idx=eval_valid_idx)
    all_preds_eval["arima"] = arima_curve
    wall_clock["arima"] = time.time() - t0
    print(f"  arima(2,1,2) [spec-fixed]: {wall_clock['arima']:.1f}s")

    for esn_name, nodes in esn_sizes.items():
        t0 = time.time()
        esn_curve = esn.forecast(
            data["train_seq"], eval_seq, target_col_idx, EVAL_HORIZONS,
            reservoir_size=nodes, val_seq=data["val_seq"], fast_mode=FAST_MODE,
            valid_idx=eval_valid_idx,
        )
        all_preds_eval[esn_name] = esn_curve
        wall_clock[esn_name] = time.time() - t0
        print(f"  {esn_name} (nodes={nodes}): {wall_clock[esn_name]:.1f}s")

    t0 = time.time()
    res_esn_curve = esn.forecast(
        data["train_seq"], eval_seq, target_col_idx, EVAL_HORIZONS,
        reservoir_size=esn_sizes["esn_dim_matched"], val_seq=data["val_seq"], fast_mode=FAST_MODE,
        valid_idx=eval_valid_idx, residual=True,
    )
    all_preds_eval["residual_esn"] = res_esn_curve
    wall_clock["residual_esn"] = time.time() - t0
    print(f"  residual_esn: {wall_clock['residual_esn']:.1f}s")

    y_train_eval = data["y_train"]
    y_val_eval = data["y_val"]

    t0 = time.time()
    null_ridge = null_control_forecast(data["X_train"], y_train_eval, data["X_val"], y_val_eval, X_eval, EVAL_HORIZONS, method="ridge")
    all_preds_eval["null_ridge"] = null_ridge
    wall_clock["null_ridge"] = time.time() - t0

    # KRR is O(n^2) in training samples -- capped for tractability, same
    # protocol/caveat as scripts/generate_baselines.py (eval set is NOT
    # subsampled, only the KRR fit's training tail).
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

    print("[4/5] Scoring (RMSE/MAE/NRMSE/skill/VPT, DM test, bootstrap CI) on pilot-eval 2022...")
    metrics = {}
    for name, preds_eval in all_preds_eval.items():
        metrics[name] = evaluate_model(name, preds_eval, y_eval, pers_eval)

    print(f"[5/5] Writing {output_path}...")
    output = {
        "FAST_MODE": FAST_MODE,
        "eval_split": "pilot_2022",
        "pilot_train_years": list(PILOT_TRAIN_YEARS),
        "pilot_eval_year": PILOT_EVAL_YEAR,
        "n_features": len(data["feature_cols"]),
        "feature_cols": data["feature_cols"],
        "target_col_idx": int(target_col_idx),
        "eval_horizons": EVAL_HORIZONS,
        "qrc_n_qubits_reference": QRC_N_QUBITS,
        "esn_dim_matched_nodes": esn_sizes.get("esn_dim_matched"),
        "null_krr_train_cap": KRR_MAX_TRAIN,
        "n_train_samples": int(len(data["X_train"])),
        "n_eval_samples": int(len(y_eval)),
        "metrics": metrics,
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
        s12 = metrics[name]["12"]["skill"] * 100 if "12" in metrics[name] else None
        print(f"  {name:<20} skill@1h={s1:.1f}%  skill@12h={s12:.1f}%" if s1 is not None else f"  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
