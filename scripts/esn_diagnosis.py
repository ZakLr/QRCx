#!/usr/bin/env python3
"""Sprint 3: standalone ESN diagnosis run, written to
results/esn_diagnosis.json — the DoD-verifying evidence that the tuned ESN
(full 384-point grid, see baselines/esn.py::tune_esn) achieves skill >= -5%
vs. persistence at 1h, and that the pre-Sprint-3 broken version
(window-flattened input, no leak-rate tuning) does not.

Three stages, all on the same real 2019-2024 KORD data
(QRCx/data/preprocessor.py, 13 features, strict temporal split):
  1. "broken": the pre-Sprint-3 window-flattened ESN-500 (frozen replica
     of the old baselines/esn.py logic, not re-imported since that file
     was fixed in place -- see docs/evaluation_protocol.md Section 2).
  2. "fixed_fast_grid": corrected raw-sequence alignment, fast (4-point)
     tuning grid.
  3. "fixed_full_grid": corrected raw-sequence alignment, full 384-point
     grid (spectral_radius x leak_rate x input_scaling x ridge_alpha) --
     this is the DoD-verifying stage, with DM test + bootstrap CI.

Scored on val-2023 by default (Sprint 3 closeout addendum: all
development decisions and DoD/gate checks are made on val, never test;
test-2024 is reserved for a one-time final confirmatory report -- see
docs/evaluation_protocol.md's split ledger and QRCx.data.split_guard).
--split test requires an explicit unlock and is logged in the ledger.

See docs/evaluation_protocol.md Section 2 for the narrative.
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
from QRCx.baselines import esn, persistence
from QRCx.metrics.forecast import rmse, skill_score
from QRCx.metrics.significance import diebold_mariano, skill_difference_ci

EVAL_HORIZONS = [1, 6]
ESN_NODES = 200


def broken_esn_forecast(X_train, y_train, X_eval, reservoir_size):
    """Frozen replica of the pre-Sprint-3 baselines/esn.py: flattens each
    24x13 window into one reservoirpy "timestep" (the diagnosed bug), sr=0.9,
    input_scaling=0.5, ridge=1.0, no leak rate set (defaults to 1.0), no
    validation-split tuning."""
    from reservoirpy.nodes import Reservoir, Ridge
    reservoir = Reservoir(units=reservoir_size, sr=0.9, seed=42, input_scaling=0.5, rc_connectivity=0.1)
    readout = Ridge(ridge=1.0)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    model = (reservoir >> readout).fit(X_train_flat, y_train, warmup=100)
    X_eval_flat = X_eval.reshape(X_eval.shape[0], -1)
    pred = model.run(X_eval_flat)
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    result = {}
    for h_idx, h in enumerate(EVAL_HORIZONS):
        result[h] = pred[:, h_idx] if pred.shape[1] > h_idx else pred[:, 0]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    split = args.split

    if split == "test":
        assert_test_unlocked(reason="scripts/esn_diagnosis.py --split test")

    print("Loading ISD data (2019-2024, cached under data/isd/)...")
    df = load_isd_range(2019, 2024, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)
    data = preprocess(df, horizons=EVAL_HORIZONS)
    target_col_idx = data["target_col_idx"]

    X_eval = data[f"X_{split}"]
    y_eval = data[f"y_{split}"]
    eval_seq = data[f"{split}_seq"]
    eval_valid_idx = data[f"{split}_valid_idx"]

    pers = persistence.forecast(X_eval, y_eval[:, 0], horizons=EVAL_HORIZONS, target_col_idx=target_col_idx)

    stages = {}

    print(f"[1/3] broken (pre-Sprint-3 window-flattened ESN), scored on {split}...")
    t0 = time.time()
    broken_pred = broken_esn_forecast(data["X_train"], data["y_train"], X_eval, ESN_NODES)
    stages["broken"] = {
        "wall_clock_s": time.time() - t0,
        "skill": {str(h): skill_score(y_eval[:, EVAL_HORIZONS.index(h)], broken_pred[h], pers[h]) for h in EVAL_HORIZONS},
        "rmse": {str(h): rmse(y_eval[:, EVAL_HORIZONS.index(h)], broken_pred[h]) for h in EVAL_HORIZONS},
    }
    print(f"  skill: {stages['broken']['skill']}")

    print(f"[2/3] fixed, fast grid (4 combos), scored on {split}...")
    t0 = time.time()
    fixed_fast = esn.forecast(data["train_seq"], eval_seq, target_col_idx, EVAL_HORIZONS,
                               reservoir_size=ESN_NODES, val_seq=data["val_seq"], fast_mode=True,
                               valid_idx=eval_valid_idx)
    stages["fixed_fast_grid"] = {
        "wall_clock_s": time.time() - t0,
        "skill": {str(h): skill_score(y_eval[:, EVAL_HORIZONS.index(h)], fixed_fast[h], pers[h]) for h in EVAL_HORIZONS},
        "rmse": {str(h): rmse(y_eval[:, EVAL_HORIZONS.index(h)], fixed_fast[h]) for h in EVAL_HORIZONS},
    }
    print(f"  skill: {stages['fixed_fast_grid']['skill']}")

    print(f"[3/3] fixed, full 384-point grid, scored on {split} (this is the DoD-verifying run)...")
    t0 = time.time()
    fixed_full = esn.forecast(data["train_seq"], eval_seq, target_col_idx, EVAL_HORIZONS,
                               reservoir_size=ESN_NODES, val_seq=data["val_seq"], fast_mode=False,
                               valid_idx=eval_valid_idx)
    full_grid_skill = {str(h): skill_score(y_eval[:, EVAL_HORIZONS.index(h)], fixed_full[h], pers[h]) for h in EVAL_HORIZONS}
    full_grid_rmse = {str(h): rmse(y_eval[:, EVAL_HORIZONS.index(h)], fixed_full[h]) for h in EVAL_HORIZONS}

    significance = {}
    for h in EVAL_HORIZONS:
        yt = y_eval[:, EVAL_HORIZONS.index(h)]
        dm = diebold_mariano(yt, fixed_full[h], pers[h], h=h)
        ci = skill_difference_ci(yt, fixed_full[h], pers[h], pers[h], n_boot=2000, seed=42)
        significance[str(h)] = {
            "dm_stat": dm["dm_stat"], "dm_p_value": dm["p_value"], "dm_maxlags": dm["maxlags"],
            "skill_diff_ci_lower": ci["lower"], "skill_diff_ci_upper": ci["upper"],
            "skill_diff_point_estimate": ci["point_estimate"], "n_boot": ci["n_boot"],
        }

    stages["fixed_full_grid"] = {
        "wall_clock_s": time.time() - t0,
        "skill": full_grid_skill,
        "rmse": full_grid_rmse,
        "significance_vs_persistence": significance,
    }
    print(f"  skill: {stages['fixed_full_grid']['skill']}")
    print(f"  significance: {significance}")

    dod_pass = stages["fixed_full_grid"]["skill"]["1"] >= -0.05
    output = {
        "eval_split": split,
        "esn_nodes": ESN_NODES,
        "eval_horizons": EVAL_HORIZONS,
        "dod_threshold_skill_1h": -0.05,
        "dod_pass": bool(dod_pass),
        "stages": stages,
    }

    out_path = REPO_ROOT / "results" / f"esn_diagnosis_{split}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"DoD (skill >= -5% @ 1h, full grid, {split}): {'PASS' if dod_pass else 'FAIL'} "
          f"({stages['fixed_full_grid']['skill']['1']*100:.1f}%)")
    return 0 if dod_pass else 1


if __name__ == "__main__":
    sys.exit(main())
