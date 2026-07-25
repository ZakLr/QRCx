#!/usr/bin/env python3
"""Sprint 4 (REVISED): export the pilot-split raw hourly sequences AND
windowed (X, y) arrays needed by the v5 and v4 GPU handoff scripts
(scripts/sprint4_v5_gpu_handoff.py, scripts/sprint4_v4_gpu_handoff.py).

Why a separate export step: the v5 GPU script is deliberately
self-contained (no QRCx import — same convention as sprint26_pipeline.py
and gpu_verify_standalone.py) so it can be pasted/uploaded directly into
a qBraid Lab GPU notebook with no package install step. Real feature
engineering + climatological-anomaly scaling still has to go through the
actual QRCx.data.preprocessor.preprocess() (single source of truth), so
that step runs locally here and its *output* (plain arrays) is what
travels to the GPU.

Pilot split (Sprint 4 REVISED spec): train 2019-2021, eval 2022. This is
a nested sub-split carved only out of the canonical train block
(2019-2022) -- test_year=2024 is passed through to preprocess() (a
required arg) but train_seq/val_seq below never read X_test/y_test/
test_seq, so the canonical locked test split is never touched and
split_guard is never invoked (see docs/evaluation_protocol.md).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

from QRCx.data.loader import load_isd_range, validate_dataframe
from QRCx.data.preprocessor import preprocess

OUT_PATH = REPO_ROOT / "data" / "sprint4_pilot_seq.npz"


def main():
    print("Loading ISD data (2019-2024, cached under data/isd/)...")
    df = load_isd_range(2019, 2024, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)

    print("Preprocessing with PILOT split: train=2019-2021, val(eval)=2022, "
          "horizons=[1,3,6,12] (Sprint 4 REVISED spec -- test_year=2024 passed "
          "through but never read below)...")
    data = preprocess(df, train_years=(2019, 2021), val_year=2022, test_year=2024,
                       horizons=[1, 3, 6, 12])

    train_seq = np.asarray(data["train_seq"], dtype=np.float64)
    val_seq = np.asarray(data["val_seq"], dtype=np.float64)

    # Real bug found while running v5 on the GPU: train_seq/val_seq (the
    # raw continuous per-hour sequence) still contain NaN at real KORD
    # ISD's missing SLP/WD readings (~1-2% of rows) -- the windowed
    # X_train/X_val below are already NaN-free because sliding_windows()
    # drops any window containing one, but v5 drives train_seq/val_seq
    # CONTINUOUSLY (no windowing), so a single NaN timestep corrupts the
    # recurrent density matrix for every step after it (confirmed on the
    # qBraid H200: crashed 44 minutes into a drive with "Ridge: Input X
    # contains NaN"). Forward-filled per column, per split (never across
    # the train/val boundary, and never using future values -- ffill only
    # propagates the last valid observation forward), with a bfill
    # fallback for the rare case of a NaN in a split's very first row(s).
    # Standard, defensible handling for continuous state-space methods fed
    # real sensor data; NOT applied to X_train/X_val since those never had
    # NaN to begin with.
    n_nan_train = int(np.isnan(train_seq).sum())
    n_nan_val = int(np.isnan(val_seq).sum())
    if n_nan_train or n_nan_val:
        print(f"Forward-filling {n_nan_train} NaN train_seq values and "
              f"{n_nan_val} NaN val_seq values (real missing SLP/WD readings)...")
        train_seq = pd.DataFrame(train_seq).ffill().bfill().to_numpy()
        val_seq = pd.DataFrame(val_seq).ffill().bfill().to_numpy()
        assert not np.isnan(train_seq).any() and not np.isnan(val_seq).any()

    target_col_idx = int(data["target_col_idx"])
    feature_cols = list(data["feature_cols"])
    X_train = np.asarray(data["X_train"], dtype=np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float64)
    X_val = np.asarray(data["X_val"], dtype=np.float64)
    y_val = np.asarray(data["y_val"], dtype=np.float64)

    print(f"train_seq shape={train_seq.shape}  val_seq shape={val_seq.shape}  "
          f"target_col_idx={target_col_idx} ({feature_cols[target_col_idx]})")
    print(f"X_train shape={X_train.shape}  X_val shape={X_val.shape} "
          f"(windowed, for v4's per-window statevector encoding)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        train_seq=train_seq,
        val_seq=val_seq,
        target_col_idx=target_col_idx,
        feature_cols=np.array(feature_cols),
        pilot_train_years=np.array([2019, 2021]),
        pilot_val_year=np.array([2022]),
        horizons=np.array([1, 3, 6, 12]),
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    )
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB) -- "
          f"upload this file alongside scripts/sprint4_v5_gpu_handoff.py to qBraid Lab.")


if __name__ == "__main__":
    main()
