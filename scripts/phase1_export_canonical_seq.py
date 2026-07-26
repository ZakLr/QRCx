#!/usr/bin/env python3
"""FINAL SPRINT Phase 1: export the CANONICAL split (train 2019-2022,
val 2023, test 2024 -- NOT the Sprint-4 pilot split train2019-2021/
eval2022, and NOT Sprint 6's full-record 2011-2024 split) so v4/v5 QRC
and every classical baseline are evaluated on IDENTICAL rows -- the
mismatch this final sprint exists specifically to fix (see
QRCx_FINAL_24H_SUPERPROMPT.md Phase 1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

from QRCx.data.loader import load_isd_range, validate_dataframe
from QRCx.data.preprocessor import preprocess

OUT_PATH = REPO_ROOT / "data" / "canonical_seq.npz"


def main():
    print("Loading ISD data (2019-2024, cached under data/isd/)...")
    df = load_isd_range(2019, 2024, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)

    print("Preprocessing with CANONICAL split: train=2019-2022, val=2023, test=2024, "
          "horizons=[1,3,6,12,24,48]...")
    data = preprocess(df, train_years=(2019, 2022), val_year=2023, test_year=2024,
                       horizons=[1, 3, 6, 12, 24, 48])

    train_seq = np.asarray(data["train_seq"], dtype=np.float64)
    val_seq = np.asarray(data["val_seq"], dtype=np.float64)
    test_seq = np.asarray(data["test_seq"], dtype=np.float64)

    for name, seq in [("train_seq", train_seq), ("val_seq", val_seq), ("test_seq", test_seq)]:
        n_nan = int(np.isnan(seq).sum())
        if n_nan:
            print(f"Forward-filling {n_nan} NaN values in {name} (real missing SLP/WD readings)...")

    def ffill(seq):
        if np.isnan(seq).any():
            return pd.DataFrame(seq).ffill().bfill().to_numpy()
        return seq

    train_seq = ffill(train_seq)
    val_seq = ffill(val_seq)
    test_seq = ffill(test_seq)
    assert not np.isnan(train_seq).any() and not np.isnan(val_seq).any() and not np.isnan(test_seq).any()

    target_col_idx = int(data["target_col_idx"])
    feature_cols = list(data["feature_cols"])
    X_train = np.asarray(data["X_train"], dtype=np.float64)
    y_train = np.asarray(data["y_train"], dtype=np.float64)
    X_val = np.asarray(data["X_val"], dtype=np.float64)
    y_val = np.asarray(data["y_val"], dtype=np.float64)
    X_test = np.asarray(data["X_test"], dtype=np.float64)
    y_test = np.asarray(data["y_test"], dtype=np.float64)
    # Window start index (into the corresponding *_seq, PRE-ffill positions
    # are identical to post-ffill positions since ffill doesn't change
    # sequence length) of each retained windowed sample -- needed to align
    # v5's per-timestep recurrent reservoir features (driven over the
    # ffilled train_seq+val_seq+test_seq) to each window's target. Without
    # this, X_train[i]'s NaN-dropped position in the raw sequence is lost.
    train_valid_idx = np.asarray(data["train_valid_idx"], dtype=np.int64)
    val_valid_idx = np.asarray(data["val_valid_idx"], dtype=np.int64)
    test_valid_idx = np.asarray(data["test_valid_idx"], dtype=np.int64)

    print(f"train_seq={train_seq.shape}  val_seq={val_seq.shape}  test_seq={test_seq.shape}  "
          f"target_col_idx={target_col_idx} ({feature_cols[target_col_idx]})")
    print(f"X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        train_seq=train_seq, val_seq=val_seq, test_seq=test_seq,
        target_col_idx=target_col_idx, feature_cols=np.array(feature_cols),
        canonical_train_years=np.array([2019, 2022]),
        canonical_val_year=np.array([2023]), canonical_test_year=np.array([2024]),
        horizons=np.array([1, 3, 6, 12, 24, 48]),
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        train_valid_idx=train_valid_idx, val_valid_idx=val_valid_idx, test_valid_idx=test_valid_idx,
    )
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
