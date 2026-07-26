#!/usr/bin/env python3
"""FINAL SPRINT Phase 4: recover the target column's StandardScaler std
(from `preprocess()`'s climatological-anomaly + StandardScaler pipeline)
on the canonical split, so headline RMSE/skill numbers (computed in
scaled-anomaly units) can be reported in physical units (deg C) too.

This std is a single global scalar (StandardScaler fit once on the
whole training set's anomaly residual), not time-varying -- so
RMSE_degC = RMSE_scaled * std exactly: the climatological-normal
subtraction is a per-timestamp additive shift that's identical for
y_true and any y_pred expressed in the same anomaly baseline, so it
cancels out of (y_true - y_pred) and only the StandardScaler's
multiplicative std remains. No approximation needed."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

from QRCx.data.loader import load_isd_range, validate_dataframe
from QRCx.data.preprocessor import preprocess

OUT_PATH = REPO_ROOT / "results" / "canonical_units.json"


def main():
    df = load_isd_range(2019, 2024, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)
    data = preprocess(df, train_years=(2019, 2022), val_year=2023, test_year=2024,
                       horizons=[1, 3, 6, 12, 24, 48])
    t_idx = int(data["target_col_idx"])
    std_target = float(data["std"][t_idx])
    mean_target = float(data["mean"][t_idx])
    out = {
        "target_col": data["feature_cols"][t_idx],
        "target_col_idx": t_idx,
        "std_scaled_to_anomaly": std_target,
        "mean_anomaly_to_scaled": mean_target,
        "note": "RMSE_degC = RMSE_scaled * std_scaled_to_anomaly (exact, since the "
                "climatological-normal subtraction cancels out of (y_true - y_pred) "
                "and only the StandardScaler's global multiplicative std remains).",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
