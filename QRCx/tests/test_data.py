import pytest
import numpy as np
import pandas as pd
from QRCx.data.preprocessor import preprocess, quality_control, build_features
from QRCx.data.loader import validate_dataframe


def test_quality_control_bounds():
    df = pd.DataFrame({
        "T_db": [100.0, 20.0],
        "T_dew": [0, 0],
        "SLP": [1000, 1000],
        "WS": [0, 0],
        "WD": [0, 0],
        "RH": [50, 50],
    })
    df = quality_control(df)
    assert pd.isna(df.loc[0, "T_db"])


def test_preprocess_shapes():
    dates = pd.date_range("2019-01-01", "2024-01-15", freq="h")
    df = pd.DataFrame(
        index=dates,
        data={
            "T_db": np.random.randn(len(dates)),
            "T_dew": np.random.randn(len(dates)),
            "SLP": 1000 + np.random.randn(len(dates)),
            "WS": np.abs(np.random.randn(len(dates))),
            "WD": np.random.rand(len(dates)) * 360,
            "RH": np.random.rand(len(dates)) * 100,
        },
    )
    result = preprocess(df)
    assert result["X_train"].shape[1:] == (24, 13)
    assert result["y_train"].shape[1] == 2
    assert "target_col_idx" in result


def test_preprocess_accepts_multi_year_val_test_ranges():
    """Sprint 6: val_year/test_year now also accept (start, end) tuples
    (same convention as train_years), not just a single int, for the
    full-record split (train 2011-2020, val 2021-2022, test 2023-2024)."""
    dates = pd.date_range("2011-01-01", "2024-12-31", freq="h")
    df = pd.DataFrame(
        index=dates,
        data={
            "T_db": np.random.randn(len(dates)),
            "T_dew": np.random.randn(len(dates)),
            "SLP": 1000 + np.random.randn(len(dates)),
            "WS": np.abs(np.random.randn(len(dates))),
            "WD": np.random.rand(len(dates)) * 360,
            "RH": np.random.rand(len(dates)) * 100,
        },
    )
    result = preprocess(df, train_years=(2011, 2020), val_year=(2021, 2022), test_year=(2023, 2024))
    assert result["X_train"].shape[1:] == (24, 13)
    assert len(result["X_val"]) > 0 and len(result["X_test"]) > 0
    # single-int val_year/test_year (original convention) must still work unchanged
    result2 = preprocess(df, train_years=(2011, 2020), val_year=2021, test_year=2022)
    assert len(result2["X_val"]) > 0 and len(result2["X_test"]) > 0
