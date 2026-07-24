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
