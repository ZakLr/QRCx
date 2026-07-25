import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .splits import DataSplit, temporal_split


def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[~df["T_db"].between(-60, 50), "T_db"] = np.nan
    df.loc[~df["T_dew"].between(-70, 40), "T_dew"] = np.nan
    df.loc[~df["SLP"].between(870, 1084), "SLP"] = np.nan
    df.loc[df["WS"] < 0, "WS"] = np.nan
    df.loc[~df["WD"].between(0, 360), "WD"] = np.nan
    df.loc[~df["RH"].between(0, 100), "RH"] = np.nan
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the 13 physical features (single source of truth).

    Ports `pipeline_demo.py`'s `engineer_features()` into the canonical
    package, closing the feature-set gap documented in README's known
    limitations. Output columns:
        [T_db, T_dew, RH, WS, SLP, WD, Wx, Wy, T_dep,
         hour_sin, hour_cos, doy_sin, doy_cos]
    Replaces the previous 10-feature set (theta/VPD/u/v) — those derived
    quantities are dropped in favor of this frozen 13-feature list per the
    project's global invariants; not silently merged alongside it.
    """
    assert isinstance(df.index, pd.DatetimeIndex), "build_features requires a DatetimeIndex for cyclical encodings"
    out = df.copy()

    wd_rad = np.radians(out["WD"].fillna(0))
    out["Wx"] = out["WS"] * np.sin(wd_rad)
    out["Wy"] = out["WS"] * np.cos(wd_rad)
    out["T_dep"] = out["T_db"] - out["T_dew"]

    h = out.index.hour.values.astype(float)
    doy = out.index.dayofyear.values.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    cols = ["T_db", "T_dew", "RH", "WS", "SLP", "WD",
            "Wx", "Wy", "T_dep",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    return out[cols]


def compute_climatological_normals(df_train: pd.DataFrame) -> pd.Series:
    return df_train.groupby([df_train.index.month, df_train.index.day, df_train.index.hour]).mean()


def apply_climatological_anomaly(df: pd.DataFrame, normals: pd.Series) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [df.index.month, df.index.day, df.index.hour],
        names=normals.index.names,
    )
    return (df - normals.reindex(idx).to_numpy()).dropna(how="all")


def sliding_windows(
    df: pd.DataFrame,
    W: int = 24,
    horizons: list[int] = [1, 6],
    target_col: str = "T_db",
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    data = df.to_numpy()
    n_features = data.shape[1]
    target_idx = list(df.columns).index(target_col)
    n_samples = data.shape[0] - W - max(horizons) + 1
    if n_samples <= 0:
        raise ValueError("Data too short for given W and horizons")

    X = np.zeros((n_samples, W, n_features), dtype=np.float64)
    y = np.zeros((n_samples, len(horizons)), dtype=np.float64)

    for i in range(n_samples):
        X[i] = data[i : i + W]
        for h, horizon in enumerate(horizons):
            y[i, h] = data[i + W - 1 + horizon, target_idx]

    valid = ~(np.any(np.isnan(X), axis=(1, 2)) | np.any(np.isnan(y), axis=1))
    valid_idx = np.nonzero(valid)[0]
    X = X[valid]
    y = y[valid]
    assert len(X) > 0, "All windows dropped due to NaN"
    assert X.shape == (len(X), W, n_features), f"Expected (n, {W}, {n_features}), got {X.shape}"
    assert y.shape == (len(y), len(horizons)), f"Expected (n, {len(horizons)}), got {y.shape}"
    # valid_idx[i] is the window start position (in the raw non-windowed
    # sequence) of the i-th retained sample — used by baselines that
    # consume the raw sequence directly (e.g. ESN) to align their
    # predictions to exactly the same (non-NaN) samples as the windowed
    # target array, for a fair identical-target comparison.
    return X, y, target_idx, valid_idx


def preprocess(
    df: pd.DataFrame,
    target_col: str = "T_db",
    train_years: tuple[int, int] = (2019, 2022),
    val_year=2023,
    test_year=2024,
    W: int = 24,
    horizons: list[int] = [1, 6],
) -> dict:
    df = quality_control(df)
    df = build_features(df)
    split = temporal_split(df, train_years, val_year, test_year)

    train_normals = compute_climatological_normals(split.train)
    train_anom = apply_climatological_anomaly(split.train, train_normals)
    val_anom = apply_climatological_anomaly(split.val, train_normals)
    test_anom = apply_climatological_anomaly(split.test, train_normals)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_anom)
    val_scaled = scaler.transform(val_anom)
    test_scaled = scaler.transform(test_anom)

    train_df = pd.DataFrame(train_scaled, columns=train_anom.columns, index=train_anom.index)
    val_df = pd.DataFrame(val_scaled, columns=val_anom.columns, index=val_anom.index)
    test_df = pd.DataFrame(test_scaled, columns=test_anom.columns, index=test_anom.index)

    X_train, y_train, t_idx, train_valid_idx = sliding_windows(train_df, W, horizons, target_col)
    X_val, y_val, t_idx2, val_valid_idx = sliding_windows(val_df, W, horizons, target_col)
    X_test, y_test, t_idx3, test_valid_idx = sliding_windows(test_df, W, horizons, target_col)

    assert t_idx == t_idx2 == t_idx3, "target_col_idx mismatch across splits"

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        # Raw (non-windowed) per-split scaled anomaly sequences, in
        # chronological order — for baselines (e.g. ESN) that should
        # consume a genuine hourly sequence rather than the QRC's
        # overlapping 24h windows (see baselines/esn.py docstring: feeding
        # flattened windows as "timesteps" was part of the Sprint 3
        # target-alignment diagnosis).
        "train_seq": train_scaled,
        "val_seq": val_scaled,
        "test_seq": test_scaled,
        # Window start index (into *_seq) of each retained (non-NaN)
        # windowed sample — lets raw-sequence baselines subselect to
        # exactly the same target set as X_*/y_* for a fair comparison.
        "train_valid_idx": train_valid_idx,
        "val_valid_idx": val_valid_idx,
        "test_valid_idx": test_valid_idx,
        "mean": scaler.mean_,
        "std": scaler.scale_,
        "normals": train_normals,
        "target_col_idx": t_idx,
        "feature_cols": list(df.columns),
    }
