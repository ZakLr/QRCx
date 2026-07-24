"""Sprint 3 fairness-protocol baselines.

Every baseline here uses the identical 13 features, identical strict
temporal splits, and identical StandardScaler as the QRC (see
data/preprocessor.py). Two additional protocol pieces live here:

  - Null control: Ridge/KRR fit directly on the flattened raw 24x13
    window, no reservoir at all. If this matches the QRC's skill, the
    reservoir buys nothing and that must be reported, not hidden.
  - Residual-Ridge: predicts the same anomaly-residual target
    (y_anom[t+h] - y_anom[t]) that architecture/residual.py's ResidualQRC
    predicts, using the flattened raw window as features instead of
    reservoir features. The residual trick is classical (it's just
    predicting a difference); the QRC must beat this WITH the trick
    applied to both sides, not because the baseline lacks it.

Dimension-matched ESN (ESN size = QRC feature count) is a config choice,
not new code — pass `reservoir_size=qrc.n_features` (or
`qrc.n_features * multiplexing` for the time-multiplexed sequential
reservoir) to baselines.esn.forecast().
"""
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..readout.ridge import RidgeReadout
from ..readout.krr import KRRReadout


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def null_control_forecast(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, horizons: list,
    method: str = "ridge",
) -> dict:
    """Ridge or KRR fit directly on the flattened raw window — no reservoir.

    Args:
        X_train/X_val/X_test: windowed feature arrays, shape (n, W, n_feat).
        y_train/y_val: windowed targets, shape (n, len(horizons)).
        horizons: forecast horizons; must match y_train/y_val's column order.
        method: "ridge" or "krr".

    Returns:
        dict horizon -> predictions, shape (n_test,).
    """
    F_train, F_val, F_test = _flatten(X_train), _flatten(X_val), _flatten(X_test)
    readout_cls = RidgeReadout if method == "ridge" else KRRReadout
    result = {}
    for h_idx, h in enumerate(horizons):
        y_t = y_train[:, h_idx:h_idx + 1] if method == "ridge" else y_train[:, h_idx]
        y_v = y_val[:, h_idx:h_idx + 1] if method == "ridge" else y_val[:, h_idx]
        readout = readout_cls() if method == "krr" else RidgeReadout()
        readout.fit(F_train, y_t, F_val, y_v)
        pred = readout.predict(F_test)
        result[h] = pred.ravel() if pred.ndim > 1 else pred
    return result


def residual_ridge_forecast(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, horizons: list,
    target_col_idx: int = 0, method: str = "ridge",
) -> dict:
    """Ridge/KRR on the flattened raw window, predicting the same
    persistence-anomaly residual (y - persist) the QRC's ResidualQRC
    architecture predicts (architecture/residual.py), then adding
    persistence back — the identical residual decomposition, classical
    features.
    """
    persist_train = X_train[:, -1, target_col_idx][:, np.newaxis]
    persist_val = X_val[:, -1, target_col_idx][:, np.newaxis]
    persist_test = X_test[:, -1, target_col_idx]

    res_train = y_train - persist_train
    res_val = y_val - persist_val

    res_preds = null_control_forecast(X_train, res_train, X_val, res_val, X_test, horizons, method=method)
    return {h: res_preds[h] + persist_test for h in horizons}


def gbm_ceiling_probe_forecast(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_eval: np.ndarray, horizons: list,
    max_iter_grid: tuple = (50, 100, 200),
    max_depth_grid: tuple = (3, 5, None),
    seed: int = 42,
) -> tuple:
    """HistGradientBoostingRegressor on the flattened raw 24x13 window —
    a predictability-ceiling probe, NOT a fairness-protocol RC baseline.

    GBMs cheaply fit nonlinear feature interactions the linear null
    control (Ridge) can't, and answer a different question than "does the
    reservoir help": "how much headroom exists in these 13 features above
    what ESN/Ridge/KRR extract?" A small (3x3) grid over
    `max_iter`/`max_depth` is tuned on val per horizon; not part of the
    RC-vs-classical comparison table.

    Returns:
        (predictions, tuning_log): predictions is {horizon: array},
        tuning_log is {horizon: {"best_params": ..., "val_rmse": ...}}.
    """
    F_train, F_val, F_eval = _flatten(X_train), _flatten(X_val), _flatten(X_eval)
    predictions = {}
    tuning_log = {}
    for h_idx, h in enumerate(horizons):
        y_t, y_v = y_train[:, h_idx], y_val[:, h_idx]
        best_rmse, best_model, best_params = np.inf, None, None
        for max_iter in max_iter_grid:
            for max_depth in max_depth_grid:
                model = HistGradientBoostingRegressor(
                    max_iter=max_iter, max_depth=max_depth, random_state=seed,
                )
                model.fit(F_train, y_t)
                pred_val = model.predict(F_val)
                val_rmse = float(np.sqrt(np.mean((y_v - pred_val) ** 2)))
                if val_rmse < best_rmse:
                    best_rmse, best_model = val_rmse, model
                    best_params = {"max_iter": max_iter, "max_depth": max_depth}
        predictions[h] = best_model.predict(F_eval)
        tuning_log[h] = {"best_params": best_params, "val_rmse": best_rmse}
    return predictions, tuning_log


def dimension_matched_esn_size(reservoir, multiplexing: int = 1) -> int:
    """QRC feature-count-matched ESN reservoir size.

    `reservoir` is any object exposing `n_features` (e.g. AtmosphericQRC:
    3*n_qubits + 3*n_qubits*(n_qubits-1)/2). `multiplexing` accounts for
    time-multiplexed readout (V) on the sequential reservoir, e.g. V=4 at
    12 qubits -> 234 * 4 = 936 (the ESN-936 example in the sprint spec).
    """
    return int(reservoir.n_features) * multiplexing
