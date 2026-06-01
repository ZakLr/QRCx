from typing import Optional

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape
    return float(np.mean(np.abs(y_true - y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape
    std = np.std(y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if std == 0:
        return 0.0 if rmse == 0 else float('inf')
    return float(rmse / std)


def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_clim: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape
    assert y_true.shape == y_clim.shape
    mse_fc = np.mean((y_true - y_pred) ** 2)
    mse_clim = np.mean((y_true - y_clim) ** 2)
    if mse_clim == 0:
        return 0.0
    return float(1.0 - mse_fc / mse_clim)


def vpt(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.4) -> float:
    assert y_true.shape == y_pred.shape
    nrmse_val = nrmse(y_true, y_pred)
    if nrmse_val < threshold:
        return 1.0
    return 0.0


def compute_vpt_curve(
    y_true_horizons: np.ndarray,
    y_pred_horizons: np.ndarray,
    threshold: float = 0.4,
    max_horizon: Optional[int] = None,
) -> int:
    """Max consecutive horizon where NRMSE(h) < threshold.
    Stops at first hour where NRMSE >= threshold.

    Args:
        y_true_horizons: Ground truth, shape (n_samples, max_horizon).
        y_pred_horizons: Model predictions, shape (n_samples, max_horizon).
        threshold: NRMSE threshold (default 0.4 for Lorenz-63 VPT).
        max_horizon: Optional cap — only evaluate up to this hour.
                     If None, uses all horizons in the input arrays.

    Returns:
        Integer — max h where NRMSE(h) < threshold.
    """
    assert y_true_horizons.ndim == 2
    assert y_pred_horizons.ndim == 2
    assert y_true_horizons.shape == y_pred_horizons.shape

    max_h = y_true_horizons.shape[1]
    if max_horizon is not None:
        max_h = min(max_h, max_horizon)
    for h in range(max_h):
        if nrmse(y_true_horizons[:, h], y_pred_horizons[:, h]) >= threshold:
            return h
    return max_h
