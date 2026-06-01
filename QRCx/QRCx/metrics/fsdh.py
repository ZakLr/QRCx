"""Forecast Skill Duration Horizon (FSDH) — Tier 1.

FSDH answers: "for how many hours is my model actually useful?"
Returns an integer number of hours, not a boolean.
"""
from typing import Optional

import numpy as np

from .forecast import rmse


def fsdh(y_true: np.ndarray, y_pred_model: np.ndarray,
         y_pred_persist: np.ndarray) -> int:
    """Returns 1 if model beats persistence at this horizon, else 0.

    Args:
        y_true: Ground truth values, shape (n,).
        y_pred_model: Model predictions, shape (n,).
        y_pred_persist: Persistence predictions, shape (n,).

    Returns:
        1 if RMSE_model < RMSE_persist, else 0.
    """
    assert y_true.ndim == 1
    assert y_pred_model.ndim == 1
    assert y_pred_persist.ndim == 1
    assert len(y_true) == len(y_pred_model) == len(y_pred_persist)

    model_beats = rmse(y_true, y_pred_model) < rmse(y_true, y_pred_persist)
    return int(model_beats)


def compute_fsdh_curve(
    y_true_horizons: np.ndarray,
    y_model_horizons: np.ndarray,
    y_persist_horizons: np.ndarray,
    max_horizon: Optional[int] = None,
) -> int:
    """Compute FSDH: max consecutive horizon where model beats persistence.
    Stops at first hour where model RMSE >= persistence RMSE.

    Args:
        y_true_horizons: Ground truth, shape (n_samples, max_horizon).
        y_model_horizons: Model predictions, shape (n_samples, max_horizon).
        y_persist_horizons: Persistence predictions, shape (n_samples, max_horizon).
        max_horizon: Optional cap — only evaluate up to this hour.
                     If None, uses all horizons in the input arrays.

    Returns:
        Integer hours — max h where RMSE_model(h) < RMSE_persist(h).
        Returns 0 if model never beats persistence.
    """
    assert y_true_horizons.ndim == 2
    assert y_model_horizons.ndim == 2
    assert y_persist_horizons.ndim == 2
    assert y_true_horizons.shape == y_model_horizons.shape == y_persist_horizons.shape

    max_h = y_true_horizons.shape[1]
    if max_horizon is not None:
        max_h = min(max_h, max_horizon)
    fsdh_val = 0
    for h in range(max_h):
        if fsdh(y_true_horizons[:, h], y_model_horizons[:, h],
                y_persist_horizons[:, h]):
            fsdh_val = h + 1
        else:
            break
    return fsdh_val
