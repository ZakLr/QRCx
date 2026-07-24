from typing import Optional

import numpy as np


class ARIMABaseline:
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.fitted = None
        self.n_horizons = 1

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.n_horizons = y_train.shape[1] if y_train.ndim == 2 else 1
        try:
            from statsmodels.tsa.arima.model import ARIMA
            y = y_train[:, 0] if y_train.ndim > 1 else y_train
            model = ARIMA(y, order=self.order)
            self.fitted = model.fit()
        except Exception as e:
            print(f"[ARIMA] Fit failed: {e}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fitted is None:
            pred = X[:, -1, 0]
        else:
            try:
                forecast = self.fitted.forecast(steps=len(X))
                pred = forecast.values if hasattr(forecast, "values") else np.asarray(forecast)
            except Exception:
                pred = X[:, -1, 0]
        if self.n_horizons > 1:
            return np.column_stack([pred] * self.n_horizons)
        return pred


def select_order_by_aic(
    train_seq: np.ndarray, target_col_idx: int,
    p_grid: tuple = (0, 1, 2, 3), q_grid: tuple = (0, 1, 2, 3), d: int = 0,
) -> dict:
    """AIC-based ARIMA(p, d, q) order selection on the train split only.

    `d=0` by default (not `d=1`): the anomaly series is already
    climatologically deseasonalized/standardized (see
    docs/evaluation_protocol.md Section 4 — the spec-fixed ARIMA(2,1,2)'s
    `d=1` over-differences this already-stationary series, a real,
    honestly-reported misconfiguration of the fixed-order baseline this
    auto-selected variant avoids by construction, not by re-tuning the
    fixed-order model).
    """
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    y_train_raw = train_seq[:, target_col_idx]
    best_aic, best_order, n_tried, n_failed = np.inf, None, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in p_grid:
            for q in q_grid:
                if p == 0 and q == 0:
                    continue
                n_tried += 1
                try:
                    fitted = ARIMA(y_train_raw, order=(p, d, q)).fit()
                    if fitted.aic < best_aic:
                        best_aic, best_order = fitted.aic, (p, d, q)
                except Exception:
                    n_failed += 1
    assert best_order is not None, "AIC order selection failed for every (p,q) tried"
    return {"order": best_order, "aic": float(best_aic), "n_tried": n_tried, "n_failed": n_failed}


def rolling_origin_forecast(
    train_seq: np.ndarray, eval_seq: np.ndarray, target_col_idx: int, horizons: list,
    order, valid_idx: Optional[np.ndarray] = None, W: int = 24, stride: Optional[int] = None,
) -> dict:
    """Genuine local h-step-ahead ARIMA forecasts, re-anchored to real
    observed history every `stride` steps (default max(horizons)) via
    statsmodels' Kalman-filter state update (`ARIMAResults.apply(...,
    refit=False)` — reuses the fitted parameters, no re-optimization, just
    refilters the state with the newly observed data, so this is cheap
    compared to a full refit).

    Diagnosis (found while building the fair auto-ARIMA line, see
    docs/evaluation_protocol.md Section 4): `forecast()`'s single
    `len(eval_seq) + max(horizons) - 1`-step trajectory from the end of
    train is not just over-differenced (the `d=1` issue) — it is also
    architecturally a single blind long-range extrapolation, never
    incorporating the real val/test data as it becomes available. Any
    stationary ARMA process's h-step forecast decays toward its
    unconditional mean as h grows, so evaluating deep into that one
    trajectory (thousands of steps from train's end) is close to a
    constant-value forecast regardless of order — confirmed empirically:
    even the auto-selected (3,0,2) order scored -4857%/-350% skill at
    h=1/h=6 through the single-trajectory path, near-identical in kind to
    the spec-fixed (2,1,2)'s failure, despite fixing the d=1 issue.

    Each block of `stride` positions shares one h-step-ahead forecast
    anchored at the block's start (not a fresh forecast at every single
    position — a documented, tractable compromise, not per-position
    refiltering) — still drastically more local than one trajectory for
    the entire eval period.
    """
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    max_h = max(horizons)
    if stride is None:
        stride = max_h

    y_train = train_seq[:, target_col_idx]
    y_eval = eval_seq[:, target_col_idx]
    n_eval = len(y_eval)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(y_train, order=order).fit()

        preds = {h: np.full(n_eval, np.nan) for h in horizons}
        origin = 0
        while origin < n_eval:
            end = min(origin + stride, n_eval)
            obs_so_far = np.concatenate([y_train, y_eval[:origin]]) if origin > 0 else y_train
            try:
                current = fitted.apply(obs_so_far, refit=False)
                fc = np.asarray(current.forecast(steps=max_h))
            except Exception:
                fc = np.full(max_h, y_eval[origin - 1] if origin > 0 else y_train[-1])
            for h in horizons:
                preds[h][origin:end] = fc[h - 1]
            origin = end

    result = {}
    for h in horizons:
        aligned = preds[h] if valid_idx is None else preds[h][valid_idx + W - 1]
        result[h] = np.nan_to_num(aligned)
    return result


def auto_order_forecast(
    train_seq: np.ndarray, eval_seq: np.ndarray, target_col_idx: int, horizons: list,
    valid_idx: Optional[np.ndarray] = None, W: int = 24,
    p_grid: tuple = (0, 1, 2, 3), q_grid: tuple = (0, 1, 2, 3), d: int = 0,
    stride: Optional[int] = None,
) -> tuple:
    """Auto-selected ARIMA(p,0,q) (AIC on train), forecast via genuine
    rolling-origin local forecasts (`rolling_origin_forecast`). A second,
    honestly comparable ARIMA line: the spec fixes ARIMA(2,1,2) for the
    primary table entry, but reporting a single-trajectory forecast as
    ARIMA's ceiling would leave a strawman in place of the one just fixed
    for ESN — this is ARIMA given a fair chance at both its own order and
    a fair (local, re-anchored) forecasting procedure.

    Returns:
        (result, selection_log): result is {horizon: predictions},
        selection_log is select_order_by_aic()'s return dict.
    """
    selection = select_order_by_aic(train_seq, target_col_idx, p_grid=p_grid, q_grid=q_grid, d=d)
    result = rolling_origin_forecast(train_seq, eval_seq, target_col_idx, horizons,
                                      selection["order"], valid_idx=valid_idx, W=W, stride=stride)
    return result, selection


def forecast(
    train_seq: np.ndarray, test_seq: np.ndarray, target_col_idx: int, horizons: list,
    valid_idx: Optional[np.ndarray] = None, W: int = 24, order=(2, 1, 2),
) -> dict:
    """ARIMA(p,d,q) fit once on the raw chronological anomaly sequence,
    producing one long multi-step forecast trajectory, sliced per horizon.

    Diagnosis (see docs/evaluation_protocol.md): the pre-Sprint-3 version
    computed a single `n_test`-length forecast trajectory and returned the
    *identical* array for every horizon — h=1 and h=6 predictions were
    bitwise identical, silently wrong for any h > 1. Fixed by generating a
    trajectory long enough to cover the largest horizon
    (`len(test_seq) + max(horizons) - 1` steps ahead of the end of
    training) and slicing `fc[h - 1 : h - 1 + n]` per horizon — the
    standard multi-step ARIMA forecast convention (also used by
    pipeline_demo.py's reference implementation).

    Alignment matches baselines/esn.py: predictions are trimmed to
    `valid_idx + W - 1` so they line up 1:1 with data["y_test"] (the same
    NaN-dropped window set every other baseline is scored against).
    """
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    y_train_raw = train_seq[:, target_col_idx]
    n_test = test_seq.shape[0]
    max_h = max(horizons)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fitted = ARIMA(y_train_raw, order=order).fit()
            fc = fitted.forecast(steps=n_test + max_h - 1)
            fc = np.asarray(fc.values if hasattr(fc, "values") else fc)
        except Exception as e:
            print(f"[ARIMA] Fit/forecast failed: {e}; falling back to persistence")
            fc = None

    result = {}
    for h in horizons:
        if fc is not None:
            pred = fc[h - 1: h - 1 + n_test]
        else:
            pred = np.full(n_test, y_train_raw[-1])
        aligned = pred if valid_idx is None else pred[valid_idx + W - 1]
        result[h] = aligned
    return result
