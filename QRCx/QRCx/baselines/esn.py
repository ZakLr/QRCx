from itertools import product
from typing import Optional

import numpy as np

# Sprint 3 tuning grid (selected on validation split only, never test).
SPECTRAL_RADIUS_GRID = [0.7, 0.9, 0.99, 1.1]
LEAK_RATE_GRID = [0.1, 0.3, 0.6, 1.0]
INPUT_SCALING_GRID = [0.1, 0.5, 1.0]
RIDGE_ALPHA_GRID = list(np.logspace(-8, -1, 8))
WASHOUT = 100

# FAST_MODE trims the grid for development/testing; paper-grade runs must
# use FAST_MODE=False (full grid above). Every experiment script accepts
# this flag per project convention.
FAST_SPECTRAL_RADIUS_GRID = [0.9]
FAST_LEAK_RATE_GRID = [0.3, 1.0]
FAST_INPUT_SCALING_GRID = [0.5]
FAST_RIDGE_ALPHA_GRID = [1e-4, 1e-1]


def prepare_sequential_targets(
    seq: np.ndarray, target_col_idx: int, horizons: list, residual: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Align a raw (non-windowed) hourly sequence into ESN input/target pairs.

    Diagnosis (see docs/evaluation_protocol.md): the pre-Sprint-3 ESN fed
    reservoirpy each 24h QRC-style window flattened into a single 312-dim
    "timestep" vector, one per sample. Consecutive samples share 23/24
    hours of overlap, so the sequence reservoirpy actually saw was a highly
    redundant, discontinuous re-encoding of the true hourly series — not a
    genuine time series an ESN's recurrent memory can exploit. This is a
    target/input-alignment bug, not a hyperparameter problem: no amount of
    tuning sr/lr/input_scaling fixes feeding the wrong sequence.

    This function instead builds the *correct* alignment: row i of the
    input is the true feature vector at hour i, and target y[i, h_idx] is
    the true value `horizons[h_idx]` hours later — exactly what
    reservoirpy's own recurrent state (accumulated over genuine
    consecutive hours, not repeated windows) is designed to predict from.

    Args:
        seq: (n_hours, n_features) chronological scaled-anomaly sequence.
        target_col_idx: column index of the forecast target.
        horizons: list of forecast horizons (hours ahead).
        residual: if True, targets are y[i+h] - y[i] (the same "anomaly
            residual" the QRC's ResidualQRC architecture predicts — see
            architecture/residual.py) instead of the raw future anomaly.
            Used by the Residual-ESN fairness baseline: predictions must
            then have the persistence term (seq[i, target_col_idx]) added
            back — see ESNBaseline(residual=True).predict().

    Returns:
        (X, y): X shape (n_hours - max(horizons), n_features),
                y shape (n_hours - max(horizons), len(horizons)).
    """
    max_h = max(horizons)
    n = seq.shape[0] - max_h
    assert n > 0, f"Sequence too short ({seq.shape[0]}) for max horizon {max_h}"
    X = np.nan_to_num(seq[:n])
    y = np.column_stack([seq[h:h + n, target_col_idx] for h in horizons])
    if residual:
        y = y - seq[:n, target_col_idx][:, np.newaxis]
    y = np.nan_to_num(y)
    return X, y


def _fit_esn(X_seq, y_seq, nodes, sr, lr, input_scaling, ridge, seed, washout, rc_connectivity=0.1):
    from reservoirpy.nodes import Reservoir, Ridge
    reservoir = Reservoir(
        units=nodes, sr=sr, lr=lr, seed=seed,
        input_scaling=input_scaling, rc_connectivity=rc_connectivity,
    )
    readout = Ridge(ridge=ridge)
    return (reservoir >> readout).fit(X_seq, y_seq, warmup=washout)


def _run_esn(model, X_seq):
    # Reset state before each call: each split is a non-contiguous calendar
    # period (different years for train/val/test, per the strict temporal
    # split), so state must not leak across splits.
    model.reset()
    preds = model.run(X_seq)
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]
    return preds


def tune_esn(
    train_seq: np.ndarray, val_seq: np.ndarray, target_col_idx: int, horizons: list,
    nodes: int = 500, seed: int = 42, washout: int = WASHOUT,
    fast_mode: bool = False, verbose: bool = True, residual: bool = False,
) -> dict:
    """Grid-search ESN hyperparameters, selected on the validation split only.

    Returns:
        dict with best hyperparameters, validation RMSE, and the grid used.
    """
    sr_grid = FAST_SPECTRAL_RADIUS_GRID if fast_mode else SPECTRAL_RADIUS_GRID
    lr_grid = FAST_LEAK_RATE_GRID if fast_mode else LEAK_RATE_GRID
    scaling_grid = FAST_INPUT_SCALING_GRID if fast_mode else INPUT_SCALING_GRID
    ridge_grid = FAST_RIDGE_ALPHA_GRID if fast_mode else RIDGE_ALPHA_GRID

    X_train, y_train = prepare_sequential_targets(train_seq, target_col_idx, horizons, residual=residual)
    X_val, y_val = prepare_sequential_targets(val_seq, target_col_idx, horizons, residual=residual)

    best_params = None
    best_val_rmse = np.inf
    n_tried = 0
    n_failed = 0

    for sr, lr, scaling, ridge in product(sr_grid, lr_grid, scaling_grid, ridge_grid):
        n_tried += 1
        try:
            model = _fit_esn(X_train, y_train, nodes, sr, lr, scaling, ridge, seed, washout)
            pred_val = _run_esn(model, X_val)
            if not np.isfinite(pred_val).all():
                n_failed += 1
                continue
            val_rmse = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
        except Exception:
            n_failed += 1
            continue
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = {"sr": sr, "leak_rate": lr, "input_scaling": scaling, "ridge": ridge}

    assert best_params is not None, f"ESN tuning failed for all {n_tried} grid points"
    if verbose:
        print(f"ESN-{nodes}: tuned {best_params}, val_RMSE={best_val_rmse:.4f} "
              f"({n_tried - n_failed}/{n_tried} configs succeeded)")

    return {
        "best_params": best_params,
        "val_rmse": best_val_rmse,
        "n_tried": n_tried,
        "n_failed": n_failed,
        "grid": {
            "spectral_radius": sr_grid, "leak_rate": lr_grid,
            "input_scaling": scaling_grid, "ridge_alpha": ridge_grid,
        },
    }


class ESNBaseline:
    """Echo State Network via reservoirpy, fed the genuine hourly sequence
    (not flattened QRC-style windows — see prepare_sequential_targets) and
    tuned on the validation split.

    Hyperparameters (spectral_radius, leak_rate, input_scaling, ridge alpha)
    are selected by grid search against val_seq — never against test — per
    the Sprint 3 fairness protocol.
    """

    def __init__(self, nodes: int = 500, seed: int = 42, washout: int = WASHOUT,
                 fast_mode: bool = False, rc_connectivity: float = 0.1, residual: bool = False):
        self.nodes = nodes
        self.seed = seed
        self.washout = washout
        self.fast_mode = fast_mode
        self.rc_connectivity = rc_connectivity
        self.residual = residual
        self.model = None
        self.tuning_result: Optional[dict] = None
        self.horizons: list = [1]
        self.target_col_idx = 0

    def fit(self, train_seq: np.ndarray, target_col_idx: int, horizons: list,
            val_seq: Optional[np.ndarray] = None):
        self.horizons = list(horizons)
        self.target_col_idx = target_col_idx

        if val_seq is not None:
            self.tuning_result = tune_esn(
                train_seq, val_seq, target_col_idx, horizons,
                nodes=self.nodes, seed=self.seed, washout=self.washout,
                fast_mode=self.fast_mode, residual=self.residual,
            )
            params = self.tuning_result["best_params"]
            sr, lr, scaling, ridge = params["sr"], params["leak_rate"], params["input_scaling"], params["ridge"]
        else:
            # No validation split provided: fall back to a mid-grid config
            # rather than an untuned guess, but this is not the fairness-
            # protocol path (see class docstring).
            sr, lr, scaling, ridge = 0.9, 0.3, 0.5, 1e-4

        X_train, y_train = prepare_sequential_targets(train_seq, target_col_idx, horizons, residual=self.residual)
        self.model = _fit_esn(X_train, y_train, self.nodes, sr, lr, scaling, ridge,
                               self.seed, self.washout, self.rc_connectivity)
        return self

    def predict(self, seq: np.ndarray) -> np.ndarray:
        """Predict at every valid position of `seq` (a raw hourly sequence
        from the same split family as fit, e.g. test_seq). If residual=True,
        the persistence term is added back so the return value is always
        in the same (raw anomaly) target space as residual=False."""
        if self.model is None:
            raise RuntimeError("ESNBaseline not fitted")
        X, _ = prepare_sequential_targets(seq, self.target_col_idx, self.horizons, residual=self.residual)
        pred = _run_esn(self.model, X)
        if self.residual:
            n = X.shape[0]
            persist = seq[:n, self.target_col_idx][:, np.newaxis]
            pred = pred + persist
        return pred


def forecast(
    train_seq: np.ndarray, test_seq: np.ndarray, target_col_idx: int, horizons: list,
    reservoir_size: int = 500, val_seq: Optional[np.ndarray] = None,
    seed: int = 42, fast_mode: bool = False,
    valid_idx: Optional[np.ndarray] = None, W: int = 24, residual: bool = False,
) -> dict:
    """Fit a tuned ESN on the raw hourly sequence and forecast at each horizon.

    If val_seq is given, hyperparameters are grid-searched on it (the
    fairness-protocol path). Otherwise falls back to a fixed config.

    `valid_idx` (from preprocess()'s "test_valid_idx") is the set of window
    start positions the windowed pipeline actually kept after dropping
    NaN-containing windows — roughly a third of windows are dropped on real
    ISD data (missing SLP/WD readings), so a plain positional slice would
    silently compare mismatched target sets. Window start i's target is
    seq[i + W - 1 + h], and this raw-sequence model's prediction at
    position (i + W - 1) targets that same value, so
    `predict(test_seq)[valid_idx + W - 1]` reproduces exactly
    data["y_test"]'s sample set for a fair, identical-target comparison.
    If valid_idx is None, no filtering is applied (only correct when the
    caller's data has no dropped windows).
    """
    model = ESNBaseline(nodes=reservoir_size, seed=seed, fast_mode=fast_mode, residual=residual)
    model.fit(train_seq, target_col_idx, horizons, val_seq=val_seq)
    pred = model.predict(test_seq)
    aligned = pred if valid_idx is None else pred[valid_idx + W - 1]
    result = {}
    for h_idx, h in enumerate(horizons):
        result[h] = aligned[:, h_idx]
    return result
