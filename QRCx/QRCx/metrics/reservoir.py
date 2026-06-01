"""Reservoir capacity metrics — Tier 2.

FIX: IPC and memory capacity must use 24h windows (not single steps)
to match the actual inference setup.
"""
import numpy as np
from sklearn.linear_model import Ridge


def measure_memory_capacity(reservoir, max_lag: int = 20, n_samples: int = 500) -> float:
    """Memory capacity at critical point (Krakoviak et al. formulation).

    For each lag k, trains a Ridge regression to predict u[t+24−k]
    (the input delivered k steps before the end of the 24‑step window)
    from the reservoir state at window t.  MC is the sum of squared
    Pearson correlations over all lags.

    Target: MC > 4 for N=8.
    """
    assert n_samples > max_lag, f"n_samples ({n_samples}) must exceed max_lag ({max_lag})"

    rng = np.random.default_rng(42)
    n_qubits = reservoir.n_qubits

    u = rng.standard_normal((n_samples + max_lag + 24, n_qubits))
    X = np.array([u[i : i + 24] for i in range(n_samples)])

    n_eval = min(n_samples, 500)
    X_eval = X[:n_eval]
    R = reservoir.transform(X_eval)
    n_eff = len(R)

    mc = 0.0
    for tau in range(1, min(max_lag + 1, 25, n_eff)):
        start = 24 - tau
        y_target = u[start : start + n_eff, 0]
        if len(y_target) < n_eff:
            break
        model = Ridge(alpha=1.0).fit(R, y_target)
        y_pred = model.predict(R)

        cov = np.cov(y_target, y_pred)[0, 1]
        var_u = np.var(y_target)
        var_pred = np.var(y_pred)
        if var_u > 1e-12 and var_pred > 1e-12:
            mc += (cov ** 2) / (var_u * var_pred)

    return float(mc)


def measure_ipc_24h(reservoir, X_test: np.ndarray, window_size: int = 24, max_lag: int = 20) -> dict:
    """Information processing capacity with 24h windows.

    For each lag k and degree d (1 = linear, 2 = quadratic), trains a
    Ridge regression to predict (u[t+24−k])^d from the reservoir state.
    IPC is the sum of squared Pearson correlations over all lags×degrees.

    Target: total_ipc ≥ 0.8 × N for N=8.
    """
    n_eff = min(len(X_test), 200, max_lag)
    if n_eff < max_lag:  # need contiguous input
        return {"linear_ipc": 0.0, "nonlinear_ipc": 0.0, "total_ipc": 0.0}

    ipc_linear = 0.0
    ipc_nonlinear = 0.0

    for w in range(0, n_eff - window_size, window_size):
        window = X_test[w : w + window_size]
        R = reservoir.transform(window)
        n_obs = len(R)
        if n_obs == 0:
            continue

        for tau in range(1, min(max_lag + 1, 25, n_obs)):
            if 24 - tau < 0:
                break
            start = 24 - tau
            y_raw = X_test[start : start + n_obs, 0]  # first feature as drive proxy
            if len(y_raw) < n_obs:
                break

            # Linear IPC (degree 1)
            lin_model = Ridge(alpha=1.0).fit(R, y_raw)
            y_lin_pred = lin_model.predict(R)
            lin_cov = np.cov(y_raw, y_lin_pred)[0, 1]
            var_y = np.var(y_raw)
            var_lin = np.var(y_lin_pred)
            if var_y > 1e-12 and var_lin > 1e-12:
                ipc_linear += (lin_cov ** 2) / (var_y * var_lin)

            # Quadratic IPC (degree 2)
            y_sq = y_raw ** 2
            sq_model = Ridge(alpha=1.0).fit(R, y_sq)
            y_sq_pred = sq_model.predict(R)
            sq_cov = np.cov(y_sq, y_sq_pred)[0, 1]
            var_sq = np.var(y_sq)
            var_sq_pred = np.var(y_sq_pred)
            if var_sq > 1e-12 and var_sq_pred > 1e-12:
                ipc_nonlinear += (sq_cov ** 2) / (var_sq * var_sq_pred)

    total_ipc = ipc_linear + ipc_nonlinear
    return {
        "linear_ipc": float(ipc_linear),
        "nonlinear_ipc": float(ipc_nonlinear),
        "total_ipc": float(total_ipc),
    }
