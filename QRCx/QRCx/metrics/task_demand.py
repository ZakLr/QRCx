"""Task demand profile (Sprint 5) — the "what shape of memory/nonlinearity
does this forecasting task actually need" counterpart to a reservoir's own
supply-side Information Processing Capacity (Cindrak et al. 2026,
arXiv:2603.21371). No published work applies this demand/supply matching
to operational weather forecasting; this module and
`docs/ipc_matching.md` are the project's attempt.

Demand D(delay, degree) is defined analogously to supply-side IPC: how
much of the h-step-ahead target's variance is explained (R², train=test,
independent per-cell regression — same convention as
`metrics/reservoir_sequential.py::measure_ipc_sequential`, not a jointly
orthogonalized Volterra decomposition) by a degree-`degree` Legendre
polynomial of the target's own value `delay` steps in the past.

Basis choice: Legendre polynomials are the orthogonal basis for a
variable uniformly distributed on [-1, 1]; the real anomaly target here
is approximately standard-normal, not uniform, so raw values are first
smoothly squashed into (-1, 1) via tanh before evaluating P_1/P_2/P_3 —
a documented simplification, not a distributional transform. This
mirrors why the supply side (`measure_ipc_by_degree`, this project's
extension of `measure_ipc_sequential` to degree 3) instead uses Hermite
polynomials for its iid-Gaussian drive: each side uses the orthogonal
basis appropriate to its own input's actual distribution.
"""
import numpy as np
from sklearn.linear_model import Ridge


def _map_to_bounded(x: np.ndarray) -> np.ndarray:
    """Smooth squash into (-1, 1) for Legendre-polynomial evaluation."""
    return np.tanh(x / 2.0)


def legendre_value(x_bounded: np.ndarray, degree: int) -> np.ndarray:
    if degree == 1:
        return x_bounded
    if degree == 2:
        return 0.5 * (3.0 * x_bounded ** 2 - 1.0)
    if degree == 3:
        return 0.5 * (5.0 * x_bounded ** 3 - 3.0 * x_bounded)
    raise ValueError(f"degree {degree} not supported (only 1, 2, 3 implemented)")


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var_y = np.var(y_true)
    var_p = np.var(y_pred)
    if var_y < 1e-12 or var_p < 1e-12:
        return 0.0
    cov = np.cov(y_true, y_pred)[0, 1]
    return float((cov ** 2) / (var_y * var_p))


def compute_demand_profile(
    series: np.ndarray,
    horizon: int,
    delays=range(0, 25),
    degrees=(1, 2, 3),
    ridge_alpha: float = 1e-2,
) -> dict:
    """`series`: 1D real target sequence (already climatological-anomaly
    scaled, e.g. train_seq[:, target_col_idx]). Regresses
    `series[t+horizon]` on `P_degree(bounded(series[t-delay]))`
    independently per (delay, degree) cell, train=test (same protocol as
    the reservoir MC/IPC metrics — see module docstring for why this
    per-cell-independent convention, not a joint decomposition, matches
    the project's established methodology).

    Returns a dict with `D` (len(degrees) x len(delays) array of R^2
    capacity values, same shape/axis convention as `measure_ipc_by_degree`'s
    `C` so `Σ min(C, D)` can be computed directly) and metadata.
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    delays = list(delays)
    degrees = list(degrees)
    max_delay = max(delays)
    start = max_delay
    end = n - horizon
    if end <= start:
        raise ValueError(
            f"series too short ({n}) for max_delay={max_delay} and horizon={horizon}"
        )
    y_target = series[start + horizon:end + horizon]
    x_bounded_full = _map_to_bounded(series)

    D = np.zeros((len(degrees), len(delays)))
    for di, degree in enumerate(degrees):
        for ki, delay in enumerate(delays):
            x_lag = x_bounded_full[start - delay:end - delay]
            feat = legendre_value(x_lag, degree).reshape(-1, 1)
            model = Ridge(alpha=ridge_alpha).fit(feat, y_target)
            pred = model.predict(feat)
            D[di, ki] = _r_squared(y_target, pred)

    return {
        "D": D,
        "delays": delays,
        "degrees": degrees,
        "horizon": horizon,
        "total_demand": float(D.sum()),
        "n_samples": int(end - start),
    }
