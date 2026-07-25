"""Memory Capacity (Jaeger 2001) and Information Processing Capacity
(Cindrak et al. 2026, arXiv:2603.21371) for SequentialDissipativeQRC.

Protocol note: the project's established MC protocol (see
docs/sprint_log/SPRINT_0 rules) is "drive with iid Gaussian windows, shape
(N, W, n_feat), train=test evaluation" -- this is how it's implemented for
the v4 *windowed* reservoir (AtmosphericQRC, pipeline_demo.py's
measure_memory_capacity), which re-encodes a fresh window from |0><0> for
every sample and has no state to carry between windows.

SequentialDissipativeQRC is architecturally different: it is a genuinely
recurrent reservoir with one continuously-evolving density matrix (that is
the entire point of Sprint 1). Jaeger 2001's ORIGINAL memory-capacity
definition drives the reservoir with a single continuous iid input stream
and asks whether u[t-k] is linearly recoverable from the state at time t --
this is what is implemented here via `drive()`, which is the natural fit
for a recurrent reservoir and avoids the O(n_samples * (washout+W)) cost of
re-deriving MC through the windowed `transform()` API (infeasible at this
reservoir's measured per-step cost -- see docs/sprint_log/SPRINT_2_REPORT.md).
Train=test evaluation (same indices for fit and score) is preserved exactly
as the established protocol requires; the OOS 70/30 split variant is never
used here, per the standing project rule.

MC and IPC accept an optional precomputed `(u, features)` pair (from
`drive_iid_gaussian`) so a characterization sweep over many hyperparameter
configs can share one drive per config between the two metrics instead of
driving twice (driving is the expensive part at this reservoir's measured
per-step cost, not the Ridge regressions).
"""
import numpy as np
from sklearn.linear_model import Ridge


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var_y = np.var(y_true)
    var_p = np.var(y_pred)
    if var_y < 1e-12 or var_p < 1e-12:
        return 0.0
    cov = np.cov(y_true, y_pred)[0, 1]
    return float((cov ** 2) / (var_y * var_p))


def _shuffle_surrogate_threshold(
    state: np.ndarray, target: np.ndarray, ridge_alpha: float,
    n_surrogates: int, percentile: float, rng: np.random.Generator,
) -> float:
    """Dambre et al. 2012 practice: shuffle `target` relative to `state`
    (breaking the true temporal correspondence while preserving each
    array's own marginal distribution) `n_surrogates` times, refit the same
    Ridge readout, and take the given percentile of the resulting r^2
    values as the significance threshold -- a measured capacity at or below
    this level is not distinguishable from a reservoir with no real memory
    of that lag/degree.
    """
    surrogate_r2 = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shuffled = rng.permutation(target)
        model = Ridge(alpha=ridge_alpha).fit(state, shuffled)
        surrogate_r2[i] = _r_squared(shuffled, model.predict(state))
    return float(np.percentile(surrogate_r2, percentile))


def drive_iid_gaussian(qrc, n_steps: int, seed: int = 0) -> tuple:
    """Drive `qrc` with a single continuous iid standard-normal stream
    broadcast through qrc.w_in, as MC/IPC below require. Returns (u, features)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n_steps)
    seq = u[:, None] * qrc.w_in[None, :]
    features = qrc.drive(seq)
    return u, features


def measure_memory_capacity_sequential(
    qrc=None, n_steps: int = 500, max_lag: int = 20, ridge_alpha: float = 1e-2, seed: int = 0,
    u: np.ndarray = None, features: np.ndarray = None,
    threshold_surrogates: bool = False, n_surrogates: int = 20, surrogate_percentile: float = 95.0,
    surrogate_seed: int = 1000,
) -> dict:
    """Linear memory capacity via a single continuous iid Gaussian drive
    (Jaeger 2001's original protocol; see module docstring for why this,
    not the windowed variant, is used for this reservoir). Train=test.

    Healthy range for the 12-qubit reference config: 5-10 (established
    project convention); reported as-is here regardless of qubit count
    actually used (see caller for which n_qubits this was run at). Pass
    precomputed (u, features) to reuse a drive already done for IPC.

    threshold_surrogates=True applies Dambre et al. 2012 shuffle-surrogate
    thresholding (see `_shuffle_surrogate_threshold`): per-lag capacities at
    or below the surrogate-null percentile are zeroed. `MC` is always the
    sum of the (possibly thresholded) per-lag values; `MC_raw` preserves the
    untresholded sum for comparison.
    """
    if u is None or features is None:
        u, features = drive_iid_gaussian(qrc, n_steps, seed)
    n_steps = len(u)
    rng = np.random.default_rng(surrogate_seed)

    mc_per_lag_raw = []
    mc_per_lag = []
    thresholds = []
    for k in range(1, max_lag + 1):
        if k >= n_steps:
            break
        target = u[:-k]
        state = features[k:]
        model = Ridge(alpha=ridge_alpha).fit(state, target)
        pred = model.predict(state)
        r2 = _r_squared(target, pred)
        mc_per_lag_raw.append(r2)
        if threshold_surrogates:
            thresh = _shuffle_surrogate_threshold(state, target, ridge_alpha, n_surrogates, surrogate_percentile, rng)
            thresholds.append(thresh)
            mc_per_lag.append(r2 if r2 > thresh else 0.0)
        else:
            mc_per_lag.append(r2)

    result = {
        "MC": float(np.sum(mc_per_lag)), "MC_raw": float(np.sum(mc_per_lag_raw)),
        "per_lag": mc_per_lag, "per_lag_raw": mc_per_lag_raw,
        "n_steps": n_steps, "max_lag": max_lag, "threshold_surrogates": threshold_surrogates,
    }
    if threshold_surrogates:
        result["surrogate_thresholds"] = thresholds
    return result


def measure_ipc_sequential(
    qrc=None, n_steps: int = 500, max_lag: int = 10, ridge_alpha: float = 1e-2, seed: int = 1,
    u: np.ndarray = None, features: np.ndarray = None,
    threshold_surrogates: bool = False, n_surrogates: int = 20, surrogate_percentile: float = 95.0,
    surrogate_seed: int = 2000,
) -> dict:
    """Linear + nonlinear (quadratic) Information Processing Capacity,
    Cindrak et al. 2026 (arXiv:2603.21371) linear-capacity framework:
    for each lag k and degree d in {1 (linear), 2 (quadratic)}, predict
    u[t-k]^d from the reservoir state at time t via Ridge, train=test,
    and sum squared correlations. Total IPC = linear + nonlinear. Pass
    precomputed (u, features) to reuse a drive already done for MC.

    threshold_surrogates=True applies Dambre et al. 2012 shuffle-surrogate
    thresholding independently to the linear and quadratic component at
    each lag (see `_shuffle_surrogate_threshold`); `total_ipc`/`linear_ipc`/
    `nonlinear_ipc` are computed from the (possibly thresholded) values,
    with `*_raw` variants preserving the unthresholded sums.
    """
    if u is None or features is None:
        u, features = drive_iid_gaussian(qrc, n_steps, seed)
    n_steps = len(u)
    rng = np.random.default_rng(surrogate_seed)

    linear_ipc = linear_ipc_raw = 0.0
    nonlinear_ipc = nonlinear_ipc_raw = 0.0
    per_lag = []
    for k in range(1, max_lag + 1):
        if k >= n_steps:
            break
        state = features[k:]
        target_lin = u[:-k]
        target_quad = u[:-k] ** 2 - 1.0  # Hermite-like: remove the mean of u^2 (E[u^2]=1 for standard normal)

        lin_model = Ridge(alpha=ridge_alpha).fit(state, target_lin)
        lin_r2 = _r_squared(target_lin, lin_model.predict(state))

        quad_model = Ridge(alpha=ridge_alpha).fit(state, target_quad)
        quad_r2 = _r_squared(target_quad, quad_model.predict(state))

        linear_ipc_raw += lin_r2
        nonlinear_ipc_raw += quad_r2

        if threshold_surrogates:
            lin_thresh = _shuffle_surrogate_threshold(state, target_lin, ridge_alpha, n_surrogates, surrogate_percentile, rng)
            quad_thresh = _shuffle_surrogate_threshold(state, target_quad, ridge_alpha, n_surrogates, surrogate_percentile, rng)
            lin_r2_t = lin_r2 if lin_r2 > lin_thresh else 0.0
            quad_r2_t = quad_r2 if quad_r2 > quad_thresh else 0.0
        else:
            lin_r2_t, quad_r2_t = lin_r2, quad_r2

        linear_ipc += lin_r2_t
        nonlinear_ipc += quad_r2_t
        per_lag.append({"lag": k, "linear": lin_r2_t, "quadratic": quad_r2_t,
                         "linear_raw": lin_r2, "quadratic_raw": quad_r2})

    return {
        "linear_ipc": float(linear_ipc),
        "nonlinear_ipc": float(nonlinear_ipc),
        "total_ipc": float(linear_ipc + nonlinear_ipc),
        "linear_ipc_raw": float(linear_ipc_raw),
        "nonlinear_ipc_raw": float(nonlinear_ipc_raw),
        "total_ipc_raw": float(linear_ipc_raw + nonlinear_ipc_raw),
        "per_lag": per_lag,
        "n_steps": n_steps,
        "max_lag": max_lag,
        "threshold_surrogates": threshold_surrogates,
    }


def _hermite_probabilists(u: np.ndarray, degree: int) -> np.ndarray:
    """Probabilists' Hermite polynomials He_d(u) -- the orthogonal basis
    for a standard-normal-distributed variable (E[He_i(u) He_j(u)] = 0 for
    i != j when u ~ N(0,1)), matching this project's iid standard-normal
    drive convention. He_1=u, He_2=u^2-1, He_3=u^3-3u (all zero-mean under
    the standard normal measure)."""
    if degree == 1:
        return u
    if degree == 2:
        return u ** 2 - 1.0
    if degree == 3:
        return u ** 3 - 3.0 * u
    raise ValueError(f"degree {degree} not supported (only 1, 2, 3 implemented)")


def measure_ipc_by_degree(
    qrc=None, n_steps: int = 500, max_lag: int = 24, degrees=(1, 2, 3), ridge_alpha: float = 1e-2, seed: int = 1,
    u: np.ndarray = None, features: np.ndarray = None,
) -> dict:
    """Sprint 5 extension of `measure_ipc_sequential` to arbitrary low-order
    degree (default 1-3, generalizing that function's fixed linear+
    quadratic split) via probabilists' Hermite polynomials of the iid
    standard-normal drive `u` — the correct orthogonal basis for this
    project's Gaussian-drive convention (see `_hermite_probabilists`).
    Independent per-(lag, degree) Ridge regression, train=test, same
    protocol as `measure_ipc_sequential` and `task_demand.compute_demand_profile`
    (so the two produce directly comparable `C`/`D` matrices of identical
    shape/axis convention for `Σ min(C, D)` matching).

    Returns `C` (len(degrees) x max_lag array of R^2 capacity values) plus
    metadata. Does NOT replace `measure_ipc_sequential` (kept as-is,
    unmodified, for backward compatibility with existing Sprint 2-4
    results/tests) -- this is a new, additive function for Sprint 5.
    """
    if u is None or features is None:
        u, features = drive_iid_gaussian(qrc, n_steps, seed)
    n_steps = len(u)
    degrees = list(degrees)

    # lags start at 0 (not 1, unlike measure_ipc_sequential/MC): lag=0 is a
    # legitimate, commonly-included IPC component (Dambre et al. 2012's
    # total capacity sums over k=0,1,2,... too, representing the readout's
    # instantaneous nonlinear-transform capacity of the CURRENT input) --
    # included here specifically so `lags` aligns with
    # `task_demand.compute_demand_profile`'s `delays` (which also starts at
    # 0) for direct Σ min(C, D) matching.
    lags = [k for k in range(0, max_lag + 1) if k < n_steps]
    C = np.zeros((len(degrees), len(lags)))
    for di, degree in enumerate(degrees):
        target_full = _hermite_probabilists(u, degree)
        for ki, k in enumerate(lags):
            if k == 0:
                target = target_full
                state = features
            else:
                target = target_full[:-k]
                state = features[k:]
            model = Ridge(alpha=ridge_alpha).fit(state, target)
            pred = model.predict(state)
            C[di, ki] = _r_squared(target, pred)

    return {
        "C": C, "degrees": degrees, "lags": lags,
        "total_capacity": float(C.sum()), "n_steps": n_steps, "max_lag": max_lag,
    }
