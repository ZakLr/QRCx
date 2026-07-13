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
) -> dict:
    """Linear memory capacity via a single continuous iid Gaussian drive
    (Jaeger 2001's original protocol; see module docstring for why this,
    not the windowed variant, is used for this reservoir). Train=test.

    Healthy range for the 12-qubit reference config: 5-10 (established
    project convention); reported as-is here regardless of qubit count
    actually used (see caller for which n_qubits this was run at). Pass
    precomputed (u, features) to reuse a drive already done for IPC.
    """
    if u is None or features is None:
        u, features = drive_iid_gaussian(qrc, n_steps, seed)
    n_steps = len(u)

    mc_per_lag = []
    for k in range(1, max_lag + 1):
        if k >= n_steps:
            break
        target = u[:-k]
        state = features[k:]
        model = Ridge(alpha=ridge_alpha).fit(state, target)
        pred = model.predict(state)
        mc_per_lag.append(_r_squared(target, pred))

    return {"MC": float(np.sum(mc_per_lag)), "per_lag": mc_per_lag, "n_steps": n_steps, "max_lag": max_lag}


def measure_ipc_sequential(
    qrc=None, n_steps: int = 500, max_lag: int = 10, ridge_alpha: float = 1e-2, seed: int = 1,
    u: np.ndarray = None, features: np.ndarray = None,
) -> dict:
    """Linear + nonlinear (quadratic) Information Processing Capacity,
    Cindrak et al. 2026 (arXiv:2603.21371) linear-capacity framework:
    for each lag k and degree d in {1 (linear), 2 (quadratic)}, predict
    u[t-k]^d from the reservoir state at time t via Ridge, train=test,
    and sum squared correlations. Total IPC = linear + nonlinear. Pass
    precomputed (u, features) to reuse a drive already done for MC.
    """
    if u is None or features is None:
        u, features = drive_iid_gaussian(qrc, n_steps, seed)
    n_steps = len(u)

    linear_ipc = 0.0
    nonlinear_ipc = 0.0
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

        linear_ipc += lin_r2
        nonlinear_ipc += quad_r2
        per_lag.append({"lag": k, "linear": lin_r2, "quadratic": quad_r2})

    return {
        "linear_ipc": float(linear_ipc),
        "nonlinear_ipc": float(nonlinear_ipc),
        "total_ipc": float(linear_ipc + nonlinear_ipc),
        "per_lag": per_lag,
        "n_steps": n_steps,
        "max_lag": max_lag,
    }
