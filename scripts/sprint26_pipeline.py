"""Sprint 2.6 standalone pipeline: precision validation + joint random
search + complex128 refine + 12q verification + redefined gate + IPC/MC
characterization at the winning config.

Self-contained (no QRCx import) so it can be pasted directly into a
qBraid Lab GPU notebook cell, mirroring scripts/gpu_verify_standalone.py's
approach. Requires numpy, scikit-learn, and (for GPU) cupy -- falls back
to numpy/CPU if cupy isn't available, but this is designed to be run on
GPU (the search alone is up to 300-500 full-protocol configs, each
thousands of steps; infeasible in reasonable time on CPU).

DEVIATIONS FROM THE LITERAL SPEC, and why (all computed and logged at
runtime, never silent):
  1. The joint search (Stage 1) uses a cheaper SEARCH_PROTOCOL
     (train=800/test=300) instead of the literal train=3000/test=1000,
     and auto-sizes its config count to a 3h budget instead of a fixed
     300-500 -- the literal full protocol combined with V in {4,8}
     multiplexing at even 10 qubits is too expensive to explore that many
     times within the sprint's 2-day time-box. FULL_PROTOCOL (the literal
     spec numbers) IS used for every stage that actually determines the
     reported result: refine, 12q verification, and the gate itself.
  2. Stages 2 and 3 (refine top-10, verify top-3) each time ONE
     representative FULL_PROTOCOL config first and adaptively shrink the
     count that follows (down to a minimum of 3 and 1 respectively) to
     fit fixed time budgets (6h, 12h) -- a single 12-qubit, V=8,
     FULL_PROTOCOL config was, in early estimation, projected to
     potentially take on the order of a day by itself; this makes the
     script self-adapting to whatever the real (uncertain, until measured
     here) GPU throughput for this heavier multiplexed+extraction
     workload turns out to be, rather than assuming and risking blowing
     the time-box or crashing partway through a fixed-count loop.

Usage in qBraid Lab:
    !pip install cupy-cuda12x[ctk]   # if not already installed
    # paste this file's contents into a cell and run
    # (takes several hours on GPU; consider running as a background
    #  script via `!python sprint26_pipeline.py > log.txt 2>&1 &` instead
    #  of a single notebook cell if the session has an idle timeout)
    # paste back the final printed JSON summary (or the saved file's
    # contents) once done

Time-boxed per the sprint spec: if this hasn't finished in ~2 days,
whatever completed so far should be reported as-is (see main()'s
checkpoint saving -- partial results are written after every stage, not
only at the very end, specifically so a timeout doesn't lose everything).
"""
import itertools
import json
import time

import numpy as np
from sklearn.linear_model import Ridge

try:
    import cupy as cp
    HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    HAS_CUPY = False

print(f"HAS_CUPY (GPU visible): {HAS_CUPY}")
if HAS_CUPY:
    print(f"GPU device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")

OUT_PATH = "sprint26_results.json"
AR_TAPS_ORDER = 10  # matches the project's established AR-baseline tap order

# ============================================================
# Core reservoir engine (mirrors QRCx/QRCx/reservoir/sequential.py and
# QRCx/QRCx/readout/correlators.py's *_dm helpers)
# ============================================================


def build_tfim_hamiltonian(n_qubits, J, g, z_np):
    dim = 2 ** n_qubits
    diag = np.zeros(dim, dtype=np.float64)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            diag += J * z_np[i] * z_np[j]
    H = np.diag(diag).astype(np.complex128)
    idx = np.arange(dim)
    for q in range(n_qubits):
        bitpos = n_qubits - 1 - q
        flipped = idx ^ (1 << bitpos)
        H[idx, flipped] += g
    return H


def ry_matrix(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


class DensityOps:
    def __init__(self, n_qubits, use_gpu, dtype_str):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.xp = cp if (use_gpu and HAS_CUPY) else np
        self.on_gpu = use_gpu and HAS_CUPY
        self.dtype = getattr(self.xp, dtype_str)
        self.float_dtype = self.xp.float64 if dtype_str == "complex128" else self.xp.float32
        idx = np.arange(self.dim)
        self._z_np = [1 - 2 * ((idx >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)]
        self._z = [self.xp.asarray(z) for z in self._z_np]

    def to_numpy(self, a):
        a = a if self.xp is np else self.xp.asnumpy(a)
        return a if a.dtype == np.complex128 else a.astype(np.complex128)

    def vacuum(self):
        rho = self.xp.zeros((self.dim, self.dim), dtype=self.dtype)
        rho[0, 0] = 1.0
        return rho

    def conjugate_1q_q(self, rho, K, qubit):
        """rho -> K rho K^dagger, K a (2,2) matrix acting on qubit `qubit`."""
        xp = self.xp
        K = xp.asarray(K, dtype=rho.dtype)
        dim = self.dim
        P, S = 2 ** qubit, 2 ** (self.n - qubit - 1)
        r = rho.reshape(P, 2, S * dim)
        r = xp.matmul(K, r)
        r = r.reshape(dim, dim)
        Kc = xp.conj(K)
        r = r.reshape(dim * P, 2, S)
        r = xp.matmul(Kc, r)
        return r.reshape(dim, dim)

    def conjugate_dense(self, rho, U):
        U = self.xp.asarray(U, dtype=self.dtype)
        return U @ rho @ U.conj().T

    def dephasing_coeff(self, gamma2):
        xp = self.xp
        c = float(np.sqrt(max(0.0, 1.0 - gamma2)))
        coeff = xp.ones((self.dim, self.dim), dtype=self.float_dtype)
        for q in range(self.n):
            same_bit = (self._z[q][:, None] * self._z[q][None, :]) > 0
            coeff = coeff * xp.where(same_bit, 1.0, c).astype(self.float_dtype)
        return coeff


def ptrace_1q_np(rho, qubit, n_qubits):
    """Single-qubit reduced density matrix, direct einsum (no transpose)."""
    P, S = 2 ** qubit, 2 ** (n_qubits - qubit - 1)
    t = rho.reshape(P, 2, S, P, 2, S)
    return np.einsum("paspbs->ab", t, optimize=True)


def ptrace_2q_np(rho, q1, q2, n_qubits):
    assert q1 < q2
    A, B, C = 2 ** q1, 2 ** (q2 - q1 - 1), 2 ** (n_qubits - q2 - 1)
    t = rho.reshape(A, 2, B, 2, C, A, 2, B, 2, C)
    reduced = np.einsum("pwqxrpyqzr->wxyz", t, optimize=True)
    return reduced.reshape(4, 4)


_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def extract_correlators_dm(rho_np, n_qubits):
    """234-dim (at n=12) Pauli correlators from a numpy (dim,dim) rho."""
    n_pairs = n_qubits * (n_qubits - 1) // 2
    n_feat = 3 * n_qubits + 3 * n_pairs
    out = np.zeros(n_feat, dtype=np.float64)
    for i in range(n_qubits):
        rho_i = ptrace_1q_np(rho_np, i, n_qubits)
        for p_idx, p in enumerate(["X", "Y", "Z"]):
            out[p_idx * n_qubits + i] = float(np.real(np.trace(rho_i @ _PAULI[p])))
    p_idx = 3 * n_qubits
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            rho_ij = ptrace_2q_np(rho_np, i, j, n_qubits)
            for p in ["Z", "X", "Y"]:
                op = np.kron(_PAULI[p], _PAULI[p])
                out[p_idx] = float(np.real(np.trace(rho_ij @ op)))
                p_idx += 1
    return out


class SequentialReservoir:
    """Mirrors QRCx.reservoir.sequential.SequentialDissipativeQRC, with
    tau and J/g ratio exposed as free parameters (Sprint 2.6's new sweep
    axes) and full-injection (n_in=n_qubits) only -- Sprint 2.5 found
    restricted injection doesn't help, so this sprint doesn't re-sweep it.
    """

    def __init__(self, n_qubits, tau, trotter_steps, gamma1, gamma2, J, g,
                 input_scaling, w_in, multiplexing, use_gpu, dtype_str, seed=42):
        self.n_qubits = n_qubits
        self.tau = tau
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.input_scaling = input_scaling
        self.w_in = w_in
        self.multiplexing = multiplexing
        self.ops = DensityOps(n_qubits, use_gpu, dtype_str)

        H = build_tfim_hamiltonian(n_qubits, J, g, self.ops._z_np)
        evals, evecs = np.linalg.eigh(H)
        self._evals, self._evecs = evals, evecs
        self._U_inc = self._propagator(tau / multiplexing)

        self._dephasing_coeff = self.ops.dephasing_coeff(gamma2) if gamma2 > 0 else None
        if gamma1 > 0:
            c1 = float(np.sqrt(max(0.0, 1.0 - gamma1)))
            self._damping_d = [
                self.ops.xp.where(self.ops._z[q] > 0, 1.0, c1).astype(self.ops.float_dtype)
                for q in range(n_qubits)
            ]
            self._damping_e1 = self.ops.xp.asarray(
                np.array([[0, np.sqrt(gamma1)], [0, 0]], dtype=np.complex128), dtype=self.ops.dtype
            )
        else:
            self._damping_d = None

    def _propagator(self, t):
        phases = np.exp(-1j * self._evals * t)
        return (self._evecs * phases) @ self._evecs.conj().T

    def _inject(self, rho, x):
        for j in range(self.n_qubits):
            rho = self.ops.conjugate_1q_q(rho, ry_matrix(self.input_scaling * self.w_in[j] * x[j]), j)
        return rho

    def _dissipate(self, rho):
        if self.gamma2 > 0:
            rho = rho * self._dephasing_coeff
        if self.gamma1 > 0:
            for i in range(self.n_qubits):
                d = self._damping_d[i]
                term0 = rho * d[:, None] * d[None, :]
                term1 = self.ops.conjugate_1q_q(rho, self._damping_e1, i)
                rho = term0 + term1
        return rho

    def drive(self, seq):
        """seq: (T, n_qubits). Returns features (T, V*n_feat)."""
        T = seq.shape[0]
        n_pairs = self.n_qubits * (self.n_qubits - 1) // 2
        n_feat = 3 * self.n_qubits + 3 * n_pairs
        V = self.multiplexing
        out = np.zeros((T, V * n_feat), dtype=np.float64)
        rho = self.ops.vacuum()
        for t in range(T):
            rho = self._inject(rho, seq[t])
            for v in range(V):
                rho = self.ops.conjugate_dense(rho, self._U_inc)
                rho_np = self.ops.to_numpy(rho)
                out[t, v * n_feat:(v + 1) * n_feat] = extract_correlators_dm(rho_np, self.n_qubits)
            rho = self._dissipate(rho)
        return out


# ============================================================
# NARMA10 + evaluation utilities
# ============================================================


def generate_narma10(n_steps, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n_steps)
    y = np.zeros(n_steps)
    for t in range(10, n_steps):
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * np.sum(y[t - 10:t]) + 1.5 * u[t - 10] * u[t - 1] + 0.1
    return u, y


def nmse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def fit_eval_with_val_tuning(X_fit, y_fit, X_val, y_val, X_test, y_test, X_train_full, y_train_full):
    best_alpha, best_score = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha).fit(X_fit, y_fit)
        score = nmse(y_val, m.predict(X_val))
        if score < best_score:
            best_score, best_alpha = score, alpha
    final_model = Ridge(alpha=best_alpha).fit(X_train_full, y_train_full)
    return nmse(y_test, final_model.predict(X_test)), best_alpha


def build_taps(u, y, order=AR_TAPS_ORDER):
    """(n-order, 2*order) tap matrix: [y[t-order..t-1], u[t-order..t-1]]."""
    n = len(y)
    X = np.stack([np.concatenate([y[t - order:t], u[t - order:t]]) for t in range(order, n)])
    return X


def evaluate_config(u, y, features, washout, n_train, n_val_frac, n_test):
    """Full valid protocol. Returns dict with track_nmse (reservoir-only)
    and gate_nmse info (taps-alone vs taps+reservoir)."""
    order = AR_TAPS_ORDER
    start = max(washout, order)
    feats = features[start:]
    y_eval = y[start:]
    u_eval = u[start:]
    taps = build_taps(u[:len(y)], y, order=order)[start - order:]

    assert len(y_eval) >= n_train + n_test, f"need {n_train + n_test} post-washout steps, got {len(y_eval)}"
    n_val = int(n_train * n_val_frac)
    n_fit = n_train - n_val

    def split(A):
        return A[:n_fit], A[n_fit:n_fit + n_val], A[n_train:n_train + n_test], A[:n_train]

    # Track: reservoir features alone.
    Xf, Xv, Xt, Xtr = split(feats)
    yf, yv, yt, ytr = split(y_eval)
    track_nmse, track_alpha = fit_eval_with_val_tuning(Xf, yf, Xv, yv, Xt, yt, Xtr, ytr)

    # Gate: taps alone vs taps+reservoir.
    Tf, Tv, Tt, Ttr = split(taps)
    taps_alone_nmse, _ = fit_eval_with_val_tuning(Tf, yf, Tv, yv, Tt, yt, Ttr, ytr)

    combo = np.concatenate([taps, feats], axis=1)
    Cf, Cv, Ct, Ctr = split(combo)
    combo_nmse, _ = fit_eval_with_val_tuning(Cf, yf, Cv, yv, Ct, yt, Ctr, ytr)

    delta_nmse = (taps_alone_nmse - combo_nmse) / taps_alone_nmse if taps_alone_nmse > 0 else 0.0

    return {
        "track_nmse": track_nmse, "track_alpha": track_alpha,
        "taps_alone_nmse": taps_alone_nmse, "taps_plus_reservoir_nmse": combo_nmse,
        "delta_nmse_relative": delta_nmse,
        "n_features": feats.shape[1], "n_train": n_train, "n_test": n_test,
    }


# ============================================================
# Precision validation (Sprint 2.6 Phase A)
# ============================================================


def precision_validation(n_qubits=12, n_steps=500, use_gpu=True):
    rng = np.random.default_rng(0)
    w_in = rng.uniform(0.5, 1.5, size=n_qubits)
    seq = rng.uniform(-1, 1, size=(n_steps, n_qubits)) * w_in[None, :]

    common = dict(n_qubits=n_qubits, tau=1.0, trotter_steps=10, gamma1=0.03, gamma2=0.1,
                  J=1.0, g=1.0, input_scaling=0.3, w_in=w_in, multiplexing=4, use_gpu=use_gpu)
    res128 = SequentialReservoir(dtype_str="complex128", **common)
    res64 = SequentialReservoir(dtype_str="complex64", **common)

    t0 = time.perf_counter()
    f128 = res128.drive(seq)
    t128 = time.perf_counter() - t0

    t0 = time.perf_counter()
    f64 = res64.drive(seq)
    t64 = time.perf_counter() - t0

    rel_err = np.max(np.abs(f128 - f64) / (np.abs(f128) + 1e-8))
    result = {
        "n_qubits": n_qubits, "n_steps": n_steps,
        "max_relative_error": float(rel_err),
        "pass": bool(rel_err < 1e-5),
        "complex128_s_per_step": t128 / n_steps,
        "complex64_s_per_step": t64 / n_steps,
    }
    print("PRECISION VALIDATION:", json.dumps(result))
    return result


# ============================================================
# Joint random search (Sprint 2.6 Phase B)
# ============================================================


def sample_config(rng):
    return {
        "gamma1": float(np.exp(rng.uniform(np.log(0.01), np.log(0.5)))),
        "gamma2": float(rng.uniform(0.0, 0.3)),
        "a": float(np.exp(rng.uniform(np.log(0.1), np.log(5.0)))),
        "tau": float(np.exp(rng.uniform(np.log(0.2), np.log(4.0)))),
        "jg_ratio": float(rng.choice([0.5, 0.8, 1.0, 1.25, 2.0])),
        "V": int(rng.choice([4, 8])),
        "washout": int(rng.choice([50, 200])),
        "w_in_seed": int(rng.choice([1, 2, 3])),
    }



# Two-tier protocol: the literal spec text asks for "full valid protocol
# (train 3000 / val slice / washout 200)" for the joint search itself, but
# combined with V in {4,8} multiplexing (each step doing 4-8x the
# evolution+extraction work) that is NOT feasible within the sprint's
# explicit 2-day time-box even on GPU -- a single 12-qubit V=8 config at
# the full protocol alone would take on the order of a day (see
# time-budget estimate printed at the start of main()). SEARCH_PROTOCOL is
# a cheaper stand-in used ONLY for the 300-500-config search stage;
# FULL_PROTOCOL (the literal spec numbers) is used for the top-10 refine,
# top-3 12q verification, and the final gate-stability check across 3
# w_in seeds -- i.e. the protocol that actually determines the reported
# gate result uses the exact spec numbers; only the cheap exploratory
# stage is reduced. This deviation is deliberate and logged, not silent.
SEARCH_PROTOCOL = dict(n_train=800, n_val_frac=0.2, n_test=300)
FULL_PROTOCOL = dict(n_train=3000, n_val_frac=0.2, n_test=1000)


def run_one_config(cfg, n_qubits, dtype_str, use_gpu, protocol):
    washout = cfg["washout"]
    # evaluate_config's usable range starts at max(washout, AR_TAPS_ORDER)
    # (taps need `order` steps of history) -- total_steps must account for
    # that when washout < AR_TAPS_ORDER, or the split will come up short.
    effective_start = max(washout, AR_TAPS_ORDER)
    total_steps = protocol["n_train"] + protocol["n_test"] + effective_start
    u, y = generate_narma10(n_steps=total_steps, seed=42)
    w_in = np.random.default_rng(cfg["w_in_seed"]).uniform(0.5, 1.5, size=n_qubits)

    g = 1.0
    J = cfg["jg_ratio"] * g
    res = SequentialReservoir(
        n_qubits=n_qubits, tau=cfg["tau"], trotter_steps=10, gamma1=cfg["gamma1"],
        gamma2=cfg["gamma2"], J=J, g=g, input_scaling=cfg["a"], w_in=w_in,
        multiplexing=cfg["V"], use_gpu=use_gpu, dtype_str=dtype_str,
    )
    seq = u[:, None] * w_in[None, :]
    t0 = time.perf_counter()
    feats = res.drive(seq)
    drive_s = time.perf_counter() - t0

    ev = evaluate_config(u, y, feats, washout, protocol["n_train"], protocol["n_val_frac"], protocol["n_test"])
    return {**cfg, **ev, "n_qubits": n_qubits, "dtype": dtype_str, "drive_wall_clock_s": drive_s}


def estimate_search_budget(n_qubits, dtype_str, use_gpu, target_hours=3.0):
    """Time ONE representative (worst-case V=8) search-stage config, then
    pick n_configs to fit `target_hours` -- printed, not silently assumed,
    so the user can see and override before committing GPU time."""
    rng = np.random.default_rng(12345)
    cfg = sample_config(rng)
    cfg["V"] = 8  # worst case, for a conservative (not optimistic) budget
    cfg["washout"] = 200  # worst case
    t0 = time.perf_counter()
    run_one_config(cfg, n_qubits, dtype_str, use_gpu, SEARCH_PROTOCOL)
    per_config_s = time.perf_counter() - t0
    n_configs = max(20, min(500, int(target_hours * 3600 / per_config_s)))
    print(f"Worst-case search-stage config took {per_config_s:.1f}s; "
          f"targeting {target_hours}h search budget -> n_configs={n_configs} "
          f"(spec asked for 300-500; this is a computed, logged deviation, not silent).")
    return n_configs


def r_squared(y_true, y_pred):
    var_y, var_p = np.var(y_true), np.var(y_pred)
    if var_y < 1e-12 or var_p < 1e-12:
        return 0.0
    cov = np.cov(y_true, y_pred)[0, 1]
    return float((cov ** 2) / (var_y * var_p))


def shuffle_threshold(state, target, alpha, n_surrogates, percentile, rng):
    """Dambre et al. 2012 shuffle-surrogate threshold (mirrors
    QRCx.metrics.reservoir_sequential._shuffle_surrogate_threshold)."""
    surrogate_r2 = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shuffled = rng.permutation(target)
        m = Ridge(alpha=alpha).fit(state, shuffled)
        surrogate_r2[i] = r_squared(shuffled, m.predict(state))
    return float(np.percentile(surrogate_r2, percentile))


def measure_mc_ipc(u, features, max_lag_mc=12, max_lag_ipc=8, ridge_alpha=1e-2,
                    n_surrogates=20, percentile=95.0, seed=1000):
    """MC + IPC (linear/nonlinear) with shuffle-surrogate thresholding,
    mirroring QRCx.metrics.reservoir_sequential exactly (same math, ported
    standalone -- see docs/sprint_log/SPRINT_2_5_REPORT.md for the
    original)."""
    rng = np.random.default_rng(seed)
    n_steps = len(u)

    mc_raw, mc_thr = 0.0, 0.0
    for k in range(1, max_lag_mc + 1):
        if k >= n_steps:
            break
        target, state = u[:-k], features[k:]
        model = Ridge(alpha=ridge_alpha).fit(state, target)
        r2 = r_squared(target, model.predict(state))
        mc_raw += r2
        thresh = shuffle_threshold(state, target, ridge_alpha, n_surrogates, percentile, rng)
        mc_thr += r2 if r2 > thresh else 0.0

    lin_raw = lin_thr = nonlin_raw = nonlin_thr = 0.0
    for k in range(1, max_lag_ipc + 1):
        if k >= n_steps:
            break
        state = features[k:]
        t_lin, t_quad = u[:-k], u[:-k] ** 2 - 1.0
        lin_m = Ridge(alpha=ridge_alpha).fit(state, t_lin)
        lin_r2 = r_squared(t_lin, lin_m.predict(state))
        quad_m = Ridge(alpha=ridge_alpha).fit(state, t_quad)
        quad_r2 = r_squared(t_quad, quad_m.predict(state))
        lin_raw += lin_r2
        nonlin_raw += quad_r2
        lin_th = shuffle_threshold(state, t_lin, ridge_alpha, n_surrogates, percentile, rng)
        quad_th = shuffle_threshold(state, t_quad, ridge_alpha, n_surrogates, percentile, rng)
        lin_thr += lin_r2 if lin_r2 > lin_th else 0.0
        nonlin_thr += quad_r2 if quad_r2 > quad_th else 0.0

    return {
        "MC": mc_thr, "MC_raw": mc_raw,
        "linear_ipc": lin_thr, "nonlinear_ipc": nonlin_thr, "total_ipc": lin_thr + nonlin_thr,
        "linear_ipc_raw": lin_raw, "nonlinear_ipc_raw": nonlin_raw, "total_ipc_raw": lin_raw + nonlin_raw,
    }


def joint_search(n_configs, n_qubits=10, dtype_str="complex64", use_gpu=True, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_configs):
        cfg = sample_config(rng)
        try:
            entry = run_one_config(cfg, n_qubits, dtype_str, use_gpu, SEARCH_PROTOCOL)
        except Exception as e:
            entry = {**cfg, "error": str(e)}
        results.append(entry)
        print(f"[{i+1}/{n_configs}]", json.dumps({k: v for k, v in entry.items() if k != "error"} if "error" not in entry else entry))
        if (i + 1) % 20 == 0:
            with open(OUT_PATH, "w") as f:
                json.dump({"stage": "joint_search_checkpoint", "n_done": i + 1, "results": results}, f, indent=2)
    return results


CFG_KEYS = ["gamma1", "gamma2", "a", "tau", "jg_ratio", "V", "washout", "w_in_seed"]


def main():
    all_results = {"stages": {}, "protocols": {"search": SEARCH_PROTOCOL, "full": FULL_PROTOCOL}}

    print("\n=== STAGE 0: precision validation (12q reference config) ===")
    prec = precision_validation(n_qubits=12, n_steps=500, use_gpu=HAS_CUPY)
    all_results["stages"]["precision_validation"] = prec
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    sweep_dtype = "complex64" if prec["pass"] else "complex128"
    print(f"Sweep precision: {sweep_dtype} (reporting precision is always complex128)")

    print("\n=== Estimating search-stage time budget ===")
    n_configs = estimate_search_budget(n_qubits=10, dtype_str=sweep_dtype, use_gpu=HAS_CUPY, target_hours=3.0)

    print(f"\n=== STAGE 1: joint random search (10q, N={n_configs} configs, SEARCH_PROTOCOL) ===")
    search_results = joint_search(n_configs=n_configs, n_qubits=10, dtype_str=sweep_dtype, use_gpu=HAS_CUPY, seed=0)
    all_results["stages"]["joint_search"] = search_results
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    valid = [r for r in search_results if "error" not in r]
    valid.sort(key=lambda r: r["track_nmse"])
    top10 = valid[:10]
    print("\nTop 10 by track_nmse (sweep precision, SEARCH_PROTOCOL):")
    for r in top10:
        print(json.dumps(r))

    # FULL_PROTOCOL (~4200 steps) at complex128 is dramatically more
    # expensive than the SEARCH_PROTOCOL numbers above suggest, especially
    # for V=8 configs -- time ONE FULL_PROTOCOL config for real before
    # committing to running 10 of them, and shrink the count to fit a
    # budget rather than assume. This is the same time-boxing philosophy
    # as estimate_search_budget(), applied here because a single 12-qubit
    # V=8 FULL_PROTOCOL config could otherwise take on the order of a day.
    print("\n=== Timing ONE FULL_PROTOCOL config at 10q before STAGE 2 ===")
    t0 = time.perf_counter()
    probe10 = run_one_config({k: top10[0][k] for k in CFG_KEYS}, n_qubits=10, dtype_str="complex128",
                              use_gpu=HAS_CUPY, protocol=FULL_PROTOCOL)
    probe10_s = time.perf_counter() - t0
    stage2_budget_hours = 6.0
    k_refine = max(3, min(10, int(stage2_budget_hours * 3600 / max(probe10_s, 1.0))))
    print(f"One FULL_PROTOCOL/10q config took {probe10_s:.1f}s; targeting {stage2_budget_hours}h "
          f"-> refining top {k_refine} (spec asked for 10; computed, logged deviation if < 10).")

    print(f"\n=== STAGE 2: top-{k_refine} refine at complex128 (10q, FULL_PROTOCOL) ===")
    refined = [probe10]
    for cfg in top10[1:k_refine]:
        c = {k: cfg[k] for k in CFG_KEYS}
        entry = run_one_config(c, n_qubits=10, dtype_str="complex128", use_gpu=HAS_CUPY, protocol=FULL_PROTOCOL)
        refined.append(entry)
        print(json.dumps(entry))
    all_results["stages"]["topk_refine_complex128"] = refined
    all_results["stages"]["stage2_k_used"] = k_refine
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    refined.sort(key=lambda r: r["track_nmse"])
    top3 = refined[:3]

    print("\n=== Timing ONE FULL_PROTOCOL config at 12q before STAGE 3 ===")
    t0 = time.perf_counter()
    probe12 = run_one_config({k: top3[0][k] for k in CFG_KEYS}, n_qubits=12, dtype_str="complex128",
                              use_gpu=HAS_CUPY, protocol=FULL_PROTOCOL)
    probe12_s = time.perf_counter() - t0
    stage3_budget_hours = 12.0
    k_verify = max(1, min(3, int(stage3_budget_hours * 3600 / max(probe12_s, 1.0))))
    print(f"One FULL_PROTOCOL/12q config took {probe12_s:.1f}s; targeting {stage3_budget_hours}h "
          f"-> verifying top {k_verify} (spec asked for 3; computed, logged deviation if < 3).")

    print(f"\n=== STAGE 3: top-{k_verify} verified at 12 qubits, complex128, FULL_PROTOCOL ===")
    verified12q = [probe12]
    for cfg in top3[1:k_verify]:
        c = {k: cfg[k] for k in CFG_KEYS}
        entry = run_one_config(c, n_qubits=12, dtype_str="complex128", use_gpu=HAS_CUPY, protocol=FULL_PROTOCOL)
        verified12q.append(entry)
        print(json.dumps(entry))
    all_results["stages"]["topk_verified_12q"] = verified12q
    all_results["stages"]["stage3_k_used"] = k_verify
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== STAGE 4: redefined gate -- stability across 3 w_in seeds at winner's config ===")
    winner = min(verified12q, key=lambda r: r["track_nmse"])
    gate_runs = []
    for seed in [1, 2, 3]:
        c = {k: winner[k] for k in CFG_KEYS if k != "w_in_seed"}
        c["w_in_seed"] = seed
        entry = run_one_config(c, n_qubits=12, dtype_str="complex128", use_gpu=HAS_CUPY, protocol=FULL_PROTOCOL)
        gate_runs.append(entry)
        print(json.dumps(entry))

    gate_pass = all(r["delta_nmse_relative"] >= 0.15 for r in gate_runs)
    all_results["stages"]["gate_check"] = {
        "winner_config": {k: winner[k] for k in CFG_KEYS if k != "w_in_seed"},
        "runs_across_seeds": gate_runs,
        "gate_pass": gate_pass,
        "nmse_trajectory": {
            "sprint1_best": 0.945, "sprint2_voided": 0.398, "sprint2_5_valid_protocol": 0.224,
            "sprint2_6_final_track_nmse": winner["track_nmse"],
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGATE {'PASSED' if gate_pass else 'FAILED'}")

    print("\n=== STAGE 5: IPC/MC characterization at winning 12q config (gamma1 grid) ===")
    # Run regardless of gate outcome -- IPC/MC characterizes the reservoir's
    # intrinsic capacity, not the NARMA10 task specifically, and is useful
    # diagnostic data either way (matches Sprint 2/2.5 precedent).
    w_in = np.random.default_rng(winner["w_in_seed"]).uniform(0.5, 1.5, size=12)
    ipc_records = []
    for gamma1 in [0.0, 0.01, 0.03, 0.1, 0.3, 0.5]:
        g = 1.0
        J = winner["jg_ratio"] * g
        res = SequentialReservoir(
            n_qubits=12, tau=winner["tau"], trotter_steps=10, gamma1=gamma1,
            gamma2=winner["gamma2"], J=J, g=g, input_scaling=winner["a"], w_in=w_in,
            multiplexing=1, use_gpu=HAS_CUPY, dtype_str="complex128",
        )
        rng_u = np.random.default_rng(0)
        u_iid = rng_u.standard_normal(200)
        seq = u_iid[:, None] * w_in[None, :]
        feats = res.drive(seq)
        mc_ipc = measure_mc_ipc(u_iid, feats)
        record = {"gamma1": gamma1, **mc_ipc}
        ipc_records.append(record)
        print(json.dumps(record))
    all_results["stages"]["ipc_mc_characterization_12q"] = {
        "winner_config": {k: winner[k] for k in ["a", "gamma2", "tau", "jg_ratio", "w_in_seed"]},
        "records": ipc_records,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll done. Full results in {OUT_PATH} -- please share this file's contents back.")
    return all_results


if __name__ == "__main__":
    main()
