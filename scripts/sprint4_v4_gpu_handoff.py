"""Sprint 4 (REVISED) — v4 GPU handoff: batched statevector simulation of
the windowed AtmosphericQRC circuit (ZZFeatureMap encode + TFIM Trotter
evolution) at the reference 12-qubit config, on GPU.

WHY THIS EXISTS: the real AtmosphericQRC.transform() (QRCx/QRCx/reservoir/
tfim.py, PennyLane lightning.qubit, one qml.QNode call per window) measured
57.5 s/sample on CPU at n_qubits=12 -- ~354h for one seed of the pilot's
16582 train + 5611 eval windows, infeasible locally (see
results/sprint4_pilot_v4.json's misc_log, which used a reduced n_qubits=8 +
subsampled-window CPU run as a stopgap; that result is SUPERSEDED by this
script both because 8q is out of spec (project invariant: 12 reference /
10 fallback only) and because it also used ZZFeatureMap before the bug fix
below). This script reimplements the same circuit as a batched (all
windows processed at once, no per-sample Python-level QNode call)
statevector simulator, GPU-capable via cupy, self-contained (no
QRCx/PennyLane import) so it can run standalone on qBraid.

REAL BUG FOUND AND FIXED WHILE BUILDING THIS SCRIPT (see
docs/sprint_log/SPRINT_4_REPORT.md for the full writeup):
QRCx/QRCx/encoding/zz_feature_map.py's `circuit()` called `qml.apply(op)`
on operators that PennyLane had ALREADY auto-queued at construction time
(since `encode()` builds them inside the active QNode's recording
context) -- so every gate in the ZZFeatureMap encoder was applied TWICE.
For the Hadamards (H*H = I, applied back-to-back with nothing in
between) this silently cancelled them out completely -- the encoder
never actually created superposition from |0...0>; for RZ/IsingZZ every
angle was silently doubled. This was caught by cross-validating this
from-scratch batched reimplementation against the real PennyLane circuit
(AtmosphericQRC.transform()) and finding they disagreed by O(1), then
bisecting gate-by-gate until the double-queuing was found. Fixed in
zz_feature_map.py (`encode()` now builds ops inside
`qml.QueuingManager.stop_recording()` so `circuit()`'s explicit
`qml.apply(op)` is the only queuing event). After the fix, this script's
output matches the real PennyLane AtmosphericQRC.transform() to ~1e-8 at
every n_qubits/n_layers/trotter_steps combination tested (verified
locally at 4 and 5 qubits before this docstring was written; the
handoff/verification is documented, not just asserted).

Unlike v5's recurrent density-matrix reservoir, v4 has NO cross-sample
recurrence -- every window is encoded fresh and independent. This makes
it "embarrassingly batchable": all N windows advance through the exact
same gate sequence together as a (N, 2^n) statevector array, and pure-state
Pauli expectations are O(dim) per operator (not O(dim^2) like v5's density
matrices), so this should be dramatically faster than both the PennyLane
CPU baseline and per-step v5 GPU cost.

CIRCUIT (mirrors QRCx/QRCx/reservoir/tfim.py + QRCx/QRCx/encoding/
zz_feature_map.py exactly):
  1. Flatten each (W=24, d=13) window to a 312-length vector, scale to
     [0, pi] via per-sample min-max (`scale_to_pi`: same formula as
     BaseEncoder.scale_to_pi).
  2. ZZFeatureMap, n_layers=3: per layer -- Hadamard all qubits; RZ(x[i%d])
     all qubits; IsingZZ((pi-x[j%d])(pi-x[k%d])) all pairs j<k.
  3. TFIM Trotter, trotter_steps=10: per step -- RX(2 g_i dt) all qubits;
     RZ(2 h_i dt) all qubits; IsingZZ(2 J_ij dt) all pairs with J_ij != 0.
     h, g, J drawn with the SAME seeded RNG recipe as AtmosphericQRC's
     constructor (seed=42, n_qubits=12) so this reproduces the reference
     config's disorder realization exactly.
  4. Extract single-body <X_i>,<Y_i>,<Z_i> and two-body <P_i P_j> (P in
     X,Y,Z) correlators from the batched statevector -- same 3N+3C(N,2)=234
     feature convention as extract_correlators.

REQUIRED INPUT FILE: data/sprint4_pilot_seq.npz (from
scripts/sprint4_export_pilot_seq.py) -- needs its X_train/y_train/X_val/
y_val keys (added alongside train_seq/val_seq specifically for this script).

Usage in qBraid Lab:
    !python sprint4_v4_gpu_handoff.py > sprint4_v4_log.txt 2>&1 &
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

try:
    import cupy as cp
    HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False

print(f"HAS_CUPY (GPU visible): {HAS_CUPY}")

INPUT_NPZ = Path("data/sprint4_pilot_seq.npz")
N_QUBITS = 20  # bumped again after 16q measured ~120s total on H200 (real, measured, results/sprint4_v4_gpu_results_16q.json) -- exploratory, out-of-spec (project invariant is 12 reference/10 fallback); flagged, not silent
OUT_PATH = Path(f"sprint4_v4_gpu_results_{N_QUBITS}q.json")  # qubit-count-specific filename so successive N_QUBITS bumps don't clobber earlier real results
N_LAYERS = 3
TROTTER_STEPS = 10
DT = 0.1
SEED = 42
HORIZONS = [1, 3, 6, 12]
REMAINING_BUDGET_HOURS = 6.0  # cost-projection preflight gate -- set before running; at 12q this was wildly conservative (real run was 16s), but scaling to n_qubits=16/20 is genuinely uncertain, so this guards against a bad extrapolation
SAMPLE_WINDOWS = 200
BATCH_SIZE = 256  # windows processed together per GPU batch (memory: BATCH_SIZE * 2^12 complex64 ~ 33.5MB/batch at complex64)
DTYPE = np.complex64
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def scale_to_pi(x):
    """Per-sample min-max scale to [0, pi] -- matches BaseEncoder.scale_to_pi exactly."""
    x_min, x_max = x.min(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)
    return np.pi * (x - x_min) / (x_max - x_min + 1e-8)


def draw_tfim_params(n_qubits, seed):
    """Reproduces AtmosphericQRC.__init__'s exact RNG recipe (tfim.py)."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.0, 1.0, size=n_qubits)
    g = rng.uniform(0.5, 1.5, size=n_qubits)
    J_raw = rng.uniform(-3.0, 3.0, size=(n_qubits, n_qubits))
    J = (J_raw + J_raw.T) / 2.0
    np.fill_diagonal(J, 0.0)
    mean_abs_J = np.mean(np.abs(J[J != 0]))
    mean_g = np.mean(g)
    raw_ratio = mean_abs_J / mean_g if mean_g != 0 else 0.0
    if raw_ratio > 0:
        J = J * (1.0 / raw_ratio)
    return h, g, J


class BatchedStatevectorEngine:
    """Batched (all samples at once) pure-state simulator for a fixed
    single/two-qubit diagonal-or-simple gate sequence. State shape:
    (batch, 2**n_qubits). Single-qubit non-diagonal gates (H, RX) applied
    via reshape+moveaxis+matmul (same trick as sprint4_v5_gpu_handoff.py's
    density-matrix Kraus application, one tensor axis lower since this is
    kets not density matrices). Diagonal gates (RZ, IsingZZ) applied as an
    elementwise phase computed via a batched linear combination of
    precomputed +-1 per-basis-state sign vectors -- avoids ever
    materializing a full n-qubit operator."""

    def __init__(self, n_qubits, batch_size, xp):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.batch = batch_size
        self.xp = xp
        self._z_np = [np.array([1.0 if (k >> (n_qubits - 1 - q)) & 1 == 0 else -1.0
                                 for k in range(self.dim)]) for q in range(n_qubits)]
        self.z = [xp.asarray(z) for z in self._z_np]  # each (dim,)
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        self.pairs = pairs
        self.zz = {(i, j): xp.asarray(self._z_np[i] * self._z_np[j]) for (i, j) in pairs}

    def vacuum_batch(self, batch):
        xp = self.xp
        state = xp.zeros((batch, self.dim), dtype=DTYPE)
        state[:, 0] = 1.0
        return state

    def apply_1q_unitary(self, state, K, q):
        """K: (2,2) complex, same for the whole batch."""
        xp = self.xp
        n, batch = self.n, state.shape[0]
        s = state.reshape([batch] + [2] * n)
        s = xp.moveaxis(s, q + 1, 1)
        shape = s.shape
        s2 = s.reshape(batch, 2, -1)
        out = xp.einsum("ab,nbx->nax", xp.asarray(K, dtype=DTYPE), s2)
        out = out.reshape(shape)
        out = xp.moveaxis(out, 1, q + 1)
        return out.reshape(batch, self.dim)

    def apply_diag_phase(self, state, angles_by_qubit=None, angle_zz=None, pair_angles_by_pair=None):
        """angles_by_qubit: dict q -> (batch,) array of RZ-equivalent
        angles (phase = exp(-i*theta/2*Z_q)); pair_angles_by_pair: dict
        (i,j) -> (batch,) array of IsingZZ angles
        (phase = exp(-i*phi/2*Z_i Z_j)). All combined additively in the
        exponent before a single exp() (batch, dim)."""
        xp = self.xp
        batch = state.shape[0]
        total = xp.zeros((batch, self.dim), dtype=xp.float32 if DTYPE == np.complex64 else xp.float64)
        if angles_by_qubit:
            for q, theta in angles_by_qubit.items():
                total = total + theta[:, None] * self.z[q][None, :]
        if pair_angles_by_pair:
            for (i, j), phi in pair_angles_by_pair.items():
                total = total + phi[:, None] * self.zz[(i, j)][None, :]
        phase = xp.exp(-1j * total / 2.0).astype(DTYPE)
        return state * phase


H_GATE = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def rx_matrix(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def encode_and_evolve_batch(x_batch, engine, h, g, J, pairs, n_qubits, n_layers, trotter_steps, dt):
    """x_batch: (batch, W*d) already scale_to_pi'd. Returns final batched
    statevector (batch, 2**n)."""
    xp = engine.xp
    batch = x_batch.shape[0]
    d = x_batch.shape[1]
    state = engine.vacuum_batch(batch)
    x_gpu = xp.asarray(x_batch)

    for _ in range(n_layers):
        for i in range(n_qubits):
            state = engine.apply_1q_unitary(state, H_GATE, i)
        rz_angles = {i: x_gpu[:, i % d] for i in range(n_qubits)}
        state = engine.apply_diag_phase(state, angles_by_qubit=rz_angles)
        pair_angles = {}
        for (j, k) in pairs:
            phi = (np.pi - x_gpu[:, j % d]) * (np.pi - x_gpu[:, k % d])
            pair_angles[(j, k)] = phi
        state = engine.apply_diag_phase(state, pair_angles_by_pair=pair_angles)

    for _ in range(trotter_steps):
        for i in range(n_qubits):
            state = engine.apply_1q_unitary(state, rx_matrix(2.0 * g[i] * dt), i)
        rz_angles = {i: xp.full((batch,), 2.0 * h[i] * dt) for i in range(n_qubits)}
        state = engine.apply_diag_phase(state, angles_by_qubit=rz_angles)
        pair_angles = {}
        for (i, j) in pairs:
            if J[i, j] != 0:
                pair_angles[(i, j)] = xp.full((batch,), 2.0 * J[i, j] * dt)
        if pair_angles:
            state = engine.apply_diag_phase(state, pair_angles_by_pair=pair_angles)

    return state


def batched_correlators(state, engine, n_qubits):
    """<X_i>,<Y_i>,<Z_i> and <P_iP_j> (P in X,Y,Z) for a batch of pure
    states, via reshape+moveaxis (no full n-qubit operator ever built).
    Returns (batch, 3n+3*C(n,2)) real array."""
    xp = engine.xp
    batch = state.shape[0]
    n = n_qubits
    n_pairs = n * (n - 1) // 2
    n_feat = 3 * n + 3 * n_pairs
    out = xp.zeros((batch, n_feat), dtype=xp.float32)
    PAULIS_1Q = {
        "X": xp.asarray(np.array([[0, 1], [1, 0]], dtype=np.complex128), dtype=DTYPE),
        "Y": xp.asarray(np.array([[0, -1j], [1j, 0]], dtype=np.complex128), dtype=DTYPE),
        "Z": xp.asarray(np.array([[1, 0], [0, -1]], dtype=np.complex128), dtype=DTYPE),
    }
    # Single-body block ordered [X_0..X_{n-1}, Y_0..Y_{n-1}, Z_0..Z_{n-1}]
    # -- matches QRCx.readout.correlators.single_body's index convention
    # (p_idx*n_qubits + i), NOT a per-qubit [X_i,Y_i,Z_i] grouping.
    s_r = state.reshape([batch] + [2] * n)
    for p_idx, P in enumerate(("X", "Y", "Z")):
        for q in range(n):
            s_q = xp.moveaxis(s_r, q + 1, 1).reshape(batch, 2, -1)
            Ps = xp.einsum("ab,nbx->nax", PAULIS_1Q[P], s_q)
            val = xp.real(xp.sum(xp.conj(s_q) * Ps, axis=(1, 2)))
            out[:, p_idx * n + q] = val
    idx = 3 * n
    for i in range(n):
        for j in range(i + 1, n):
            others = [k for k in range(n) if k not in (i, j)]
            perm = [0, i + 1, j + 1] + [k + 1 for k in others]
            s_ij = xp.transpose(s_r, perm).reshape(batch, 2, 2, -1)
            for P in ("Z", "X", "Y"):
                Ps = xp.einsum("ab,nbcx->nacx", PAULIS_1Q[P], s_ij)
                Ps = xp.einsum("cd,nadx->nacx", PAULIS_1Q[P], Ps)
                val = xp.real(xp.sum(xp.conj(s_ij) * Ps, axis=(1, 2, 3)))
                out[:, idx] = val
                idx += 1
    return xp.asnumpy(out) if HAS_CUPY and engine.xp is cp else out


CHECKPOINT_EVERY_BATCHES = 5  # cheap insurance against the instance's ~45min forced stop (observed twice, cause unconfirmed) -- resuming loses at most a few batches, not the whole drive


def drive_all_windows(X, engine, h, g, J, pairs, n_qubits, n_layers, trotter_steps, dt, batch_size, label=""):
    n_samples = X.shape[0]
    x_flat = X.reshape(n_samples, -1)
    x_scaled = scale_to_pi(x_flat)
    n_feat = 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2
    ckpt_path = Path(f"sprint4_v4_ckpt_{label}_{n_qubits}q.npz")

    if ckpt_path.exists():
        with np.load(ckpt_path) as ckpt:
            features = ckpt["features"].copy()
            done = int(ckpt["done"])
            elapsed_prior = float(ckpt["elapsed"])
        print(f"  [{label}] resuming from checkpoint: {done}/{n_samples} windows already done "
              f"({elapsed_prior:.1f}s prior wall-clock).")
    else:
        features = np.zeros((n_samples, n_feat), dtype=np.float64)
        done = 0
        elapsed_prior = 0.0

    t0 = time.perf_counter()
    for start in range(done, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        state = encode_and_evolve_batch(x_scaled[start:end], engine, h, g, J, pairs,
                                         n_qubits, n_layers, trotter_steps, dt)
        features[start:end] = batched_correlators(state, engine, n_qubits)
        done = end
        batch_idx = start // batch_size
        if batch_idx % CHECKPOINT_EVERY_BATCHES == 0:
            elapsed = elapsed_prior + (time.perf_counter() - t0)
            rate = elapsed / done
            np.savez(ckpt_path, features=features, done=done, elapsed=elapsed)
            print(f"    [{label}] {done}/{n_samples} windows "
                  f"({rate:.5f} s/window, eta {(n_samples - done) * rate:.1f}s) -- checkpoint saved")
    elapsed = elapsed_prior + (time.perf_counter() - t0)
    print(f"  [{label}] done: {n_samples} windows in {elapsed:.1f}s ({elapsed / n_samples:.5f} s/window)")
    ckpt_path.unlink(missing_ok=True)  # drive fully complete -- checkpoint no longer needed
    return features, elapsed


def fit_eval_ridge(X_fit, y_fit, X_val_tune, y_val_tune, X_eval, y_eval):
    best_alpha, best_score = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha).fit(X_fit, y_fit)
        score = float(np.mean((y_val_tune - m.predict(X_val_tune)) ** 2))
        if score < best_score:
            best_score, best_alpha = score, alpha
    final_model = Ridge(alpha=best_alpha).fit(np.concatenate([X_fit, X_val_tune]),
                                               np.concatenate([y_fit, y_val_tune]))
    return final_model.predict(X_eval), best_alpha


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def skill(y, p, y_persist):
    rp, rpe = rmse(y, p), rmse(y, y_persist)
    return float(1.0 - rp / rpe) if rpe > 0 else float("nan")


def diebold_mariano_vs_persistence(y_true, y_pred, y_persist, h):
    """Squared-error DM test, Bartlett/Newey-West HAC (maxlags=h-1) --
    duplicated from sprint4_v5_gpu_handoff.py (not imported, so this
    script stays independently runnable if uploaded alone) -- ported from
    QRCx.metrics.significance.diebold_mariano (verified there against
    statsmodels to rel=1e-8, Sprint 3)."""
    e_model = y_true - y_pred
    e_base = y_true - y_persist
    d = e_model ** 2 - e_base ** 2
    n = len(d)
    maxlags = max(h - 1, 0)
    dbar = float(d.mean())
    dc = d - dbar
    gamma0 = np.sum(dc ** 2) / n
    var = gamma0
    for lag in range(1, maxlags + 1):
        w = 1.0 - lag / (maxlags + 1)
        cov = np.sum(dc[lag:] * dc[:-lag]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    dm_stat = 0.0 if se == 0.0 else dbar / se
    from scipy import stats as _stats
    p_value = float(2.0 * (1.0 - _stats.norm.cdf(abs(dm_stat))))
    return {"dm_stat": float(dm_stat), "p_value": p_value, "n": int(n), "maxlags": int(maxlags)}


def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"{INPUT_NPZ} not found -- run scripts/sprint4_export_pilot_seq.py locally first.")
    npz = np.load(INPUT_NPZ)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val, y_val = npz["X_val"], npz["y_val"]
    target_col_idx = int(npz["target_col_idx"])
    if "horizons" in npz:
        npz_horizons = [int(h) for h in npz["horizons"]]
        assert npz_horizons == HORIZONS, (
            f"data/sprint4_pilot_seq.npz was exported with horizons={npz_horizons} but this "
            f"script expects HORIZONS={HORIZONS} -- re-run scripts/sprint4_export_pilot_seq.py "
            f"and re-upload, or the y_train/y_val columns will be silently mismatched to the wrong horizon."
        )
    else:
        raise AssertionError(
            "data/sprint4_pilot_seq.npz has no 'horizons' key (stale export) -- "
            "re-run scripts/sprint4_export_pilot_seq.py and re-upload before continuing "
            f"(y_train/y_val must have exactly {len(HORIZONS)} columns matching HORIZONS={HORIZONS})."
        )
    assert y_train.shape[1] == len(HORIZONS), \
        f"y_train has {y_train.shape[1]} columns, expected {len(HORIZONS)} (HORIZONS={HORIZONS})"
    print(f"Loaded windowed pilot data: X_train={X_train.shape}, X_val={X_val.shape}, "
          f"target_col_idx={target_col_idx}, horizons={npz_horizons}")

    xp = cp if HAS_CUPY else np
    engine = BatchedStatevectorEngine(N_QUBITS, BATCH_SIZE, xp)
    h, g, J = draw_tfim_params(N_QUBITS, SEED)
    n_pairs = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]

    # --- Cost-projection preflight: measure real s/window on a small
    # sample at N_QUBITS before committing to the full ~22k-window pilot.
    # Cheap insurance now that N_QUBITS is being pushed past the 12q
    # reference config into genuinely-uncertain scaling territory.
    n_total = X_train.shape[0] + X_val.shape[0]
    sample = min(SAMPLE_WINDOWS, X_train.shape[0])
    x_sample = X_train[:sample].reshape(sample, -1)
    x_sample_scaled = scale_to_pi(x_sample)
    t0 = time.perf_counter()
    for start in range(0, sample, BATCH_SIZE):
        end = min(start + BATCH_SIZE, sample)
        st = encode_and_evolve_batch(x_sample_scaled[start:end], engine, h, g, J, n_pairs,
                                      N_QUBITS, N_LAYERS, TROTTER_STEPS, DT)
        batched_correlators(st, engine, N_QUBITS)
    probe_elapsed = time.perf_counter() - t0
    s_per_window = probe_elapsed / sample
    projected_hours = n_total * s_per_window / 3600
    print(f"\nCost projection (n_qubits={N_QUBITS}, sampled {sample} windows): "
          f"{s_per_window:.5f} s/window -> {projected_hours:.3f}h projected for all "
          f"{n_total} windows. Budget: {REMAINING_BUDGET_HOURS:.2f}h.")
    if projected_hours > REMAINING_BUDGET_HOURS:
        raise RuntimeError(
            f"Projected cost ({projected_hours:.2f}h) exceeds REMAINING_BUDGET_HOURS "
            f"({REMAINING_BUDGET_HOURS:.2f}h) at n_qubits={N_QUBITS}. Stopping before the "
            f"full run rather than committing blind -- raise REMAINING_BUDGET_HOURS if this "
            f"is expected/acceptable, or reduce N_QUBITS."
        )

    print(f"\nDriving train windows ({X_train.shape[0]}) at n_qubits={N_QUBITS}...")
    feats_train, t_train = drive_all_windows(X_train, engine, h, g, J, n_pairs,
                                              N_QUBITS, N_LAYERS, TROTTER_STEPS, DT, BATCH_SIZE, "train")
    print(f"\nDriving val windows ({X_val.shape[0]})...")
    feats_val, t_val = drive_all_windows(X_val, engine, h, g, J, n_pairs,
                                          N_QUBITS, N_LAYERS, TROTTER_STEPS, DT, BATCH_SIZE, "val")

    results = {
        "cost_projection": {"s_per_window": s_per_window, "projected_hours": projected_hours,
                             "sample_windows": sample, "budget_hours": REMAINING_BUDGET_HOURS},
        "n_qubits": N_QUBITS, "n_layers": N_LAYERS, "trotter_steps": TROTTER_STEPS,
        "seed": SEED, "batch_size": BATCH_SIZE, "dtype": str(DTYPE),
        "n_train_windows": int(X_train.shape[0]), "n_val_windows": int(X_val.shape[0]),
        "wall_clock_train_s": t_train, "wall_clock_val_s": t_val,
        "s_per_window": (t_train + t_val) / (X_train.shape[0] + X_val.shape[0]),
        "horizons": HORIZONS, "cells": [],
    }

    for h_idx, horizon in enumerate(HORIZONS):
        for architecture in ("direct", "residual"):
            y_tr_full = y_train[:, h_idx]
            y_v_full = y_val[:, h_idx]
            y_tr_pers = X_train[:, -1, target_col_idx]
            y_v_pers = X_val[:, -1, target_col_idx]
            y_tr_reg = (y_tr_full - y_tr_pers) if architecture == "residual" else y_tr_full
            y_v_reg = (y_v_full - y_v_pers) if architecture == "residual" else y_v_full

            n_val_tune = int(0.2 * len(feats_train))
            X_fit, X_vt = feats_train[:-n_val_tune], feats_train[-n_val_tune:]
            y_fit, y_vt = y_tr_reg[:-n_val_tune], y_tr_reg[-n_val_tune:]

            pred_reg, alpha = fit_eval_ridge(X_fit, y_fit, X_vt, y_vt, feats_val, y_v_reg)
            pred_true = pred_reg + y_v_pers if architecture == "residual" else pred_reg

            cell = {
                "architecture": architecture, "horizon": int(horizon),
                "rmse": rmse(y_v_full, pred_true),
                "skill_vs_persistence": skill(y_v_full, pred_true, y_v_pers),
                "dm_vs_persistence": diebold_mariano_vs_persistence(y_v_full, pred_true, y_v_pers, horizon),
                "best_alpha": alpha,
            }
            results["cells"].append(cell)
            print(f"  [{architecture} h={horizon}] skill_vs_persistence="
                  f"{cell['skill_vs_persistence'] * 100:.2f}%  rmse={cell['rmse']:.4f}")

        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nAll done. Full results in {OUT_PATH} -- please share this file's contents back.")
    return results


if __name__ == "__main__":
    main()
