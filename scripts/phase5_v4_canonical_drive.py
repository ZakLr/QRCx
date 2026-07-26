"""FINAL SPRINT Phase 5: v4 (windowed statevector AtmosphericQRC) driven
on the CANONICAL split (data/canonical_seq.npz: train 2019-2022,
val 2023, test 2024), reusing sprint4_v4_gpu_handoff.py's verified
batched-statevector engine (cross-validated against the real PennyLane
circuit to ~1e-8, Sprint 4) at N_QUBITS=20 -- v4's own accepted
qubit count (project invariant: v5 reference=12q density matrix,
v4 may use 20q statevector, these are different architectures by
design, not a contradiction).

Saves raw correlator FEATURES for all three splits (not final ridge
tables) so a single Phase 5 analysis script can combine v4 features,
v5 features (already produced by phase1_v5_canonical_drive.py), and
the classical baselines into one consistent final table.

Real measured cost from Sprint 4 (H200, 20q): 0.125 s/window.
32,854 canonical windows (22143 train + 5269 val + 5442 test) ->
projected ~1.14h. Checked before committing to the full run below.
"""
import json
import time
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False

print(f"HAS_CUPY (GPU visible): {HAS_CUPY}")

INPUT_NPZ = Path("canonical_seq.npz")
N_QUBITS = 20
N_LAYERS = 3
TROTTER_STEPS = 10
DT = 0.1
SEED = 42
BATCH_SIZE = 256
DTYPE = np.complex64
REMAINING_BUDGET_HOURS = 3.0
SAMPLE_WINDOWS = 200
OUT_DIR = Path(".")


def scale_to_pi(x):
    x_min, x_max = x.min(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)
    return np.pi * (x - x_min) / (x_max - x_min + 1e-8)


def draw_tfim_params(n_qubits, seed):
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
    def __init__(self, n_qubits, batch_size, xp):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.batch = batch_size
        self.xp = xp
        self._z_np = [np.array([1.0 if (k >> (n_qubits - 1 - q)) & 1 == 0 else -1.0
                                 for k in range(self.dim)]) for q in range(n_qubits)]
        self.z = [xp.asarray(z) for z in self._z_np]
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        self.pairs = pairs
        self.zz = {(i, j): xp.asarray(self._z_np[i] * self._z_np[j]) for (i, j) in pairs}

    def vacuum_batch(self, batch):
        xp = self.xp
        state = xp.zeros((batch, self.dim), dtype=DTYPE)
        state[:, 0] = 1.0
        return state

    def apply_1q_unitary(self, state, K, q):
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

    def apply_diag_phase(self, state, angles_by_qubit=None, pair_angles_by_pair=None):
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


CHECKPOINT_EVERY_BATCHES = 5


def drive_all_windows(X, engine, h, g, J, pairs, n_qubits, n_layers, trotter_steps, dt, batch_size, label=""):
    n_samples = X.shape[0]
    x_flat = X.reshape(n_samples, -1)
    x_scaled = scale_to_pi(x_flat)
    n_feat = 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2
    ckpt_path = Path(f"phase5_v4_ckpt_{label}_{n_qubits}q.npz")

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
    ckpt_path.unlink(missing_ok=True)
    np.save(OUT_DIR / f"phase5_v4_features_{label}.npy", features.astype(np.float32))
    return features, elapsed


def main():
    npz = np.load(INPUT_NPZ)
    X_train, X_val, X_test = npz["X_train"], npz["X_val"], npz["X_test"]
    print(f"Loaded canonical windows: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    xp = cp if HAS_CUPY else np
    engine = BatchedStatevectorEngine(N_QUBITS, BATCH_SIZE, xp)
    h, g, J = draw_tfim_params(N_QUBITS, SEED)
    pairs = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]

    n_total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    sample = min(SAMPLE_WINDOWS, X_train.shape[0])
    x_sample = scale_to_pi(X_train[:sample].reshape(sample, -1))
    t0 = time.perf_counter()
    for start in range(0, sample, BATCH_SIZE):
        end = min(start + BATCH_SIZE, sample)
        st = encode_and_evolve_batch(x_sample[start:end], engine, h, g, J, pairs,
                                      N_QUBITS, N_LAYERS, TROTTER_STEPS, DT)
        batched_correlators(st, engine, N_QUBITS)
    probe_elapsed = time.perf_counter() - t0
    s_per_window = probe_elapsed / sample
    projected_hours = n_total * s_per_window / 3600
    print(f"\nCost projection: {s_per_window:.5f} s/window -> {projected_hours:.3f}h projected "
          f"for all {n_total} windows. Budget: {REMAINING_BUDGET_HOURS:.2f}h.")
    if projected_hours > REMAINING_BUDGET_HOURS:
        raise RuntimeError(f"Projected cost ({projected_hours:.2f}h) exceeds budget "
                            f"({REMAINING_BUDGET_HOURS:.2f}h). Stopping before committing blind.")

    results = {"n_qubits": N_QUBITS, "n_layers": N_LAYERS, "trotter_steps": TROTTER_STEPS,
               "seed": SEED, "batch_size": BATCH_SIZE, "dtype": str(DTYPE),
               "cost_projection": {"s_per_window": s_per_window, "projected_hours": projected_hours},
               "n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]), "n_test": int(X_test.shape[0]),
               "drives": {}}

    for label, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        print(f"\nDriving {label} windows ({X.shape[0]})...")
        feats, elapsed = drive_all_windows(X, engine, h, g, J, pairs, N_QUBITS, N_LAYERS,
                                            TROTTER_STEPS, DT, BATCH_SIZE, label)
        results["drives"][label] = {"wall_clock_s": elapsed, "s_per_window": elapsed / X.shape[0]}
        with open(OUT_DIR / "phase5_v4_canonical_drive_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\nAll done.")


if __name__ == "__main__":
    main()
