"""Sprint 4 (REVISED) — v5 GPU handoff: drive the real pilot weather
sequence (train 2019-2021 / eval 2022, 3 years / 1 year, real KORD ISD
data) through the sequential dissipative reservoir on GPU, and produce
every ablation-matrix cell's forecast metrics from the driven features.

Self-contained (no QRCx import), same convention as sprint26_pipeline.py
/ gpu_verify_standalone.py, so it can be pasted into (or uploaded and
imported into) a qBraid Lab GPU notebook with no package install step.
Requires numpy, scikit-learn, and (for GPU) cupy — falls back to
numpy/CPU if cupy isn't visible, but at 26,304+8,760 = ~35,000 real
hourly steps this is designed to run on GPU.

REQUIRED INPUT FILE (upload alongside this script, same directory):
    data/sprint4_pilot_seq.npz
  produced locally by scripts/sprint4_export_pilot_seq.py, which calls
  the real QRCx.data.preprocessor.preprocess() (13 engineered features,
  climatological-anomaly residual, StandardScaler fit on train only) on
  the pilot split (train 2019-2021, eval 2022) — a nested sub-split of
  the canonical locked train block, never touching val=2023/test=2024.

TRAJECTORY-CACHING DESIGN (the key cost-saving idea communicated to the
user before this script was written): the ablation matrix is 2 gamma1
values (tuned=0.03 vs a 0.0 "dissipation-off" diagnostic) x 2
multiplexing-V values (1, 4) x 2 architectures (direct, residual) x 4
horizons = 32 (architecture, horizon) cells, but architecture and horizon
are *readout-level* decisions made on top of already-driven reservoir
features — they do not require re-driving the quantum reservoir. So this
script drives the reservoir only
    2 (gamma1: tuned vs diagnostic_off) x 2 (V: 1 vs 4) x 3 (w_in seeds,
    for a stability check) = 12 unique drives
each producing a feature trajectory. NOTE: train_seq and val_seq are
driven as two SEPARATE passes (each starting from the vacuum state, not
one continuous pass carrying state across the train/val boundary) --
this is a real simplification versus true single-continuous-trajectory
deployment, made to keep this script's cost/time projection simple and
because `washout` (200 steps into val_seq's own transient) already
absorbs most of val_seq's own settling-in period; flagged explicitly here
so it can be revisited (concatenate train_seq+val_seq and slice) if the
paper needs the fully continuous version. All (architecture, horizon)
cells are obtained as independent Ridge readout fits on top of the 12
cached feature sets — cheap, CPU-side, no additional GPU time.

COST-PROJECTION PREFLIGHT (same lesson as sprint26_stage3plus.py): before
committing to any full-length drive, this script times SAMPLE_STEPS on
the real GPU and extrapolates. If a drive's projected cost exceeds
REMAINING_BUDGET_HOURS (set this before running), it does NOT run blind
-- it falls back to driving only the most recent TRAIN_STEPS_FALLBACK
steps of train_seq (still real data, just a shorter window) and logs the
reduction honestly in the output JSON, rather than hanging for an
unknown/possibly multi-day duration.

Usage in qBraid Lab:
    # set REMAINING_BUDGET_HOURS below to your actual available time first
    !python sprint4_v5_gpu_handoff.py > sprint4_v5_log.txt 2>&1 &
    # background process, not a blocking notebook cell (lesson from the
    # Sprint 2.6 laptop-sleep incident: this must survive independently
    # of any client connection). Poll sprint4_v5_results.json for
    # per-drive progress (written incrementally after each of the 12
    # drives, not only at the end) and paste its contents back when done.
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
OUT_PATH = Path("sprint4_v5_results.json")
N_QUBITS = 12  # reference config, GPU-conditional (results/gpu_verification.json)
DTYPE_STR = "complex64"  # sweep precision; verified stable to <1e-5 rel err at 12q for the search analog (Sprint 2.6 precision_validation) -- if a future run flags high error here, fall back to complex128 (slower) and log it, do not silently trust complex64
WASHOUT = 200
HORIZONS = [1, 3, 6, 12]
# Scope reduced (2026-07-24, user-approved time-box): 3 seeds -> 1 (drops the
# seed-stability check), V in {1,4} -> {1} only (drops the multiplexed-readout
# ablation cell -- V=4 costs ~4x V=1 per step, the single biggest cost driver),
# and REMAINING_BUDGET_HOURS set low enough to force the existing
# TRAIN_STEPS_FALLBACK (1yr instead of 3yr train) via the cost-projection
# preflight below. Real measured ~0.36 s/step (V=1, 12q, complex64, H200,
# post-throughput-fix) projects this to ~3.5h total (2 drives x ~1.75h) --
# down from an unworkable ~42h+ for the original full 12-drive/3yr plan.
# Honestly logged here, not silently descoped -- see docs/sprint_log/SPRINT_4_REPORT.md.
W_IN_SEEDS = [123]  # 123 = the frozen config's seed (configs/v5_reference.yaml)
GAMMA1_VALUES = {"tuned": 0.03, "diagnostic_off": 0.0}
GAMMA2 = 0.1
REMAINING_BUDGET_HOURS = 4.0  # <-- forces the 1yr-train fallback below via the preflight; raise if you have more time
SAMPLE_STEPS = 300
TRAIN_STEPS_FALLBACK = 8760  # 1 year, if the full 3-year drive doesn't fit budget
V_VALUES = [1]
TAU = 1.0
INPUT_SCALING = 0.3
J, G = 1.0, 1.0
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


# ============================================================
# Reservoir engine (mirrors QRCx.reservoir.sequential.SequentialDissipativeQRC
# / scripts/sprint26_pipeline.py's SequentialReservoir, validated bit-identical
# to the real package in Sprint 2.6 — see that script's docstring)
# ============================================================

class DensityOps:
    def __init__(self, n_qubits, use_gpu, dtype_str):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.xp = cp if (use_gpu and HAS_CUPY) else np
        self.on_gpu = use_gpu and HAS_CUPY
        self.dtype = getattr(self.xp, dtype_str)
        self.float_dtype = self.xp.float32 if dtype_str == "complex64" else self.xp.float64
        self._z_np = [np.array([1.0 if (k >> (n_qubits - 1 - q)) & 1 == 0 else -1.0
                                 for k in range(self.dim)]) for q in range(n_qubits)]
        self._z = [self.xp.asarray(z, dtype=self.float_dtype) for z in self._z_np]

    def asarray(self, a):
        return self.xp.asarray(a, dtype=self.dtype)

    def to_numpy(self, a):
        return cp.asnumpy(a) if self.on_gpu else a

    def vacuum(self):
        rho = self.xp.zeros((self.dim, self.dim), dtype=self.dtype)
        rho[0, 0] = 1.0
        return rho

    def conjugate_dense(self, rho, U):
        return U @ rho @ U.conj().T

    def conjugate_1q(self, rho, K, q):
        return _conjugate_1q_direct(self.xp, rho, K, q, self.n, self.dim)

    def dephasing_coeff(self, gamma2):
        n = self.n
        z_list = self._z
        coeff = self.xp.ones((self.dim, self.dim), dtype=self.float_dtype)
        lam = 1.0 - gamma2
        for q in range(n):
            same = (z_list[q][:, None] * z_list[q][None, :]) > 0
            coeff = self.xp.where(same, coeff, coeff * lam)
        return coeff.astype(self.dtype)


def _conjugate_1q_direct(xp, rho, K, q, n, dim):
    """K rho K^dagger acting on qubit q only, via explicit einsum (kept
    simple/obviously-correct rather than the fastest possible path — this
    script's cost is dominated by the coherent-evolution matmul and
    correlator extraction, not single-qubit Kraus application)."""
    rho_r = rho.reshape([2] * (2 * n))
    axes = list(range(2 * n))
    # bring axis q (ket) and n+q (bra) to front
    rho_r = xp.moveaxis(rho_r, [q, n + q], [0, 1])
    rho_r = rho_r.reshape(2, 2, -1)
    tmp = xp.einsum("ab,bcx->acx", K, rho_r)
    tmp = xp.einsum("acx,dc->adx", tmp, K.conj())
    tmp = tmp.reshape([2, 2] + [2] * (2 * n - 2))
    tmp = xp.moveaxis(tmp, [0, 1], [q, n + q])
    return tmp.reshape(dim, dim)


def ry_matrix(theta, xp=np):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _build_tfim_full(n_qubits, J, g):
    """Standard TFIM: H = -J sum ZZ (nearest-neighbor ring) - g sum X,
    built explicitly and exactly (small n_qubits, done once per drive)."""
    dim = 2 ** n_qubits
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    def kron_at(op, q):
        mats = [I] * n_qubits
        mats[q] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = np.zeros((dim, dim))
    for q in range(n_qubits):
        H += -g * kron_at(X, q)
    for q in range(n_qubits):
        q2 = (q + 1) % n_qubits
        mats = [I] * n_qubits
        mats[q] = Z
        mats[q2] = Z
        term = mats[0]
        for m in mats[1:]:
            term = np.kron(term, m)
        H += -J * term
    return H


def ptrace_1q_np(rho, qubit, n_qubits):
    n = n_qubits
    rho_r = rho.reshape([2] * (2 * n))
    perm = [qubit] + [i for i in range(n) if i != qubit] + [qubit + n] + [i + n for i in range(n) if i != qubit]
    rho_p = np.transpose(rho_r, perm)
    rho_p = rho_p.reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))
    return np.trace(rho_p, axis1=1, axis2=3)


def ptrace_2q_np(rho, q1, q2, n_qubits):
    n = n_qubits
    rho_r = rho.reshape([2] * (2 * n))
    others = [i for i in range(n) if i not in (q1, q2)]
    perm = [q1, q2] + others + [q1 + n, q2 + n] + [o + n for o in others]
    rho_p = np.transpose(rho_r, perm)
    rho_p = rho_p.reshape(4, 2 ** (n - 2), 4, 2 ** (n - 2))
    return np.trace(rho_p, axis1=1, axis2=3)


PAULIS = {
    "I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.array([[1, 0], [0, -1]]),
}


def extract_correlators_dm(rho_np, n_qubits):
    """CPU (numpy) reference implementation -- kept for correctness
    cross-checks; NOT used in the hot loop (see extract_correlators_dm_xp,
    which stays on GPU and is what drive() actually calls)."""
    n = n_qubits
    n_pairs = n * (n - 1) // 2
    n_feat = 3 * n + 3 * n_pairs
    out = np.zeros(n_feat)
    idx = 0
    for q in range(n):
        dm = ptrace_1q_np(rho_np, q, n)
        for P in ("X", "Y", "Z"):
            out[idx] = np.real(np.trace(dm @ PAULIS[P]))
            idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            dm2 = ptrace_2q_np(rho_np, i, j, n)
            for P in ("X", "Y", "Z"):
                PP = np.kron(PAULIS[P], PAULIS[P])
                out[idx] = np.real(np.trace(dm2 @ PP))
                idx += 1
    return out


def _ptrace_1q_xp(rho, qubit, n_qubits, xp):
    n = n_qubits
    rho_r = rho.reshape([2] * (2 * n))
    perm = [qubit] + [i for i in range(n) if i != qubit] + [qubit + n] + [i + n for i in range(n) if i != qubit]
    rho_p = xp.transpose(rho_r, perm)
    rho_p = rho_p.reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))
    return xp.trace(rho_p, axis1=1, axis2=3)


def _ptrace_2q_xp(rho, q1, q2, n_qubits, xp):
    n = n_qubits
    rho_r = rho.reshape([2] * (2 * n))
    others = [i for i in range(n) if i not in (q1, q2)]
    perm = [q1, q2] + others + [q1 + n, q2 + n] + [o + n for o in others]
    rho_p = xp.transpose(rho_r, perm)
    rho_p = rho_p.reshape(4, 2 ** (n - 2), 4, 2 ** (n - 2))
    return xp.trace(rho_p, axis1=1, axis2=3)


def extract_correlators_dm_xp(rho, n_qubits, xp, paulis_xp):
    """GPU-native correlator extraction -- stays entirely on `xp` (cupy
    when available) until the very last step, when the small (n_feat,)
    output vector (not the full density matrix) is pulled to host. Fixes
    a real throughput bug: the original extract_correlators_dm forced a
    full density-matrix GPU->CPU transfer (and pure-numpy ptrace) EVERY
    single step, which -- found by direct observation on the qBraid H200
    -- made v5 unable to complete even 500 steps within a ~45-minute
    session (the instance's observed forced-stop interval). Verified
    numerically identical (rel err ~1e-6, complex64) to extract_correlators_dm
    before this replaced it in the hot loop -- see docs/sprint_log/
    SPRINT_4_REPORT.md."""
    n = n_qubits
    n_pairs = n * (n - 1) // 2
    n_feat = 3 * n + 3 * n_pairs
    out = xp.zeros(n_feat, dtype=xp.float32)
    idx = 0
    for q in range(n):
        dm = _ptrace_1q_xp(rho, q, n, xp)
        for P in ("X", "Y", "Z"):
            out[idx] = xp.real(xp.trace(dm @ paulis_xp[P]))
            idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            dm2 = _ptrace_2q_xp(rho, i, j, n, xp)
            for P in ("X", "Y", "Z"):
                out[idx] = xp.real(xp.trace(dm2 @ paulis_xp[f"{P}{P}"]))
                idx += 1
    return (cp.asnumpy(out) if (HAS_CUPY and xp is cp) else out).astype(np.float64)


class SequentialReservoir:
    def __init__(self, n_qubits, tau, gamma1, gamma2, J, g, input_scaling,
                 w_in, multiplexing, use_gpu, dtype_str):
        self.n_qubits = n_qubits
        self.tau = tau
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.input_scaling = input_scaling
        self.w_in = w_in
        self.multiplexing = multiplexing
        self.ops = DensityOps(n_qubits, use_gpu, dtype_str)

        _P1 = {"X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
               "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
               "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128)}
        self._paulis_xp = {P: self.ops.asarray(m) for P, m in _P1.items()}
        for P in ("X", "Y", "Z"):
            self._paulis_xp[f"{P}{P}"] = self.ops.asarray(np.kron(_P1[P], _P1[P]))

        self._precompute_fast_extraction_indices()

        H = _build_tfim_full(n_qubits, J, g)
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

    def _precompute_fast_extraction_indices(self):
        """Precomputes gather-index arrays (independent of rho) for a
        transpose-free correlator extraction. Real throughput bug found on
        the qBraid H200: the original extract_correlators_dm_xp path did
        ~700+ individual small GPU calls per step (a transpose+trace per
        qubit and per qubit-pair) -- with real per-call dispatch overhead
        on the order of 100-300us each, this dominated the observed
        ~0.36 s/step, far more than the actual FLOPs required. This
        precomputes index arrays once (at construction) so the hot loop
        can replace those ~700 calls with a handful of large vectorized
        gathers instead (same math -- Pauli expectations from a density
        matrix expressed as diagonal sums and off-diagonal element sums,
        not partial traces via reshape+transpose). Verified numerically
        identical to extract_correlators_dm_xp (itself already verified
        against extract_correlators_dm, the CPU reference) before this
        replaced it in the hot loop -- see docs/sprint_log/SPRINT_4_REPORT.md.
        """
        n = self.n_qubits
        dim = self.ops.dim
        xp = self.ops.xp
        all_k = np.arange(dim)

        def mask(q):
            return 1 << (n - 1 - q)

        # Single-qubit: <Z_q> from the diagonal (matmul with sign matrix);
        # <X_q>,<Y_q> from S_q = sum_{k: bit_q(k)=0} rho[k, k^mask_q].
        rows1 = np.stack([all_k[(all_k & mask(q)) == 0] for q in range(n)])  # (n, dim/2)
        masks1 = np.array([mask(q) for q in range(n)])
        cols1 = rows1 ^ masks1[:, None]
        self._rows1 = xp.asarray(rows1)
        self._cols1 = xp.asarray(cols1)
        Zmat = np.stack([self.ops._z_np[q] for q in range(n)])  # (n, dim)
        self._Zmat = xp.asarray(Zmat, dtype=self.ops.float_dtype)

        # Two-qubit pairs (same iteration order as the CPU reference:
        # i ascending outer, j ascending inner from i+1).
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self._pairs = pairs
        dim4 = dim // 4
        rowsT1 = np.zeros((len(pairs), dim4), dtype=np.int64)
        rowsT2 = np.zeros((len(pairs), dim4), dtype=np.int64)
        maskcols = np.zeros(len(pairs), dtype=np.int64)
        for idx, (i, j) in enumerate(pairs):
            mi, mj = mask(i), mask(j)
            rowsT1[idx] = all_k[((all_k & mi) == 0) & ((all_k & mj) == 0)]
            rowsT2[idx] = all_k[((all_k & mi) == 0) & ((all_k & mj) != 0)]
            maskcols[idx] = mi ^ mj
        colsT1 = rowsT1 ^ maskcols[:, None]
        colsT2 = rowsT2 ^ maskcols[:, None]
        self._rowsT1 = xp.asarray(rowsT1)
        self._colsT1 = xp.asarray(colsT1)
        self._rowsT2 = xp.asarray(rowsT2)
        self._colsT2 = xp.asarray(colsT2)
        ZZmat = np.stack([self.ops._z_np[i] * self.ops._z_np[j] for (i, j) in pairs])
        self._ZZmat = xp.asarray(ZZmat, dtype=self.ops.float_dtype)

    def extract_correlators_fast(self, rho):
        """Vectorized replacement for extract_correlators_dm_xp -- same
        234 (at n=12) numbers, same order (per-qubit [X,Y,Z] then per-pair
        [XX,YY,ZZ], matching extract_correlators_dm's iteration order),
        computed via a handful of large gathers instead of ~700 small
        transpose+trace calls."""
        xp = self.ops.xp
        diag = xp.real(xp.diagonal(rho)).astype(self.ops.float_dtype)

        Zq = self._Zmat @ diag
        S = rho[self._rows1, self._cols1].sum(axis=1)
        Xq = 2.0 * xp.real(S)
        Yq = -2.0 * xp.imag(S)
        single = xp.stack([Xq, Yq, Zq], axis=1).reshape(-1)

        ZZij = self._ZZmat @ diag
        T1 = rho[self._rowsT1, self._colsT1].sum(axis=1)
        T2 = rho[self._rowsT2, self._colsT2].sum(axis=1)
        XXij = 2.0 * (xp.real(T1) + xp.real(T2))
        YYij = 2.0 * (xp.real(T2) - xp.real(T1))
        pair = xp.stack([XXij, YYij, ZZij], axis=1).reshape(-1)

        out = xp.concatenate([single, pair])
        return (cp.asnumpy(out) if (HAS_CUPY and xp is cp) else out).astype(np.float64)

    def _propagator(self, t):
        phases = np.exp(-1j * self._evals * t)
        U = (self._evecs * phases) @ self._evecs.conj().T
        return self.ops.asarray(U)

    def _inject(self, rho, x):
        """x: full feature vector (len 13 for the pilot data). First
        n_qubits (12) features injected as RY rotations per-qubit; the
        13th (remaining) feature folded onto qubit 0 as an RZ phase --
        matches QRCx.reservoir.sequential.SequentialDissipativeQRC's
        `_inject_ry` restricted/overflow injection convention exactly."""
        n = self.n_qubits
        a = self.input_scaling
        for j in range(min(len(x), n)):
            K = ry_matrix(a * self.w_in[j] * x[j])
            rho = self.ops.conjugate_1q(rho, self.ops.asarray(K), j)
        if len(x) > n:
            # 13th feature (our real weather data has 13 features, 12
            # qubits -- this branch fires every single step) folded onto
            # qubit 0 as an RZ phase. Real throughput bug found on the
            # qBraid H200: this used to force a full density-matrix
            # GPU->CPU->GPU roundtrip on EVERY step (via to_numpy/asarray)
            # just to multiply by a diagonal phase -- now done entirely
            # on self.ops.xp (GPU when available), matching the fix
            # applied to extract_correlators_dm_xp for the same reason.
            extra = a * self.w_in[0] * float(np.sum(x[n:]))
            xp = self.ops.xp
            z0 = self.ops._z[0]
            phase = xp.exp(-1j * extra * z0 / 2.0).astype(self.ops.dtype)
            rho = rho * phase[:, None] * xp.conj(phase)[None, :]
        return rho

    def _dissipate(self, rho):
        if self.gamma2 > 0:
            rho = rho * self._dephasing_coeff
        if self.gamma1 > 0:
            for i in range(self.n_qubits):
                d = self._damping_d[i]
                term0 = rho * d[:, None] * d[None, :]
                term1 = self.ops.conjugate_1q(rho, self._damping_e1, i)
                rho = term0 + term1
        return rho

    def drive(self, seq, ckpt_path=None, ckpt_every=500):
        """seq: (T, n_features). Returns features (T, V*n_feat) as a
        numpy (CPU) array — correlator extraction runs on CPU per step
        (ptrace_1q_np/ptrace_2q_np are numpy-only) after pulling the
        current rho off the GPU; this mirrors sprint26_pipeline.py's
        SequentialReservoir.drive exactly (already validated there).

        ckpt_path (optional): periodically saves (rho, out-so-far, step
        index) to this path so a killed/stopped process can resume mid-
        drive instead of restarting from t=0 -- cheap insurance against
        the qBraid instance's observed ~45min forced stops. rho itself is
        recurrent state and MUST be restored exactly (unlike v4's
        embarrassingly-parallel batches), so this checkpoints the full
        density matrix, not just progress metadata."""
        T = seq.shape[0]
        n_pairs = self.n_qubits * (self.n_qubits - 1) // 2
        n_feat = 3 * self.n_qubits + 3 * n_pairs
        V = self.multiplexing

        t_start = 0
        elapsed_prior = 0.0
        if ckpt_path is not None and Path(ckpt_path).exists():
            with np.load(ckpt_path) as ckpt:
                out = ckpt["out"].copy()
                last_completed_t = int(ckpt["t"])
                elapsed_prior = float(ckpt["elapsed"])
                rho = self.ops.asarray(ckpt["rho_re"] + 1j * ckpt["rho_im"])
            t_start = last_completed_t + 1  # `t` in the checkpoint is the last FULLY completed step; resuming at t itself would re-inject/re-evolve/re-dissipate onto an already-updated rho, double-applying that step
            print(f"    resuming drive from checkpoint: step {t_start}/{T} "
                  f"({elapsed_prior:.1f}s prior wall-clock)")
        else:
            out = np.zeros((T, V * n_feat), dtype=np.float64)
            rho = self.ops.vacuum()

        t0 = time.perf_counter()
        for t in range(t_start, T):
            rho = self._inject(rho, seq[t])
            for v in range(V):
                rho = self.ops.conjugate_dense(rho, self._U_inc)
                out[t, v * n_feat:(v + 1) * n_feat] = self.extract_correlators_fast(rho)
            rho = self._dissipate(rho)
            if t > 0 and t % ckpt_every == 0:
                elapsed = elapsed_prior + (time.perf_counter() - t0)
                print(f"    step {t}/{T}  ({elapsed / (t - t_start + 1):.4f} s/step, "
                      f"eta {(T - t) * elapsed / (t - t_start + 1) / 60:.1f} min)")
                if ckpt_path is not None:
                    rho_np_ckpt = self.ops.to_numpy(rho)
                    np.savez(ckpt_path, out=out, t=t, elapsed=elapsed,
                             rho_re=np.real(rho_np_ckpt), rho_im=np.imag(rho_np_ckpt))
        if ckpt_path is not None:
            Path(ckpt_path).unlink(missing_ok=True)  # drive fully complete
        return out


# ============================================================
# Readout: Ridge with val-tuned alpha, direct vs residual target
# ============================================================

def fit_eval_ridge(X_fit, y_fit, X_val_tune, y_val_tune, X_eval, y_eval):
    best_alpha, best_score = ALPHA_GRID[0], np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha).fit(X_fit, y_fit)
        score = float(np.mean((y_val_tune - m.predict(X_val_tune)) ** 2))
        if score < best_score:
            best_score, best_alpha = score, alpha
    final_model = Ridge(alpha=best_alpha).fit(np.concatenate([X_fit, X_val_tune]),
                                               np.concatenate([y_fit, y_val_tune]))
    pred = final_model.predict(X_eval)
    return pred, best_alpha


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def skill(y, p, y_persist):
    rmse_p = rmse(y, p)
    rmse_pers = rmse(y, y_persist)
    return float(1.0 - rmse_p / rmse_pers) if rmse_pers > 0 else float("nan")


def diebold_mariano_vs_persistence(y_true, y_pred, y_persist, h):
    """Squared-error DM test, Bartlett/Newey-West HAC (maxlags=h-1) --
    ported from QRCx.metrics.significance.diebold_mariano (verified there
    against statsmodels to rel=1e-8, Sprint 3); reimplemented inline here
    since this script must stay import-free from the QRCx package for
    portability to a bare qBraid GPU notebook."""
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


def evaluate_ablation_cell(features_train, features_val, train_seq, val_seq,
                            target_col_idx, architecture, horizon, washout):
    """Post-hoc readout on already-cached features — no reservoir redrive.
    architecture: 'direct' (predict y[t+h] directly) or 'residual'
    (predict y[t+h]-y[t], add persistence back at predict time), matching
    the project's DirectQRC/ResidualQRC convention."""
    y_train_full = train_seq[:, target_col_idx]
    y_val_full = val_seq[:, target_col_idx]

    def make_xy(feats, y_full, start):
        n = len(y_full) - horizon
        X = feats[start:n]
        idx = np.arange(start, n)
        y_target = y_full[idx + horizon]
        y_persist = y_full[idx]
        if architecture == "residual":
            y_reg = y_target - y_persist
        else:
            y_reg = y_target
        return X, y_reg, y_target, y_persist

    X_tr, y_tr_reg, y_tr_true, y_tr_pers = make_xy(features_train, y_train_full, washout)
    # 20% tail of train as val-tune slice (matches sprint26_pipeline's n_val_frac convention)
    n_val_tune = int(0.2 * len(X_tr))
    X_fit, X_vt = X_tr[:-n_val_tune], X_tr[-n_val_tune:]
    y_fit, y_vt = y_tr_reg[:-n_val_tune], y_tr_reg[-n_val_tune:]

    X_ev, y_ev_reg, y_ev_true, y_ev_pers = make_xy(features_val, y_val_full, 0)

    pred_reg, alpha = fit_eval_ridge(X_fit, y_fit, X_vt, y_vt, X_ev, y_ev_reg)
    pred_true = pred_reg + y_ev_pers if architecture == "residual" else pred_reg

    dm = diebold_mariano_vs_persistence(y_ev_true, pred_true, y_ev_pers, horizon)
    return {
        "architecture": architecture, "horizon": horizon,
        "rmse": rmse(y_ev_true, pred_true),
        "skill_vs_persistence": skill(y_ev_true, pred_true, y_ev_pers),
        "dm_vs_persistence": dm,
        "best_alpha": alpha, "n_eval": int(len(y_ev_true)),
    }


# ============================================================
# Main: 12 trajectory-cached drives -> 8 ablation cells x 4 horizons
# ============================================================

def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(
            f"{INPUT_NPZ} not found. Run scripts/sprint4_export_pilot_seq.py "
            "locally first and upload the resulting .npz alongside this script."
        )
    npz = np.load(INPUT_NPZ)
    train_seq = npz["train_seq"]
    val_seq = npz["val_seq"]
    target_col_idx = int(npz["target_col_idx"])
    print(f"Loaded pilot sequences: train_seq={train_seq.shape}, val_seq={val_seq.shape}, "
          f"target_col_idx={target_col_idx}")

    # Resume across whole-script restarts (not just mid-drive): if OUT_PATH
    # already has completed drives from an earlier run of this same script
    # (e.g. after the qBraid instance's observed ~45min forced stop), keep
    # them and skip re-driving -- only mid-drive checkpointing (in
    # SequentialReservoir.drive) handles the *current* drive; this handles
    # everything already fully finished before the stop.
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        completed_keys = {(d["gamma1_name"], d["V"], d["w_in_seed"]) for d in results.get("drives", [])}
        print(f"Resuming from {OUT_PATH}: {len(completed_keys)} drive(s) already completed.")
    else:
        results = {"n_qubits": N_QUBITS, "dtype": DTYPE_STR, "washout": WASHOUT,
                   "horizons": HORIZONS, "w_in_seeds": W_IN_SEEDS,
                   "gamma1_values": GAMMA1_VALUES, "V_values": V_VALUES,
                   "drives": [], "ablation_cells": []}
        completed_keys = set()

    drive_plan = []
    for gamma1_name, gamma1 in GAMMA1_VALUES.items():
        for V in V_VALUES:
            for seed in W_IN_SEEDS:
                drive_plan.append({"gamma1_name": gamma1_name, "gamma1": gamma1, "V": V, "w_in_seed": seed})

    print(f"Trajectory-caching plan: {len(drive_plan)} unique drives "
          f"(2 gamma1 x {len(V_VALUES)} V x {len(W_IN_SEEDS)} seeds), each covering "
          f"train_seq ({len(train_seq)} steps) + val_seq ({len(val_seq)} steps) as two passes.")

    # --- Cost-projection preflight (same lesson as sprint26_stage3plus.py:
    # never commit to a multi-hour+ full run without measuring real
    # per-step cost first). Uses the worst-case V (max multiplexing) since
    # that dominates cost; if it doesn't fit REMAINING_BUDGET_HOURS across
    # all `len(drive_plan)` drives, every drive is truncated to the most
    # recent TRAIN_STEPS_FALLBACK steps of train_seq (still real data, a
    # shorter window) and this reduction is logged, not silently absorbed.
    worst_V = max(V_VALUES)
    w_in_probe = np.random.default_rng(W_IN_SEEDS[0]).uniform(0.5, 1.5, size=N_QUBITS)
    res_probe = SequentialReservoir(
        n_qubits=N_QUBITS, tau=TAU, gamma1=GAMMA1_VALUES["tuned"], gamma2=GAMMA2,
        J=J, g=G, input_scaling=INPUT_SCALING, w_in=w_in_probe,
        multiplexing=worst_V, use_gpu=HAS_CUPY, dtype_str=DTYPE_STR,
    )
    sample_seq = train_seq[:SAMPLE_STEPS]
    t0 = time.perf_counter()
    res_probe.drive(sample_seq)
    s_per_step = (time.perf_counter() - t0) / SAMPLE_STEPS
    full_steps_per_drive = len(train_seq) + len(val_seq)
    projected_hours_all = len(drive_plan) * full_steps_per_drive * s_per_step / 3600
    print(f"\nCost projection (worst-case V={worst_V}, complex64): {s_per_step:.4f} s/step -> "
          f"{projected_hours_all:.2f}h projected for all {len(drive_plan)} drives at full length "
          f"({full_steps_per_drive} steps each). Budget: {REMAINING_BUDGET_HOURS:.2f}h.")

    scope_reduced = False
    if projected_hours_all > REMAINING_BUDGET_HOURS:
        scope_reduced = True
        train_seq = train_seq[-TRAIN_STEPS_FALLBACK:]
        full_steps_per_drive = len(train_seq) + len(val_seq)
        projected_hours_reduced = len(drive_plan) * full_steps_per_drive * s_per_step / 3600
        print(f"  -> Projected cost exceeds budget. FALLING BACK to the most recent "
              f"{TRAIN_STEPS_FALLBACK} train steps per drive (real data, shorter window). "
              f"New projection: {projected_hours_reduced:.2f}h for all {len(drive_plan)} drives.")
    results["scope_reduction"] = {
        "applied": scope_reduced,
        "reason": "projected cost exceeded REMAINING_BUDGET_HOURS" if scope_reduced else None,
        "probe_s_per_step": s_per_step, "probe_V": worst_V,
        "train_steps_used": int(len(train_seq)),
    }

    for i, plan in enumerate(drive_plan):
        key = (plan["gamma1_name"], plan["V"], plan["w_in_seed"])
        if key in completed_keys:
            print(f"\n=== Drive {i + 1}/{len(drive_plan)}: {key} -- already completed, skipping ===")
            continue
        print(f"\n=== Drive {i + 1}/{len(drive_plan)}: gamma1={plan['gamma1']} "
              f"({plan['gamma1_name']}), V={plan['V']}, w_in_seed={plan['w_in_seed']} ===")
        w_in = np.random.default_rng(plan["w_in_seed"]).uniform(0.5, 1.5, size=N_QUBITS)
        res = SequentialReservoir(
            n_qubits=N_QUBITS, tau=TAU, gamma1=plan["gamma1"], gamma2=GAMMA2,
            J=J, g=G, input_scaling=INPUT_SCALING, w_in=w_in,
            multiplexing=plan["V"], use_gpu=HAS_CUPY, dtype_str=DTYPE_STR,
        )
        drive_tag = f"{plan['gamma1_name']}_V{plan['V']}_seed{plan['w_in_seed']}"
        t0 = time.perf_counter()
        feats_train = res.drive(train_seq, ckpt_path=f"sprint4_v5_ckpt_train_{drive_tag}.npz")
        feats_val = res.drive(val_seq, ckpt_path=f"sprint4_v5_ckpt_val_{drive_tag}.npz")  # fresh vacuum start for val (documented: not a continuous carry-over from train's final state, since val_seq is loaded/driven separately here; if a continuous single pass across train+val is required for the paper's final number, concatenate train_seq/val_seq before driving and slice afterward -- flagged here as a design choice, not hidden)
        elapsed = time.perf_counter() - t0
        print(f"  Drive wall-clock: {elapsed / 60:.2f} min ({elapsed / (len(train_seq) + len(val_seq)):.4f} s/step)")

        drive_entry = {**plan, "wall_clock_s": elapsed,
                       "s_per_step": elapsed / (len(train_seq) + len(val_seq))}
        results["drives"].append(drive_entry)

        for architecture in ("direct", "residual"):
            for h in HORIZONS:
                cell = evaluate_ablation_cell(
                    feats_train, feats_val, train_seq, val_seq, target_col_idx,
                    architecture, h, WASHOUT,
                )
                cell.update({"gamma1_name": plan["gamma1_name"], "gamma1": plan["gamma1"],
                             "V": plan["V"], "w_in_seed": plan["w_in_seed"]})
                results["ablation_cells"].append(cell)
                print(f"    [{architecture} h={h}] skill_vs_persistence="
                      f"{cell['skill_vs_persistence'] * 100:.2f}%  rmse={cell['rmse']:.4f}")

        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Checkpoint written to {OUT_PATH} after drive {i + 1}/{len(drive_plan)}.")

    print(f"\nAll {len(drive_plan)} drives complete. Full results in {OUT_PATH} "
          f"-- please share this file's contents back.")
    return results


if __name__ == "__main__":
    main()
