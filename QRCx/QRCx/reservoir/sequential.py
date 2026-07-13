"""Sequential dissipative quantum reservoir (v5 paradigm, Sprint 1 + Sprint 2
performance pass).

Replaces the v4 windowed quantum-kernel design (AtmosphericQRC / tfim.py,
which re-encodes a fresh 24h window from |0><0> for every sample) with a
genuinely recurrent reservoir: a single density matrix rho is carried
forward hour-by-hour across the whole dataset, so state at time t depends
on the reservoir's own history rather than only on the window's raw inputs.

Step update (Hou et al. 2026 style, arXiv:2508.12383):

    rho_k = D_{g1,g2} o e^{-iH tau} o U(s_k) [rho_{k-1}]

  U(s_k)      input injection of the (13-dim) feature vector s_k
  e^{-iH tau} existing TFIM (J=1.0, g=1.0)
  D_{g1,g2}   per-qubit amplitude damping (g1) + dephasing (g2), once per
              step -- the fading-memory / ESP mechanism. Non-unital noise
              (T1 relaxation) is the resource here, not depolarizing noise
              on encoding angles (that was the v4 convention, e.g.
              metrics/noise.py's depolarising_sweep; that hook is kept
              as-is elsewhere in the codebase for ablation, not reimplemented
              in this module).

Sprint 2 Phase 2.0 performance pass -- two propagator modes for e^{-iH tau}:
  - propagator="exact" (default): the *full* 12-qubit TFIM Hamiltonian
    H = g*sum(X_i) + J*sum_{i<j} Z_i Z_j is built once (dense 2**n x 2**n)
    and exactly exponentiated once via eigendecomposition
    (U = V exp(-i*evals*tau) V^dagger). Each step then costs 2 dense
    matmuls (rho -> U rho U^dagger) instead of trotter_steps*(n_qubits RX +
    combined ZZ phase) einsum-based gate applications -- this is what
    actually made the Sprint 1 runtime gate NO-GO at 12 qubits feasible to
    revisit (see docs/sprint_log/SPRINT_2_REPORT.md for the re-benchmark).
    This is not something real hardware can execute (no device applies an
    arbitrary dense 2**n-dim unitary directly) -- it is a *faster and more
    accurate* classical simulation of "what perfect unitary evolution under
    H for time tau would produce", with zero Trotter error.
  - propagator="trotter": the original Sprint 1 gate-by-gate first-order
    Trotter path (RX + IsingZZ per substep), which *is* what a real Trotterized
    circuit executes on hardware. Kept for hardware-matched runs and to
    validate propagator="exact" against it (see validate_trotter_vs_exact
    in this module).

Density-matrix backends (benchmarked, fastest wins as backend="auto"):
  - "numpy":       hand-rolled Kraus propagation via reshape+einsum,
                   avoiding ever materialising a full 2**n x 2**n Pauli
                   operator (see also readout/correlators.py's *_dm
                   helpers). No external quantum-simulator dependency.
                   Optionally runs on GPU via CuPy if installed and
                   requested (use_gpu=True; silently falls back to numpy
                   if CuPy/a GPU is unavailable).
  - "qiskit_aer":  AerSimulator(method="density_matrix"), Kraus channels
                   via qiskit_aer.noise.
  - "pennylane_mixed": qml.device("default.mixed") with qml.AmplitudeDamping
                   / qml.PhaseDamping channels.

Trajectory caching (default): because every labeled sample's rewind window
is a contiguous suffix of the SAME real input sequence, driving the whole
sequence once and reading off features at each step after the first
`washout` steps is mathematically equivalent (up to ESP convergence error)
to independently rewinding-and-redriving from t-washout for every t --
that independence from the pre-washout state is exactly the echo state
property this reservoir is required to have. `drive()` implements this
single-pass path; `transform()` (the BaseReservoir-compatible per-sample
API) offers the naive independent-rewind path for validation/testing,
where it is cheap enough to check the two agree.

Time multiplexing (Sprint 2, Fujii & Nakajima 2017): with
`multiplexing=V > 1`, the Pauli correlators are read out at V equally
spaced sub-times t_i = i*tau/V within each step's coherent evolution
(non-destructively -- rho keeps evolving after each readout, no collapse),
giving V*234 features per step instead of 234. Cost is V observable
evaluations per step, not V re-simulations: the same fixed increment
propagator U_inc = e^{-iH tau/V} (computed once, whichever propagator mode
is active) is applied V times in a row before damping is applied once at
the end of the full step.
"""
from typing import Optional

import numpy as np

from ..readout.correlators import extract_correlators_dm
from .base import BaseReservoir

_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _hadamard() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def _amplitude_damping_kraus(gamma: float) -> list:
    if gamma <= 0.0:
        return [np.eye(2, dtype=np.complex128)]
    e0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    e1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [e0, e1]


def _get_xp(use_gpu: bool):
    """Return (array module, actually_using_gpu). Falls back to numpy
    silently if CuPy isn't installed or no CUDA device is visible."""
    if not use_gpu:
        return np, False
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp, True
    except Exception:
        pass
    return np, False


def build_tfim_hamiltonian(n_qubits: int, J: float, g: float, z_arrays: list) -> np.ndarray:
    """Dense full TFIM Hamiltonian H = g*sum(X_i) + J*sum_{i<j} Z_i Z_j.

    Built directly via bit manipulation (never via kron of 2**n x 2**n
    single-qubit operators): the Z*Z part is diagonal (O(n^2 * dim) to
    accumulate), the X part flips one bit per qubit (O(n * dim) fancy
    indexing). One-time O(dim^2) cost to allocate/fill the dense matrix,
    dominated in practice by the subsequent eigh, not this construction.
    """
    dim = 2 ** n_qubits
    diag = np.zeros(dim, dtype=np.float64)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            diag += J * z_arrays[i] * z_arrays[j]
    H = np.diag(diag).astype(np.complex128)
    idx = np.arange(dim)
    for q in range(n_qubits):
        bitpos = n_qubits - 1 - q
        flipped = idx ^ (1 << bitpos)
        H[idx, flipped] += g
    return H


class NumpyDensityOps:
    """Reshape+einsum density-matrix propagation for n_qubits <= ~14,
    optionally on GPU via CuPy.

    rho is kept as a plain (dim, dim) complex128 matrix, dim = 2**n_qubits.
    Qubit q's ket "leg" is the block of size 2 at reshape position q when
    the row index is unravelled as (2,)*n_qubits (big-endian: qubit 0 is
    the most significant bit); analogously for the column/bra index.
    """

    def __init__(self, n_qubits: int, use_gpu: bool = False):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.xp, self.on_gpu = _get_xp(use_gpu)
        xp = self.xp
        idx = np.arange(self.dim)
        # bit_of[q] : +1 if qubit q is |0> for this basis index, -1 if |1>
        # (Z-eigenvalue convention), shape (dim,), one array per qubit.
        self._z_np = [1 - 2 * ((idx >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)]
        self._z = [xp.asarray(z) for z in self._z_np]

    def asarray(self, a: np.ndarray):
        return self.xp.asarray(a)

    def to_numpy(self, a) -> np.ndarray:
        return a if self.xp is np else self.xp.asnumpy(a)

    def vacuum(self):
        rho = self.xp.zeros((self.dim, self.dim), dtype=self.xp.complex128)
        rho[0, 0] = 1.0
        return rho

    def conjugate_1q(self, rho, K: np.ndarray, qubit: int):
        """rho -> K rho K^dagger, K acting on `qubit` only (one Kraus term)."""
        xp = self.xp
        n = self.n
        K = xp.asarray(K)
        P, S = 2 ** qubit, 2 ** (n - qubit - 1)
        r = rho.reshape(P, 2, S, self.dim)
        r = xp.einsum("ab,xbyz->xayz", K, r, optimize=True)
        r = r.reshape(self.dim, P, 2, S)
        r = xp.einsum("ba,xyaz->xybz", xp.conj(K), r, optimize=True)
        return r.reshape(self.dim, self.dim)

    def conjugate_dense(self, rho, U):
        """rho -> U rho U^dagger for a full (dim, dim) unitary (2 dense matmuls)."""
        U = self.xp.asarray(U)
        return U @ rho @ U.conj().T

    def apply_channel_1q(self, rho, kraus_ops: list, qubit: int):
        """Full Kraus channel: rho -> sum_k K_k rho K_k^dagger."""
        if len(kraus_ops) == 1:
            return self.conjugate_1q(rho, kraus_ops[0], qubit)
        out = self.xp.zeros_like(rho)
        for K in kraus_ops:
            out = out + self.conjugate_1q(rho, K, qubit)
        return out

    def apply_diagonal_phase(self, rho, ket_phase):
        """rho -> diag(phase) rho diag(phase)^dagger for a diagonal unitary."""
        return rho * ket_phase[:, None] * self.xp.conj(ket_phase)[None, :]

    def dephasing_coeff(self, gamma2: float, qubits: Optional[list] = None):
        """Precomputed real (dim,dim) elementwise factor for dephasing on `qubits`.

        Per qubit, coherence between |0> and |1> decays by sqrt(1-gamma2);
        populations (i==j on that qubit) are unaffected. This is the
        standard closed form for two diagonal Kraus operators
        diag(1, sqrt(1-g2)) / diag(0, sqrt(g2)); see module docstring.
        """
        xp = self.xp
        qubits = range(self.n) if qubits is None else qubits
        c = float(np.sqrt(max(0.0, 1.0 - gamma2)))
        coeff = xp.ones((self.dim, self.dim), dtype=xp.float64)
        for q in qubits:
            same_bit = (self._z[q][:, None] * self._z[q][None, :]) > 0
            coeff = coeff * xp.where(same_bit, 1.0, c)
        return coeff

    def z_basis_phase(self, single_angles: dict, pair_angles: dict):
        """Combined ket-side phase vector for a set of commuting Z/ZZ rotations.

        single_angles: {qubit: theta} for RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2})
        pair_angles:   {(i,j): phi} for IsingZZ(phi), diag entries
                       exp(-i phi/2 * z_i z_j).
        """
        xp = self.xp
        total = xp.zeros(self.dim, dtype=xp.float64)
        for q, theta in single_angles.items():
            total = total + (-0.5 * theta) * self._z[q]
        for (i, j), phi in pair_angles.items():
            total = total + (-0.5 * phi) * self._z[i] * self._z[j]
        return xp.exp(1j * total)


class SequentialDissipativeQRC(BaseReservoir):
    """Recurrent dissipative TFIM reservoir with amplitude damping + dephasing.

    Parameters mirror the Sprint 1 spec plus Sprint 2's Phase 2.0
    performance pass (propagator, use_gpu) and Phase 2.1/2.2 additions
    (multiplexing, w_in). `backend="auto"` resolves to "numpy" (the
    runtime-gate winner; see docs/sprint_log/SPRINT_1_REPORT.md and
    SPRINT_2_REPORT.md) rather than benchmarking live at construction
    time -- see scripts/benchmark_sequential_backends.py for the actual
    backend comparison.
    """

    def __init__(
        self,
        n_qubits: int = 12,
        tau: float = 1.0,
        trotter_steps: int = 10,
        gamma1: float = 0.02,
        gamma2: float = 0.02,
        input_scaling: float = 1.0,
        washout: int = 24,
        injection: str = "ry",
        backend: str = "auto",
        J: float = 1.0,
        g: float = 1.0,
        seed: int = 42,
        propagator: str = "exact",
        use_gpu: bool = False,
        multiplexing: int = 1,
        w_in: Optional[np.ndarray] = None,
    ):
        assert injection in ("ry", "zz")
        assert propagator in ("exact", "trotter")
        assert multiplexing >= 1
        self.n_qubits = n_qubits
        self.tau = tau
        self.trotter_steps = trotter_steps
        self.dt = tau / trotter_steps
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.input_scaling = input_scaling
        self.washout = washout
        self.injection = injection
        self.J = J
        self.g = g
        self.seed = seed
        self.propagator = propagator
        self.multiplexing = multiplexing
        self.backend = "numpy" if backend == "auto" else backend
        if self.backend != "numpy":
            raise NotImplementedError(
                f"backend={self.backend!r} not implemented in SequentialDissipativeQRC; "
                "use QRCx.reservoir.sequential_backends for qiskit_aer/pennylane_mixed "
                "benchmarking wrappers."
            )
        self.ops = NumpyDensityOps(n_qubits, use_gpu=use_gpu)
        self.on_gpu = self.ops.on_gpu
        self._pair_list = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

        # Phase 2.0d: w_in is a first-class, explicit, seeded parameter --
        # never unseeded, never hand-picked. Defaults to all-ones (neutral;
        # preserves exact Sprint 1 behavior / test outputs when not given).
        # Callers that need to break qubit-exchange symmetry for a
        # single-input task (e.g. NARMA10) pass an explicit seeded
        # np.random.default_rng(seed).uniform(...) vector, persisted
        # alongside the rest of a frozen config in configs/*.yaml.
        self.w_in = np.ones(n_qubits) if w_in is None else np.asarray(w_in)
        assert self.w_in.shape == (n_qubits,)

        self._U_inc_cache = {}
        if self.propagator == "exact":
            H = build_tfim_hamiltonian(n_qubits, J, g, self.ops._z_np)
            evals, evecs = np.linalg.eigh(H)
            self._evals, self._evecs = evals, evecs
            self._U_full = self._propagator_for_time(tau)
        else:
            self._evals = self._evecs = self._U_full = None

    def _propagator_for_time(self, t: float) -> np.ndarray:
        """Exact e^{-iHt} reusing the cached eigendecomposition (cheap: one
        O(dim) diagonal exponential + 2 dense matmuls, done once per
        distinct t needed, not per step)."""
        phases = np.exp(-1j * self._evals * t)
        return (self._evecs * phases) @ self._evecs.conj().T

    # -- gate layers -----------------------------------------------------

    def _inject_ry(self, rho, x: np.ndarray):
        n = self.n_qubits
        a = self.input_scaling
        single_angles = {}
        for j in range(min(len(x), n)):
            rho = self.ops.conjugate_1q(rho, _ry(a * self.w_in[j] * x[j]), j)
        if len(x) > n:
            # fold remaining (spec: feature 13) onto qubit 0 as RZ
            single_angles[0] = a * self.w_in[0] * float(np.sum(x[n:]))
            phase = self.ops.z_basis_phase(single_angles, {})
            rho = self.ops.apply_diagonal_phase(rho, phase)
        return rho

    def _inject_zz(self, rho, x: np.ndarray):
        n = self.n_qubits
        d = len(x)
        for i in range(n):
            rho = self.ops.conjugate_1q(rho, _hadamard(), i)
        single_angles = {i: self.w_in[i] * x[i % d] for i in range(n)}
        pair_angles = {
            (i, j): self.w_in[i] * self.w_in[j] * (np.pi - x[i % d]) * (np.pi - x[j % d])
            for (i, j) in self._pair_list
        }
        phase = self.ops.z_basis_phase(single_angles, pair_angles)
        return self.ops.apply_diagonal_phase(rho, phase)

    def _trotter_substeps(self, rho, n_substeps: int):
        """Advance by `n_substeps` first-order Trotter substeps (each dt =
        tau/trotter_steps); used both for the full-step evolution
        (n_substeps=trotter_steps) and for multiplexed sub-evolution
        (n_substeps=trotter_steps // multiplexing)."""
        n = self.n_qubits
        dt = self.dt
        pair_angle = 2.0 * self.J * dt
        rx_angle = 2.0 * self.g * dt
        for _ in range(n_substeps):
            for i in range(n):
                rho = self.ops.conjugate_1q(rho, _rx(rx_angle), i)
            pair_angles = {pair: pair_angle for pair in self._pair_list}
            phase = self.ops.z_basis_phase({}, pair_angles)
            rho = self.ops.apply_diagonal_phase(rho, phase)
        return rho

    def _evolve_increment(self, rho, U_inc: Optional[np.ndarray], n_trotter_substeps: int):
        """One "increment" of coherent evolution (tau/multiplexing worth),
        exact (2 matmuls with the precomputed U_inc) or Trotter
        (n_trotter_substeps gate-by-gate substeps)."""
        if self.propagator == "exact":
            return self.ops.conjugate_dense(rho, U_inc)
        return self._trotter_substeps(rho, n_trotter_substeps)

    def _dissipate(self, rho):
        n = self.n_qubits
        if self.gamma2 > 0:
            rho = rho * self.ops.dephasing_coeff(self.gamma2)
        if self.gamma1 > 0:
            kraus = _amplitude_damping_kraus(self.gamma1)
            for i in range(n):
                rho = self.ops.apply_channel_1q(rho, kraus, i)
        return rho

    def step(self, rho, x: np.ndarray):
        """One full update, single-block (no multiplexed intermediate
        readout): rho_k = D o e^{-iH tau} o U(s_k) [rho_{k-1}]."""
        rho = self._inject_zz(rho, x) if self.injection == "zz" else self._inject_ry(rho, x)
        rho = self._evolve_increment(rho, self._U_full, self.trotter_steps)
        rho = self._dissipate(rho)
        return rho

    def step_multiplexed(self, rho, x: np.ndarray) -> tuple:
        """One full update with V=`multiplexing` intermediate Pauli-correlator
        readouts during the coherent evolution (non-destructive -- rho keeps
        evolving after each readout). Damping is applied once, after the
        full tau has elapsed, exactly as in `step()`.

        Returns:
            (rho_after_full_step, features) where features has shape
            (multiplexing, n_correlators).
        """
        V = self.multiplexing
        rho = self._inject_zz(rho, x) if self.injection == "zz" else self._inject_ry(rho, x)

        if self.propagator == "exact":
            if V not in self._U_inc_cache:
                self._U_inc_cache[V] = self._propagator_for_time(self.tau / V)
            U_inc = self._U_inc_cache[V]
            n_sub = None
        else:
            U_inc = None
            assert self.trotter_steps % V == 0, (
                f"trotter_steps={self.trotter_steps} must be divisible by "
                f"multiplexing={V} (spec: measure after every M/V Trotter sub-blocks)"
            )
            n_sub = self.trotter_steps // V

        n_feat = 3 * self.n_qubits + 3 * self.n_qubits * (self.n_qubits - 1) // 2
        feats = np.zeros((V, n_feat), dtype=np.float64)
        for v in range(V):
            rho = self._evolve_increment(rho, U_inc, n_sub)
            rho_np = self.ops.to_numpy(rho)
            feats[v] = extract_correlators_dm(rho_np, self.n_qubits)

        rho = self._dissipate(rho)
        return rho, feats

    # -- driving / features ------------------------------------------------

    def drive(self, sequence: np.ndarray, return_states: bool = False):
        """Single continuous pass through the whole input sequence.

        Args:
            sequence: shape (T, n_features).
            return_states: if True also return the list of post-step rho.

        Returns:
            features: shape (T, multiplexing * 234) -- Pauli correlators
            read out at `multiplexing` equally spaced sub-times within each
            step's coherent evolution (234 at multiplexing=1), concatenated
            per step. Steps before `washout` are still returned; callers
            should discard them, matching the rewind protocol where only
            post-washout states are used for a labeled sample.
        """
        assert sequence.ndim == 2, f"Expected (T, n_features), got {sequence.shape}"
        T = sequence.shape[0]
        n_feat = 3 * self.n_qubits + 3 * self.n_qubits * (self.n_qubits - 1) // 2
        V = self.multiplexing
        features = np.zeros((T, V * n_feat), dtype=np.float64)
        rho = self.ops.vacuum()
        states = [] if return_states else None
        for k in range(T):
            if V == 1:
                rho = self.step(rho, sequence[k])
                rho_np = self.ops.to_numpy(rho)
                features[k] = extract_correlators_dm(rho_np, self.n_qubits)
            else:
                rho, feats_v = self.step_multiplexed(rho, sequence[k])
                features[k] = feats_v.reshape(-1)
            if return_states:
                states.append(rho)
        if return_states:
            return features, states
        return features

    def transform(self, X: np.ndarray) -> np.ndarray:
        """BaseReservoir-compatible per-sample API: naive independent rewind.

        For each sample's window X[i] (shape (W, d)), rewinds `washout`
        steps before it is not available here (X only contains the window
        itself), so this drives from the vacuum through `washout` copies of
        X[i, 0] (steady pre-roll) followed by the W window steps, and reads
        off the feature vector after the final step. This is the
        spec-literal "rewind and re-drive from t-n_wo" protocol, evaluated
        independently per sample -- expensive (O(n_samples * (washout+W))
        steps) and intended for small-scale validation/testing that it
        agrees with `drive()`'s single-pass trajectory cache, not for
        production-scale datasets (use `drive()` there).
        """
        assert X.ndim == 3, f"Expected (n, W, d), got {X.shape}"
        n_samples, W, d = X.shape
        n_feat = 3 * self.n_qubits + 3 * self.n_qubits * (self.n_qubits - 1) // 2
        out = np.zeros((n_samples, n_feat), dtype=np.float64)
        for i in range(n_samples):
            rho = self.ops.vacuum()
            for _ in range(self.washout):
                rho = self.step(rho, X[i, 0])
            for w in range(W):
                rho = self.step(rho, X[i, w])
            rho_np = self.ops.to_numpy(rho)
            out[i] = extract_correlators_dm(rho_np, self.n_qubits)
        return out

    def verify_esp(
        self, input_sequence: np.ndarray, n_initial_states: int = 8,
        n_steps: Optional[int] = None, convergence_threshold: float = 1e-2,
    ) -> tuple:
        """Sequential-mode ESP check (trace distance, not statevector L2 --
        see reservoir/esp_sequential.py); overrides BaseReservoir.verify_esp,
        which assumes a statevector TFIM reservoir like AtmosphericQRC.
        """
        from .esp_sequential import verify_esp_sequential
        return verify_esp_sequential(
            self, input_sequence, n_initial_states, n_steps, convergence_threshold,
        )


def validate_trotter_vs_exact(
    n_qubits: int = 4, trotter_steps_fine: int = 200, tau: float = 1.0,
    J: float = 1.0, g: float = 1.0, seed: int = 0,
) -> dict:
    """Sprint 2 Phase 2.0a validation: confirm the Trotter gate-by-gate
    implementation converges to the exact propagator as trotter_steps grows
    (validates the Trotter CODE, not the *production* trotter_steps=10
    setting -- that has real, larger, honestly-reported Trotter error; see
    docs/sprint_log/SPRINT_2_REPORT.md for both numbers).

    Returns a dict with the fine-Trotter-vs-exact discrepancy (shrinks as
    O(1/trotter_steps_fine), standard first-order Trotter-Suzuki scaling --
    empirically ~5e-3 at M=50 down to ~5e-5 at M=5000 for this
    densely-connected all-to-all-ZZ Hamiltonian; see
    docs/sprint_log/SPRINT_2_REPORT.md) and the production-setting
    (trotter_steps=10) discrepancy (expected larger).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=n_qubits)

    exact = SequentialDissipativeQRC(
        n_qubits=n_qubits, tau=tau, J=J, g=g, propagator="exact",
        gamma1=0.0, gamma2=0.0, washout=0, seed=seed,
    )
    rho0 = exact.ops.vacuum()
    rho0 = exact._inject_ry(rho0, x)
    rho_exact = exact.ops.conjugate_dense(rho0, exact._U_full)

    fine = SequentialDissipativeQRC(
        n_qubits=n_qubits, tau=tau, trotter_steps=trotter_steps_fine, J=J, g=g,
        propagator="trotter", gamma1=0.0, gamma2=0.0, washout=0, seed=seed,
    )
    rho_fine = fine._trotter_substeps(rho0.copy(), trotter_steps_fine)

    prod = SequentialDissipativeQRC(
        n_qubits=n_qubits, tau=tau, trotter_steps=10, J=J, g=g,
        propagator="trotter", gamma1=0.0, gamma2=0.0, washout=0, seed=seed,
    )
    rho_prod = prod._trotter_substeps(rho0.copy(), 10)

    fine_err = float(np.max(np.abs(rho_fine - rho_exact)))
    prod_err = float(np.max(np.abs(rho_prod - rho_exact)))
    return {
        "n_qubits": n_qubits,
        "trotter_steps_fine": trotter_steps_fine,
        "fine_vs_exact_max_abs_diff": fine_err,
        "production_trotter_steps": 10,
        "production_vs_exact_max_abs_diff": prod_err,
    }
