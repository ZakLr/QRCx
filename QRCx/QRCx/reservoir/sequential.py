"""Sequential dissipative quantum reservoir (v5 paradigm, Sprint 1).

Replaces the v4 windowed quantum-kernel design (AtmosphericQRC / tfim.py,
which re-encodes a fresh 24h window from |0><0> for every sample) with a
genuinely recurrent reservoir: a single density matrix rho is carried
forward hour-by-hour across the whole dataset, so state at time t depends
on the reservoir's own history rather than only on the window's raw inputs.

Step update (Hou et al. 2026 style, arXiv:2508.12383):

    rho_k = D_{g1,g2} o e^{-iH tau} o U(s_k) [rho_{k-1}]

  U(s_k)      input injection of the (13-dim) feature vector s_k
  e^{-iH tau} existing TFIM (J=1.0, g=1.0), first-order Trotter, M steps
  D_{g1,g2}   per-qubit amplitude damping (g1) + dephasing (g2), once per
              step -- the fading-memory / ESP mechanism. Non-unital noise
              (T1 relaxation) is the resource here, not depolarizing noise
              on encoding angles (that was the v4 convention, e.g.
              metrics/noise.py's depolarising_sweep; that hook is kept
              as-is elsewhere in the codebase for ablation, not reimplemented
              in this module).

Density-matrix backends (benchmarked, fastest wins as backend="auto"):
  - "numpy":       hand-rolled Kraus propagation via reshape+einsum,
                   avoiding ever materialising a full 2**n x 2**n Pauli
                   operator (see also readout/correlators.py's *_dm
                   helpers). No external quantum-simulator dependency.
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


class NumpyDensityOps:
    """Reshape+einsum density-matrix propagation for n_qubits <= ~14.

    rho is kept as a plain (dim, dim) complex128 matrix, dim = 2**n_qubits.
    Qubit q's ket "leg" is the block of size 2 at reshape position q when
    the row index is unravelled as (2,)*n_qubits (big-endian: qubit 0 is
    the most significant bit); analogously for the column/bra index.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        idx = np.arange(self.dim)
        # bit_of[q] : +1 if qubit q is |0> for this basis index, -1 if |1>
        # (Z-eigenvalue convention), shape (dim,), one array per qubit.
        self._z = [
            1 - 2 * ((idx >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)
        ]

    def vacuum(self) -> np.ndarray:
        rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        return rho

    def conjugate_1q(self, rho: np.ndarray, K: np.ndarray, qubit: int) -> np.ndarray:
        """rho -> K rho K^dagger, K acting on `qubit` only (one Kraus term)."""
        n = self.n
        P, S = 2 ** qubit, 2 ** (n - qubit - 1)
        r = rho.reshape(P, 2, S, self.dim)
        r = np.einsum("ab,xbyz->xayz", K, r, optimize=True)
        r = r.reshape(self.dim, P, 2, S)
        r = np.einsum("ba,xyaz->xybz", np.conj(K), r, optimize=True)
        return r.reshape(self.dim, self.dim)

    def apply_channel_1q(self, rho: np.ndarray, kraus_ops: list, qubit: int) -> np.ndarray:
        """Full Kraus channel: rho -> sum_k K_k rho K_k^dagger."""
        if len(kraus_ops) == 1:
            return self.conjugate_1q(rho, kraus_ops[0], qubit)
        out = np.zeros_like(rho)
        for K in kraus_ops:
            out += self.conjugate_1q(rho, K, qubit)
        return out

    def apply_diagonal_phase(self, rho: np.ndarray, ket_phase: np.ndarray) -> np.ndarray:
        """rho -> diag(phase) rho diag(phase)^dagger for a diagonal unitary."""
        return rho * ket_phase[:, None] * np.conj(ket_phase)[None, :]

    def dephasing_coeff(self, gamma2: float, qubits: Optional[list] = None) -> np.ndarray:
        """Precomputed real (dim,dim) elementwise factor for dephasing on `qubits`.

        Per qubit, coherence between |0> and |1> decays by sqrt(1-gamma2);
        populations (i==j on that qubit) are unaffected. This is the
        standard closed form for two diagonal Kraus operators
        diag(1, sqrt(1-g2)) / diag(0, sqrt(g2)); see module docstring.
        """
        qubits = range(self.n) if qubits is None else qubits
        c = np.sqrt(max(0.0, 1.0 - gamma2))
        coeff = np.ones((self.dim, self.dim), dtype=np.float64)
        for q in qubits:
            same_bit = (self._z[q][:, None] * self._z[q][None, :]) > 0
            coeff *= np.where(same_bit, 1.0, c)
        return coeff

    def z_basis_phase(self, single_angles: dict, pair_angles: dict) -> np.ndarray:
        """Combined ket-side phase vector for a set of commuting Z/ZZ rotations.

        single_angles: {qubit: theta} for RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2})
        pair_angles:   {(i,j): phi} for IsingZZ(phi), diag entries
                       exp(-i phi/2 * z_i z_j).
        """
        total = np.zeros(self.dim, dtype=np.float64)
        for q, theta in single_angles.items():
            total += -0.5 * theta * self._z[q]
        for (i, j), phi in pair_angles.items():
            total += -0.5 * phi * self._z[i] * self._z[j]
        return np.exp(1j * total)


class SequentialDissipativeQRC(BaseReservoir):
    """Recurrent dissipative TFIM reservoir with amplitude damping + dephasing.

    Parameters mirror the Sprint 1 spec. `backend="auto"` resolves to
    "numpy" (the runtime-gate winner at the 10-qubit fallback config; see
    docs/sprint_log/SPRINT_1_REPORT.md) rather than benchmarking live at
    construction time -- see scripts/benchmark_sequential_backends.py for
    the actual backend comparison.
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
    ):
        assert injection in ("ry", "zz")
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
        self.backend = "numpy" if backend == "auto" else backend
        if self.backend != "numpy":
            raise NotImplementedError(
                f"backend={self.backend!r} not implemented in SequentialDissipativeQRC; "
                "use QRCx.reservoir.sequential_backends for qiskit_aer/pennylane_mixed "
                "benchmarking wrappers."
            )
        self.ops = NumpyDensityOps(n_qubits)
        self._pair_list = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    # -- gate layers -----------------------------------------------------

    def _inject_ry(self, rho: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        a = self.input_scaling
        single_angles = {}
        for j in range(min(len(x), n)):
            rho = self.ops.conjugate_1q(rho, _ry(a * x[j]), j)
        if len(x) > n:
            # fold remaining (spec: feature 13) onto qubit 0 as RZ
            single_angles[0] = a * float(np.sum(x[n:]))
            phase = self.ops.z_basis_phase(single_angles, {})
            rho = self.ops.apply_diagonal_phase(rho, phase)
        return rho

    def _inject_zz(self, rho: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        d = len(x)
        for i in range(n):
            rho = self.ops.conjugate_1q(rho, _hadamard(), i)
        single_angles = {i: x[i % d] for i in range(n)}
        pair_angles = {
            (i, j): (np.pi - x[i % d]) * (np.pi - x[j % d]) for (i, j) in self._pair_list
        }
        phase = self.ops.z_basis_phase(single_angles, pair_angles)
        return self.ops.apply_diagonal_phase(rho, phase)

    def _trotter_evolve(self, rho: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        dt = self.dt
        pair_angle = 2.0 * self.J * dt
        rx_angle = 2.0 * self.g * dt
        for _ in range(self.trotter_steps):
            for i in range(n):
                rho = self.ops.conjugate_1q(rho, _rx(rx_angle), i)
            pair_angles = {pair: pair_angle for pair in self._pair_list}
            phase = self.ops.z_basis_phase({}, pair_angles)
            rho = self.ops.apply_diagonal_phase(rho, phase)
        return rho

    def _dissipate(self, rho: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        if self.gamma2 > 0:
            rho = rho * self.ops.dephasing_coeff(self.gamma2)
        if self.gamma1 > 0:
            kraus = _amplitude_damping_kraus(self.gamma1)
            for i in range(n):
                rho = self.ops.apply_channel_1q(rho, kraus, i)
        return rho

    def step(self, rho: np.ndarray, x: np.ndarray) -> np.ndarray:
        """One full update: rho_k = D o e^{-iH tau} o U(s_k) [rho_{k-1}]."""
        rho = self._inject_zz(rho, x) if self.injection == "zz" else self._inject_ry(rho, x)
        rho = self._trotter_evolve(rho)
        rho = self._dissipate(rho)
        return rho

    # -- driving / features ------------------------------------------------

    def drive(self, sequence: np.ndarray, return_states: bool = False):
        """Single continuous pass through the whole input sequence.

        Args:
            sequence: shape (T, n_features).
            return_states: if True also return the list of post-step rho.

        Returns:
            features: shape (T, 234) -- Pauli correlators of rho after
            each step (steps before `washout` are still returned; callers
            should discard them, matching the rewind protocol where only
            post-washout states are used for a labeled sample).
        """
        assert sequence.ndim == 2, f"Expected (T, n_features), got {sequence.shape}"
        T = sequence.shape[0]
        n_feat = 3 * self.n_qubits + 3 * self.n_qubits * (self.n_qubits - 1) // 2
        features = np.zeros((T, n_feat), dtype=np.float64)
        rho = self.ops.vacuum()
        states = [] if return_states else None
        for k in range(T):
            rho = self.step(rho, sequence[k])
            features[k] = extract_correlators_dm(rho, self.n_qubits)
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
            out[i] = extract_correlators_dm(rho, self.n_qubits)
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
