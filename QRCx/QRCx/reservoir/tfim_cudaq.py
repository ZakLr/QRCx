from typing import Optional

import numpy as np

try:
    import cudaq
except ImportError:
    cudaq = None

from ..encoding.zz_feature_map import ZZFeatureMap
from .base import BaseReservoir


class AtmosphericQRCCudaQ(BaseReservoir):
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        trotter_steps: int = 10,
        dt: float = 0.1,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.trotter_steps = trotter_steps
        self.dt = dt
        rng = np.random.default_rng(seed)

        self.h = rng.uniform(-1.0, 1.0, size=n_qubits)
        self.g = rng.uniform(0.5, 1.5, size=n_qubits)
        J_raw = rng.uniform(-3.0, 3.0, size=(n_qubits, n_qubits))
        self.J = (J_raw + J_raw.T) / 2.0
        np.fill_diagonal(self.J, 0.0)

        mean_abs_J = np.mean(np.abs(self.J[self.J != 0]))
        mean_g = np.mean(self.g)
        self._jg_ratio = mean_abs_J / mean_g if mean_g != 0 else 0.0

        n_pairs = n_qubits * (n_qubits - 1) // 2
        two_qubit_gates_per_step = n_qubits + n_pairs * 2
        total_two_qubit = two_qubit_gates_per_step * trotter_steps
        total_two_qubit += n_layers * n_pairs * 2

        print(f"J/g ratio: {self._jg_ratio:.3f} (target ~1.0 for critical point)")
        print(f"Estimated two-qubit gate count (N={n_qubits}): ~{total_two_qubit}")

        self.encoder = ZZFeatureMap(n_qubits, n_layers)
        self._available = cudaq is not None
        if not self._available:
            print("CUDA-Q unavailable; falling back to PennyLane lightning.qubit")
            from .tfim import AtmosphericQRC
            self._fallback = AtmosphericQRC(n_qubits, n_layers, trotter_steps, dt, seed)
        else:
            J_flat = self.J[np.triu_indices(n_qubits, k=1)]
            h_list = self.h.tolist()
            g_list = self.g.tolist()
            J_list = J_flat.tolist()
            self._kernel = self._build_kernel(J_list, h_list, g_list)

    @property
    def jg_ratio(self) -> float:
        return self._jg_ratio

    def _build_kernel(self, J_list, h_list, g_list):
        n = self.n_qubits
        steps = self.trotter_steps
        layers = self.n_layers
        dt = self.dt

        @cudaq.kernel
        def tfim_kernel(n_qubits: int, x_enc: list[float], phis: list[float]):
            q = cudaq.qvector(n_qubits)
            d_enc = len(x_enc) // layers
            for ell in range(layers):
                for i in range(n_qubits):
                    cudaq.h(q[i])
                    xi = x_enc[ell * n_qubits + i] if (ell * n_qubits + i) < len(x_enc) else 0.0
                    cudaq.rz(xi, q[i])
                p_idx = ell * n * (n - 1) // 2
                for j in range(n_qubits):
                    for k in range(j + 1, n_qubits):
                        phi = phis[p_idx] if p_idx < len(phis) else 0.0
                        cudaq.cx(q[j], q[k])
                        cudaq.rz(phi, q[k])
                        cudaq.cx(q[j], q[k])
                        p_idx += 1
            for _ in range(steps):
                for i in range(n_qubits):
                    cudaq.rx(2.0 * g_list[i] * dt, q[i])
                    cudaq.rz(2.0 * h_list[i] * dt, q[i])
                j_idx = 0
                for j in range(n_qubits):
                    for k in range(j + 1, n_qubits):
                        J_val = J_list[j_idx]
                        cudaq.cx(q[j], q[k])
                        cudaq.rz(2.0 * J_val * dt, q[k])
                        cudaq.cx(q[j], q[k])
                        j_idx += 1

        return tfim_kernel

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 3, f"Expected 3D input, got shape {X.shape}"
        if not self._available:
            return self._fallback.transform(X)

        n_samples, W, d = X.shape
        n_pairs = self.n_qubits * (self.n_qubits - 1) // 2
        n_features = 3 * self.n_qubits + 3 * n_pairs
        features = np.zeros((n_samples, n_features), dtype=np.float64)

        use_statevector = hasattr(cudaq, "get_state")

        for i in range(n_samples):
            x_flat = X[i].ravel()
            x_scaled = self.encoder.scale_to_pi(x_flat)
            x_list = np.tile(x_scaled, max(1, self.n_layers))[:self.n_layers * self.n_qubits].tolist()
            phis = []
            for ell in range(self.n_layers):
                for j in range(self.n_qubits):
                    for k in range(j + 1, self.n_qubits):
                        phi = (np.pi - x_scaled[j % d]) * (np.pi - x_scaled[k % d])
                        phis.append(phi)

            if use_statevector:
                sv = cudaq.get_state(self._kernel, self.n_qubits, x_list, phis)
                state = np.array(sv, dtype=np.complex128)
            else:
                count = cudaq.count(self._kernel, self.n_qubits, x_list, phis)
                state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
                for bitstring, probability in count.items():
                    idx = int(bitstring, 2)
                    state[idx] = np.sqrt(probability)
            features[i] = self._extract_correlators(state)

        assert features.shape == (n_samples, n_features)
        return features

    def _extract_correlators(self, state: np.ndarray) -> np.ndarray:
        from ..readout.correlators import extract_correlators
        return extract_correlators(state, self.n_qubits)
