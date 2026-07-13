from typing import Optional

import numpy as np
import pennylane as qml

from ..encoding.zz_feature_map import ZZFeatureMap
from ..readout.correlators import extract_correlators
from .base import BaseReservoir


class AtmosphericQRC(BaseReservoir):
    def __init__(
        self,
        n_qubits: int = 12,
        n_layers: int = 3,
        trotter_steps: int = 10,
        dt: float = 0.1,
        seed: int = 42,
        use_lightning: bool = True,
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
        self.encoder = ZZFeatureMap(n_qubits, n_layers)

        mean_abs_J = np.mean(np.abs(self.J[self.J != 0]))
        mean_g = np.mean(self.g)
        raw_ratio = mean_abs_J / mean_g if mean_g != 0 else 0.0
        self._jg_ratio = raw_ratio
        if raw_ratio > 0:
            self.J *= 1.0 / raw_ratio  # rescale J to make J/g = 1.0
            self._jg_ratio = 1.0

        n_pairs = n_qubits * (n_qubits - 1) // 2
        two_qubit_gates_per_step = n_qubits + n_pairs * 2
        total_two_qubit = two_qubit_gates_per_step * trotter_steps
        total_two_qubit += n_layers * n_pairs * 2

        print(f"J/g ratio: {self._jg_ratio:.3f} (target ~1.0 for critical point)")
        print(f"Estimated two-qubit gate count (N={n_qubits}): ~{total_two_qubit}")

        device_name = "lightning.qubit" if use_lightning else "default.qubit"
        self._dev = qml.device(device_name, wires=n_qubits)

    @property
    def jg_ratio(self) -> float:
        return self._jg_ratio

    def _circuit(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1

        @qml.qnode(self._dev)
        def _qnode(x: np.ndarray) -> list[np.ndarray]:
            self.encoder.circuit(x)

            for _ in range(self.trotter_steps):
                for i in range(self.n_qubits):
                    qml.RX(2.0 * self.g[i] * self.dt, wires=i)
                for i in range(self.n_qubits):
                    qml.RZ(2.0 * self.h[i] * self.dt, wires=i)
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        if self.J[i, j] != 0:
                            qml.IsingZZ(2.0 * self.J[i, j] * self.dt, wires=[i, j])

            return qml.state()

        return _qnode(x)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 3, f"Expected 3D input (n, W, d), got shape {X.shape}"
        n_samples, W, d = X.shape
        n_features = 3 * self.n_qubits + 3 * self.n_qubits * (self.n_qubits - 1) // 2
        features = np.zeros((n_samples, n_features), dtype=np.float64)

        for i in range(n_samples):
            x_flat = X[i].ravel()
            x_scaled = self.encoder.scale_to_pi(x_flat)
            state = np.array(self._circuit(x_scaled), dtype=np.complex128)
            features[i] = extract_correlators(state, self.n_qubits)

        assert features.shape == (n_samples, n_features)
        span = features.max() - features.min()
        assert span > 0.01, \
            f"Feature range only {span:.4f} — circuit produces trivial output"
        return features
