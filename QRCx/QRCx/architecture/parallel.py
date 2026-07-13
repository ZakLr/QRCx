from typing import Optional

import numpy as np

from ..reservoir.parallel import ParallelReservoir as _ParallelReservoir


class ParallelQRC:
    def __init__(
        self,
        n_qubits: int = 8,
        trotter_steps: int = 10,
        dt: float = 0.1,
        n_layers_enc: int = 3,
        seed: int = 42,
    ):
        self.reservoir = _ParallelReservoir(
            n_qubits=n_qubits, n_layers=n_layers_enc, trotter_steps=trotter_steps, dt=dt,
            seed_a=seed, seed_b=seed + 1,
        )

    def run(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 3, f"Expected 3D input, got shape {X.shape}"
        return self.reservoir.transform(X)
