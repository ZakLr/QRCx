import numpy as np

from .base import BaseReservoir
from .tfim_cudaq import AtmosphericQRCCudaQ


class ParallelReservoir(BaseReservoir):
    def __init__(
        self,
        n_qubits: int = 12,
        n_layers: int = 3,
        trotter_steps: int = 10,
        seed_a: int = 42,
        seed_b: int = 43,
        **kwargs,
    ):
        self.res_a = AtmosphericQRCCudaQ(
            n_qubits=n_qubits, n_layers=n_layers, trotter_steps=trotter_steps, seed=seed_a, **kwargs
        )
        self.res_b = AtmosphericQRCCudaQ(
            n_qubits=n_qubits, n_layers=n_layers, trotter_steps=trotter_steps, seed=seed_b, **kwargs
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 3, f"Expected 3D input, got shape {X.shape}"
        n_samples = X.shape[0]

        try:
            import cudaq
            if hasattr(cudaq, "par_execute"):
                feat_a, feat_b = cudaq.par_execute(
                    lambda: self.res_a.transform(X),
                    lambda: self.res_b.transform(X),
                )
                features = np.concatenate([feat_a, feat_b], axis=1)
            else:
                feat_a = self.res_a.transform(X)
                feat_b = self.res_b.transform(X)
                features = np.concatenate([feat_a, feat_b], axis=1)
        except Exception:
            feat_a = self.res_a.transform(X)
            feat_b = self.res_b.transform(X)
            features = np.concatenate([feat_a, feat_b], axis=1)

        n_features = features.shape[1]
        assert features.shape == (n_samples, n_features)
        return features
