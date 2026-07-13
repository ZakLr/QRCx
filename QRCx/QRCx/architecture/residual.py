from typing import Optional

import numpy as np


class ResidualQRC:
    def __init__(
        self,
        reservoir,
        readout,
        target_col_idx: int = 0,
    ):
        self.reservoir = reservoir
        self.readout = readout
        self.target_col_idx = target_col_idx
        self._fitted = False

    def _build_residual_dataset(self, X_raw: np.ndarray, y_raw: np.ndarray):
        assert X_raw.ndim == 3, f"Expected 3D input, got shape {X_raw.shape}"
        assert y_raw.ndim == 2, f"Expected 2D targets, got shape {y_raw.shape}"
        n_samples = X_raw.shape[0]
        persist = X_raw[:, -1, self.target_col_idx]
        assert persist.shape == (n_samples,), f"Expected ({n_samples},), got {persist.shape}"
        residuals = y_raw - persist[:, np.newaxis]
        assert residuals.shape == y_raw.shape
        return residuals, persist

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        residuals_train, _ = self._build_residual_dataset(X_train, y_train)
        residuals_val, _ = self._build_residual_dataset(X_val, y_val)

        F_train = self.reservoir.transform(X_train)
        F_val = self.reservoir.transform(X_val)

        self.readout.fit(F_train, residuals_train, F_val, residuals_val)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Model not fitted"
        assert X.ndim == 3, f"Expected 3D input, got shape {X.shape}"

        persist = X[:, -1, self.target_col_idx, np.newaxis]
        F = self.reservoir.transform(X)
        residual_pred = self.readout.predict(F)
        if residual_pred.ndim == 1:
            residual_pred = residual_pred[:, np.newaxis]

        result = persist + residual_pred
        return result

    # Verification (reference only, not executed):
    # from QRCx.reservoir.tfim import AtmosphericQRC
    # from QRCx.readout.krr import KRRReadout
    # qrc = AtmosphericQRC(n_qubits=4, trotter_steps=3, seed=42)
    # readout = KRRReadout()
    # model = ResidualQRC(qrc, readout, target_col_idx=0)
    # X = np.random.randn(10, 24, 9)
    # y = np.random.randn(10, 2)
    # model.fit(X, y, X, y)
    # pred = model.predict(X)
    # assert pred.shape == (10, 2)
