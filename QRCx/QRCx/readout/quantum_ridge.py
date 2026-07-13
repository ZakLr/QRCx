from typing import Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseReadout


class QuantumRidgeReadout(BaseReadout):
    def __init__(self, gamma: float = 0.1, alpha: float = 1e-3):
        self.gamma = gamma
        self.alpha = alpha
        self.model: Optional[KernelRidge] = None

    def fit(
        self,
        F_train: np.ndarray,
        y_train: np.ndarray,
        F_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        assert F_train.ndim == 2
        y_target = y_train
        self.model = KernelRidge(kernel="rbf", gamma=self.gamma, alpha=self.alpha)
        self.model.fit(F_train, y_target)

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        assert F.ndim == 2
        return self.model.predict(F)
