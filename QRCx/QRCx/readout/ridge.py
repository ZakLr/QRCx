from typing import Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseReadout


class RidgeReadout(BaseReadout):
    def __init__(self, alphas: Optional[np.ndarray] = None):
        self.alphas = alphas if alphas is not None else np.logspace(-6, 1, 20)
        self.model: Optional[RidgeCV] = None

    def fit(
        self,
        F_train: np.ndarray,
        y_train: np.ndarray,
        F_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        assert F_train.ndim == 2, f"Expected 2D features, got shape {F_train.shape}"

        y_target = y_train

        tscv = TimeSeriesSplit(n_splits=5)
        self.model = RidgeCV(alphas=self.alphas, cv=tscv)
        self.model.fit(F_train, y_target)
        print(f"Ridge: best alpha={self.model.alpha_:.6f}")

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        assert F.ndim == 2, f"Expected 2D features, got shape {F.shape}"
        return self.model.predict(F)
