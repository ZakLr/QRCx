from typing import Optional

import numpy as np
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseReadout


class LassoReadout(BaseReadout):
    def __init__(self, alphas: Optional[np.ndarray] = None):
        self.alphas = alphas if alphas is not None else np.logspace(-4, 0, 10)
        self.model: Optional[MultiTaskLassoCV] = None

    def fit(
        self,
        F_train: np.ndarray,
        y_train: np.ndarray,
        F_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        assert F_train.ndim == 2
        y_target = y_train
        if y_target.ndim == 1:
            y_target = y_target[:, np.newaxis]
        tscv = TimeSeriesSplit(n_splits=5)
        self.model = MultiTaskLassoCV(alphas=self.alphas, cv=tscv, max_iter=5000)
        self.model.fit(F_train, y_target)

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        assert F.ndim == 2
        pred = self.model.predict(F)
        return pred.ravel() if pred.ndim > 1 and pred.shape[1] == 1 else pred
