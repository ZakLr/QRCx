from typing import Optional

import numpy as np
from sklearn.neural_network import MLPRegressor

from .base import BaseReadout


class MLPReadout(BaseReadout):
    def __init__(self, hidden_layer_sizes=(32, 16), max_iter=500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model: Optional[MLPRegressor] = None

    def fit(
        self,
        F_train: np.ndarray,
        y_train: np.ndarray,
        F_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        assert F_train.ndim == 2
        y_target = y_train
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
        )
        self.model.fit(F_train, y_target)

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        assert F.ndim == 2
        return self.model.predict(F)
