from typing import Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseReadout


class KRRReadout(BaseReadout):
    GAMMA_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    ALPHA_GRID = np.logspace(-6, -1, 6)

    def __init__(self):
        self.best_gamma: Optional[float] = None
        self.best_alpha: Optional[float] = None
        self.model: Optional[KernelRidge] = None

    def fit(
        self,
        F_train: np.ndarray,
        y_train: np.ndarray,
        F_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        assert F_train.ndim == 2, f"Expected 2D features, got shape {F_train.shape}"

        y_target = y_train

        best_score = np.inf
        tscv = TimeSeriesSplit(n_splits=5)

        for gamma in self.GAMMA_GRID:
            for alpha in self.ALPHA_GRID:
                cv_scores = []
                for train_idx, val_idx in tscv.split(F_train):
                    F_tr, F_v = F_train[train_idx], F_train[val_idx]
                    y_tr, y_v = y_target[train_idx], y_target[val_idx]
                    model = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
                    model.fit(F_tr, y_tr)
                    pred = model.predict(F_v)
                    cv_scores.append(np.sqrt(np.mean((y_v - pred) ** 2)))
                mean_score = np.mean(cv_scores)

                if F_val is not None and y_val is not None:
                    y_val_target = y_val
                    model = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
                    model.fit(F_train, y_target)
                    pred = model.predict(F_val)
                    val_score = np.sqrt(np.mean((y_val_target - pred) ** 2))
                    score = val_score
                else:
                    score = mean_score

                if score < best_score:
                    best_score = score
                    self.best_gamma = gamma
                    self.best_alpha = alpha

        self.model = KernelRidge(kernel="rbf", gamma=self.best_gamma, alpha=self.best_alpha)
        self.model.fit(F_train, y_target)
        print(f"KRR: best gamma={self.best_gamma}, best alpha={self.best_alpha}, validation RMSE={best_score:.4f}")

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        assert F.ndim == 2, f"Expected 2D features, got shape {F.shape}"
        return self.model.predict(F)
