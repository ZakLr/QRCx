import numpy as np


class PersistenceBaseline:
    def fit(self, X_train, y_train):
        self.n_horizons = y_train.shape[1] if y_train.ndim == 2 else 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        persist = X[:, -1, 0:1]
        return np.repeat(persist, self.n_horizons, axis=1) if self.n_horizons > 1 else persist.ravel()


def forecast(X_test: np.ndarray, y_test: np.ndarray, horizons: list[int], target_col_idx: int = 0) -> dict:
    persist = X_test[:, -1, target_col_idx]
    result = {}
    for h in horizons:
        result[h] = persist
    return result