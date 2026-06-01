"""GFS 6h forecast loader — TIER 3 skeleton.

Operational context only: NOT a direct fairness comparison (GFS uses
global initial conditions, our model uses station-only data).
Include in paper with explicit caveat.
"""
import numpy as np


class GFSBaseline:
    def __init__(self):
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Model not fitted")
        return X[:, -1, 0]


def forecast(X_test: np.ndarray, y_test: np.ndarray,
             horizons: list[int]) -> dict:
    """Placeholder: returns persistence as fallback.

    Full implementation would load GFS operational 6h forecasts
    for KORD station and return them for each horizon.
    """
    result = {}
    for h in horizons:
        result[h] = X_test[:, -1, 0]
    return result
