import numpy as np


class ESNBaseline:
    """Echo State Network via reservoirpy. Requires reservoirpy installed."""

    def __init__(self, nodes: int = 500, seed: int = 42, sr: float = 0.9,
                 input_scaling: float = 0.5, rc_connectivity: float = 0.1,
                 ridge: float = 1.0):
        self.nodes = nodes
        self.seed = seed
        self.sr = sr
        self.input_scaling = input_scaling
        self.rc_connectivity = rc_connectivity
        self.ridge = ridge
        self.model = None
        self.n_horizons = 1

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.n_horizons = y_train.shape[1] if y_train.ndim == 2 else 1
        from reservoirpy.nodes import Reservoir, Ridge
        reservoir = Reservoir(
            units=self.nodes, sr=self.sr, seed=self.seed,
            input_scaling=self.input_scaling,
            rc_connectivity=self.rc_connectivity,
        )
        readout = Ridge(ridge=self.ridge)
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model = (reservoir >> readout).fit(X_flat, y_train, warmup=100)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ESNBaseline not fitted")
        X_flat = X.reshape(X.shape[0], -1)
        preds = self.model.run(X_flat)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        return preds


def forecast(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
             y_test: np.ndarray, reservoir_size: int = 500,
             horizons: list[int] = None) -> dict:
    from reservoirpy.nodes import Reservoir, Ridge
    reservoir = Reservoir(
        units=reservoir_size, sr=0.9, seed=42,
        input_scaling=0.5, rc_connectivity=0.1,
    )
    readout = Ridge(ridge=1.0)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    model = (reservoir >> readout).fit(X_train_flat, y_train, warmup=100)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    pred = model.run(X_test_flat)
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    if horizons is None:
        horizons = [1, 6]
    result = {}
    for h_idx, h in enumerate(horizons):
        result[h] = pred[:, h_idx] if pred.shape[1] > h_idx else pred[:, 0]
    return result
