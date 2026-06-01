from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseReadout(ABC):
    @abstractmethod
    def fit(
        self,
        F_train: np.ndarray,
        y_train: np.ndarray,
        F_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        ...

    @abstractmethod
    def predict(self, F: np.ndarray) -> np.ndarray:
        ...
