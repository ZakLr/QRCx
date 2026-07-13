from abc import ABC, abstractmethod

import numpy as np


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, x: np.ndarray) -> list:
        ...

    def scale_to_pi(self, x: np.ndarray) -> np.ndarray:
        x_min = x.min()
        x_max = x.max()
        return np.pi * (x - x_min) / (x_max - x_min + 1e-8)
