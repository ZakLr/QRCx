from abc import ABC, abstractmethod

import numpy as np


class BaseReservoir(ABC):
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def verify_esp(
        self, input_sequence: np.ndarray, n_initial_states: int = 20,
        n_steps: int = 100, convergence_threshold: float = 1e-2,
    ) -> tuple:
        from .esp import verify_esp as _verify_esp
        return _verify_esp(self, input_sequence, n_initial_states, n_steps, convergence_threshold)
