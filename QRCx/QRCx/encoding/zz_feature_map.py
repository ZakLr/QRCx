import pennylane as qml
import numpy as np

from .base import BaseEncoder


class ZZFeatureMap(BaseEncoder):
    def __init__(self, n_qubits: int = 12, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def encode(self, x: np.ndarray) -> list:
        """Returns the op list WITHOUT queuing it to any active tape
        (`QueuingManager.stop_recording()`) -- constructing a PennyLane
        operator inside an active QNode's recording context auto-queues
        it immediately. Sprint 4 bug fix: `circuit()` used to call
        `qml.apply(op)` on top of that auto-queuing, silently applying
        every gate TWICE (H*H = identity -- the encoder never actually
        created superposition -- and every RZ/IsingZZ angle was doubled).
        Caught while cross-validating a from-scratch GPU reimplementation
        of this exact circuit against this module's real PennyLane output
        (scripts/sprint4_v4_gpu_handoff.py) -- see
        docs/sprint_log/SPRINT_4_REPORT.md."""
        assert x.ndim == 1, f"Expected 1D feature vector, got shape {x.shape}"
        d = len(x)
        with qml.QueuingManager.stop_recording():
            ops = []
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    ops.append(qml.Hadamard(wires=i))
                for i in range(self.n_qubits):
                    ops.append(qml.RZ(x[i % d], wires=i))
                for j in range(self.n_qubits):
                    for k in range(j + 1, self.n_qubits):
                        phi = (np.pi - x[j % d]) * (np.pi - x[k % d])
                        ops.append(qml.IsingZZ(phi, wires=[j, k]))
        return ops

    def circuit(self, x: np.ndarray):
        assert x.ndim == 1, f"Expected 1D feature vector, got shape {x.shape}"
        for op in self.encode(x):
            qml.apply(op)
