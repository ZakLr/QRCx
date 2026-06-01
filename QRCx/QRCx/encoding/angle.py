"""Angle encoder skeleton — Tier 3."""
import pennylane as qml
import numpy as np
from .base import BaseEncoder


class AngleEncoder(BaseEncoder):
    """Simple RX/RY/RZ rotations — baseline for ablation."""

    def __init__(self, n_qubits: int = 8, rotation: str = "RY"):
        super().__init__(n_qubits)
        self.rotation = rotation

    def encode(self, x: np.ndarray) -> list:
        ops = []
        x_scaled = self.scale_to_pi(x[:self.n_qubits])
        gate_map = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
        gate = gate_map.get(self.rotation, qml.RY)
        for q in range(self.n_qubits):
            ops.append(gate(x_scaled[q], wires=q))
        return ops
