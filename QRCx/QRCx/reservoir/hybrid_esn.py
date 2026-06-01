"""Hybrid ESN+QRC skeleton — Tier 3."""
from .base import BaseReservoir


class HybridESN(BaseReservoir):
    """Classical ESN concatenated with QRC features — skeleton only."""

    def __init__(self, n_qubits: int = 8):
        super().__init__(n_qubits)

    def transform(self, X):
        raise NotImplementedError("HybridESN is a Tier-3 skeleton.")
