"""Amplitude encoder skeleton — Tier 3.

Direct state preparation |ψ(x)⟩ = Σ_i x_i |i⟩.
Computationally expensive for N=8 (256 amplitudes); included for completeness.
"""
import numpy as np
from .base import BaseEncoder


class AmplitudeEncoder(BaseEncoder):
    """Amplitude encoding — skeleton only."""

    def encode(self, x: np.ndarray) -> list:
        # Placeholder: full implementation requires MottonenStatePrep
        return []
