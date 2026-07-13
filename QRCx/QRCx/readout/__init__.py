from .base import BaseReadout
from .krr import KRRReadout
from .ridge import RidgeReadout
from .lasso import LassoReadout
from .mlp import MLPReadout
from .quantum_ridge import QuantumRidgeReadout


def get_readout(readout_name: str) -> BaseReadout:
    mapping = {
        "krr": KRRReadout,
        "ridge": RidgeReadout,
        "lasso": LassoReadout,
        "mlp": MLPReadout,
        "quantum_ridge": QuantumRidgeReadout,
    }
    if readout_name not in mapping:
        raise ValueError(f"Unknown readout: {readout_name}. Choose from {list(mapping.keys())}")
    return mapping[readout_name]()
