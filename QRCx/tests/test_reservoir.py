import numpy as np
from QRCx.reservoir.tfim import AtmosphericQRC
from QRCx.reservoir.parallel import ParallelReservoir


def test_atmospheric_qrc_shape():
    qrc = AtmosphericQRC(n_qubits=12, trotter_steps=5, seed=42)
    X = np.random.randn(2, 24, 9)
    F = qrc.transform(X)
    assert F.shape == (2, 234)
    assert F.max() - F.min() > 0.1


def test_jg_ratio():
    qrc = AtmosphericQRC(n_qubits=12, seed=42)
    assert 0.8 < qrc.jg_ratio < 1.2


def test_parallel_reservoir_shape():
    par = ParallelReservoir(n_qubits=12, trotter_steps=5)
    X = np.random.randn(2, 24, 9)
    F = par.transform(X)
    assert F.shape == (2, 468)
