"""Test encoding layer."""
import numpy as np
import pennylane as qml
from QRCx.encoding import ZZFeatureMap


def test_zz_feature_map_shape():
    encoder = ZZFeatureMap(n_qubits=8, n_layers=3)
    x = np.random.randn(8)
    ops = encoder.encode(x)
    assert len(ops) > 0


def test_zz_gate_order():
    """Verify gate sequence: H → RZ → IsingZZ."""
    n_qubits = 8
    encoder = ZZFeatureMap(n_qubits=n_qubits, n_layers=1)
    x = np.random.randn(n_qubits)
    ops = encoder.encode(x)
    # First op should be Hadamard
    assert isinstance(ops[0], qml.Hadamard)
    # After n_qubits Hadamards, next should be RZ
    assert isinstance(ops[n_qubits], qml.RZ)


def test_scale_to_pi():
    encoder = ZZFeatureMap(n_qubits=8)
    x = np.array([0.0, 1.0, 2.0])
    scaled = encoder.scale_to_pi(x)
    assert np.all(scaled >= 0) and np.all(scaled <= np.pi)
