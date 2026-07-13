import numpy as np
import pennylane as qml

from QRCx.reservoir.sequential import SequentialDissipativeQRC
from QRCx.readout.correlators import extract_correlators, extract_correlators_dm

N_QUBITS = 4  # small for test speed; reference config is 12 (see README)


def test_trace_preserved_each_step():
    rng = np.random.default_rng(0)
    qrc = SequentialDissipativeQRC(
        n_qubits=N_QUBITS, trotter_steps=3, gamma1=0.05, gamma2=0.05,
        injection="ry", washout=2,
    )
    seq = rng.uniform(-1, 1, size=(10, 13))
    rho = qrc.ops.vacuum()
    for k in range(10):
        rho = qrc.step(rho, seq[k])
        assert abs(np.trace(rho).real - 1.0) < 1e-8
        assert np.max(np.abs(rho - rho.conj().T)) < 1e-8


def test_esp_convergence_under_damping():
    rng = np.random.default_rng(1)
    qrc = SequentialDissipativeQRC(
        n_qubits=N_QUBITS, trotter_steps=3, gamma1=0.15, gamma2=0.1,
        injection="ry", washout=20,
    )
    seq = rng.uniform(-1, 1, size=(20, 13))
    converged, distances = qrc.verify_esp(seq, n_initial_states=5, n_steps=20)
    assert distances[-1] < distances[0]  # fading memory: distance must shrink
    assert converged


def test_feature_matrix_shape():
    rng = np.random.default_rng(2)
    qrc = SequentialDissipativeQRC(n_qubits=N_QUBITS, trotter_steps=2, washout=3)
    seq = rng.uniform(-1, 1, size=(7, 13))
    features = qrc.drive(seq)
    n_feat = 3 * N_QUBITS + 3 * N_QUBITS * (N_QUBITS - 1) // 2
    assert features.shape == (7, n_feat)


def test_determinism_under_fixed_seed():
    rng = np.random.default_rng(3)
    seq = rng.uniform(-1, 1, size=(6, 13))
    qrc_a = SequentialDissipativeQRC(n_qubits=N_QUBITS, trotter_steps=2, gamma1=0.05, gamma2=0.05, seed=42)
    qrc_b = SequentialDissipativeQRC(n_qubits=N_QUBITS, trotter_steps=2, gamma1=0.05, gamma2=0.05, seed=42)
    feats_a = qrc_a.drive(seq)
    feats_b = qrc_b.drive(seq)
    assert np.array_equal(feats_a, feats_b)


def test_zero_noise_zz_injection_matches_unitary_statevector_limit():
    """gamma1=gamma2=0, injection='zz' must reproduce the v4-equivalent
    pure-unitary evolution exactly (up to floating point), validating the
    density-matrix engine against an independent PennyLane statevector
    reference circuit built from the same gate sequence."""
    n = N_QUBITS
    trotter = 2
    dt = 1.0 / trotter
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, size=n)

    qrc = SequentialDissipativeQRC(
        n_qubits=n, trotter_steps=trotter, gamma1=0.0, gamma2=0.0,
        injection="zz", J=1.0, g=1.0, washout=0,
    )
    rho = qrc.step(qrc.ops.vacuum(), x)

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def ref(x):
        for i in range(n):
            qml.Hadamard(wires=i)
        for i in range(n):
            qml.RZ(x[i % n], wires=i)
        for i in range(n):
            for j in range(i + 1, n):
                phi = (np.pi - x[i % n]) * (np.pi - x[j % n])
                qml.IsingZZ(phi, wires=[i, j])
        for _ in range(trotter):
            for i in range(n):
                qml.RX(2.0 * dt, wires=i)
            for i in range(n):
                for j in range(i + 1, n):
                    qml.IsingZZ(2.0 * dt, wires=[i, j])
        return qml.state()

    psi = np.array(ref(x))
    rho_ref = np.outer(psi, psi.conj())
    assert np.allclose(rho, rho_ref, atol=1e-8)

    feats_dm = extract_correlators_dm(rho, n)
    feats_pure = extract_correlators(psi, n)
    assert np.allclose(feats_dm, feats_pure, atol=1e-8)
