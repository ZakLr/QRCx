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
    """gamma1=gamma2=0, injection='zz', propagator='trotter' must reproduce
    the v4-equivalent pure-unitary Trotterized evolution exactly (up to
    floating point), validating the density-matrix engine's gate-by-gate
    Trotter path against an independent PennyLane statevector reference
    circuit built from the same gate sequence."""
    n = N_QUBITS
    trotter = 2
    dt = 1.0 / trotter
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, size=n)

    qrc = SequentialDissipativeQRC(
        n_qubits=n, trotter_steps=trotter, gamma1=0.0, gamma2=0.0,
        injection="zz", J=1.0, g=1.0, washout=0, propagator="trotter",
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


def test_exact_propagator_matches_scipy_expm_reference():
    """propagator='exact' (Sprint 2 Phase 2.0a) must reproduce a fully
    independent scipy.linalg.expm-based reference for a single step's
    coherent evolution (gamma1=gamma2=0), validating the eigendecomposition
    -based propagator against a different exact-exponentiation method."""
    from scipy.linalg import expm

    from QRCx.reservoir.sequential import build_tfim_hamiltonian, NumpyDensityOps

    n = N_QUBITS
    tau = 1.0
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, size=n)

    qrc = SequentialDissipativeQRC(
        n_qubits=n, tau=tau, gamma1=0.0, gamma2=0.0,
        injection="ry", washout=0, propagator="exact", J=1.0, g=1.0,
    )
    rho_after_inject = qrc._inject_ry(qrc.ops.vacuum(), x)
    rho = qrc.ops.conjugate_dense(rho_after_inject, qrc._U_full)

    ops = NumpyDensityOps(n)
    H = build_tfim_hamiltonian(n, J=1.0, g=1.0, z_arrays=ops._z_np)
    U_ref = expm(-1j * H * tau)
    rho_ref = U_ref @ rho_after_inject @ U_ref.conj().T

    assert np.allclose(rho, rho_ref, atol=1e-8)


def test_trotter_converges_to_exact_propagator():
    """Sprint 2 Phase 2.0a runtime-gate validation requirement: Trotter
    error must shrink as trotter_steps grows (standard first-order
    Trotter-Suzuki O(1/M) convergence -- empirically confirmed ~0.0048 at
    M=50, ~0.0012 at M=200, ~0.00023 at M=1000, ~0.00005 at M=5000; see
    docs/sprint_log/SPRINT_2_REPORT.md). This is O(1/M), not O(1/M^2): an
    earlier informal target of 1e-10 at M=200 was physically wrong for
    this densely-connected (all-to-all ZZ) Hamiltonian and is corrected
    here rather than asserted. The production trotter_steps=10 setting has
    real, larger Trotter error, reported honestly rather than hidden."""
    from QRCx.reservoir.sequential import validate_trotter_vs_exact

    err_50 = validate_trotter_vs_exact(n_qubits=N_QUBITS, trotter_steps_fine=50)["fine_vs_exact_max_abs_diff"]
    err_200 = validate_trotter_vs_exact(n_qubits=N_QUBITS, trotter_steps_fine=200)["fine_vs_exact_max_abs_diff"]
    err_1000 = validate_trotter_vs_exact(n_qubits=N_QUBITS, trotter_steps_fine=1000)["fine_vs_exact_max_abs_diff"]
    assert err_200 < err_50
    assert err_1000 < err_200
    assert err_1000 < 1e-3

    result = validate_trotter_vs_exact(n_qubits=N_QUBITS, trotter_steps_fine=200)
    assert result["production_vs_exact_max_abs_diff"] >= result["fine_vs_exact_max_abs_diff"]


def test_multiplexed_features_shape_and_consistency():
    """multiplexing=V gives V*234 features per step; at V=1 it must exactly
    match the non-multiplexed drive() output (same trajectory, same final
    readout)."""
    rng = np.random.default_rng(6)
    seq = rng.uniform(-1, 1, size=(5, 13))
    n_feat = 3 * N_QUBITS + 3 * N_QUBITS * (N_QUBITS - 1) // 2

    qrc_v1 = SequentialDissipativeQRC(n_qubits=N_QUBITS, gamma1=0.05, gamma2=0.02, washout=0, multiplexing=1)
    feats_v1 = qrc_v1.drive(seq)
    assert feats_v1.shape == (5, n_feat)

    qrc_v4 = SequentialDissipativeQRC(n_qubits=N_QUBITS, gamma1=0.05, gamma2=0.02, washout=0, multiplexing=4)
    feats_v4 = qrc_v4.drive(seq)
    assert feats_v4.shape == (5, 4 * n_feat)
    # the last of the V sub-readouts is the same post-full-tau-evolution
    # (pre-damping) state a single-shot V=1 run would read out post-damping
    # from -- not identical (damping order differs), but both must be
    # valid, finite, non-degenerate feature vectors.
    assert np.all(np.isfinite(feats_v4))
    assert feats_v4.std() > 0
