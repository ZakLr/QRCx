"""Alternative density-matrix backends for benchmarking against
SequentialDissipativeQRC's default "numpy" engine (see sequential.py's
module docstring). These are minimal, benchmarking-only drivers: they
implement the same per-step update (RY/ZZ injection -> TFIM Trotter ->
amplitude damping + dephasing) and return the same 234-dim Pauli-correlator
features, carrying the density matrix across steps, but are not wrapped as
a full BaseReservoir subclass since only one backend ("numpy") is used in
production once the runtime gate picks a winner.
"""
import time

import numpy as np

from ..readout.correlators import extract_correlators_dm


def _feature_dim(n_qubits: int) -> int:
    return 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2


def drive_pennylane_mixed(
    sequence: np.ndarray,
    n_qubits: int,
    trotter_steps: int = 10,
    tau: float = 1.0,
    gamma1: float = 0.02,
    gamma2: float = 0.02,
    input_scaling: float = 1.0,
    injection: str = "ry",
    J: float = 1.0,
    g: float = 1.0,
) -> np.ndarray:
    import pennylane as qml

    dt = tau / trotter_steps
    dev = qml.device("default.mixed", wires=n_qubits)
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    @qml.qnode(dev)
    def step_circuit(rho_in, x):
        qml.QubitDensityMatrix(rho_in, wires=range(n_qubits))
        if injection == "ry":
            for j in range(min(len(x), n_qubits)):
                qml.RY(input_scaling * x[j], wires=j)
            if len(x) > n_qubits:
                qml.RZ(input_scaling * float(np.sum(x[n_qubits:])), wires=0)
        else:
            d = len(x)
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(x[i % d], wires=i)
            for (i, j) in pairs:
                phi = (np.pi - x[i % d]) * (np.pi - x[j % d])
                qml.IsingZZ(phi, wires=[i, j])
        for _ in range(trotter_steps):
            for i in range(n_qubits):
                qml.RX(2.0 * g * dt, wires=i)
            for (i, j) in pairs:
                qml.IsingZZ(2.0 * J * dt, wires=[i, j])
        if gamma2 > 0:
            for i in range(n_qubits):
                qml.PhaseDamping(gamma2, wires=i)
        if gamma1 > 0:
            for i in range(n_qubits):
                qml.AmplitudeDamping(gamma1, wires=i)
        return qml.density_matrix(wires=range(n_qubits))

    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[0, 0] = 1.0
    T = sequence.shape[0]
    features = np.zeros((T, _feature_dim(n_qubits)))
    for k in range(T):
        rho = np.array(step_circuit(rho, sequence[k]))
        features[k] = extract_correlators_dm(rho, n_qubits)
    return features


def drive_qiskit_aer(
    sequence: np.ndarray,
    n_qubits: int,
    trotter_steps: int = 10,
    tau: float = 1.0,
    gamma1: float = 0.02,
    gamma2: float = 0.02,
    input_scaling: float = 1.0,
    injection: str = "ry",
    J: float = 1.0,
    g: float = 1.0,
) -> np.ndarray:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Kraus
    from qiskit_aer import AerSimulator

    dt = tau / trotter_steps
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    sim = AerSimulator(method="density_matrix")

    def amp_damp_kraus(gamma):
        e0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
        e1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
        return Kraus([e0, e1])

    def phase_damp_kraus(gamma):
        c = np.sqrt(max(0.0, 1.0 - gamma))
        k0 = np.array([[1, 0], [0, c]], dtype=np.complex128)
        k1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)
        return Kraus([k0, k1])

    # Qiskit orders qubits little-endian (qubit 0 = least-significant tensor
    # factor), the opposite of NumpyDensityOps/correlators' big-endian
    # convention (qubit 0 = most-significant). Flipping the physical qubit
    # index at gate-application time (not the returned matrix) makes Aer's
    # returned density matrix directly comparable/identical to the numpy
    # backend's, verified against a single-gate reference case.
    def q(i):
        return n_qubits - 1 - i

    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[0, 0] = 1.0
    T = sequence.shape[0]
    features = np.zeros((T, _feature_dim(n_qubits)))

    for k in range(T):
        x = sequence[k]
        qc = QuantumCircuit(n_qubits)
        qc.set_density_matrix(DensityMatrix(rho))
        if injection == "ry":
            for j in range(min(len(x), n_qubits)):
                qc.ry(input_scaling * x[j], q(j))
            if len(x) > n_qubits:
                qc.rz(input_scaling * float(np.sum(x[n_qubits:])), q(0))
        else:
            d = len(x)
            for i in range(n_qubits):
                qc.h(q(i))
            for i in range(n_qubits):
                qc.rz(x[i % d], q(i))
            for (i, j) in pairs:
                phi = (np.pi - x[i % d]) * (np.pi - x[j % d])
                qc.rzz(phi, q(i), q(j))
        for _ in range(trotter_steps):
            for i in range(n_qubits):
                qc.rx(2.0 * g * dt, q(i))
            for (i, j) in pairs:
                qc.rzz(2.0 * J * dt, q(i), q(j))
        if gamma2 > 0:
            k_ch = phase_damp_kraus(gamma2)
            for i in range(n_qubits):
                qc.append(k_ch, [q(i)])
        if gamma1 > 0:
            k_ch = amp_damp_kraus(gamma1)
            for i in range(n_qubits):
                qc.append(k_ch, [q(i)])
        qc.save_density_matrix()
        result = sim.run(qc).result()
        rho = np.array(result.data(0)["density_matrix"])
        features[k] = extract_correlators_dm(rho, n_qubits)
    return features


def time_backend(fn, sequence: np.ndarray, **kwargs) -> tuple:
    """Return (steps_per_second, wall_clock_seconds, features)."""
    t0 = time.perf_counter()
    features = fn(sequence, **kwargs)
    elapsed = time.perf_counter() - t0
    steps = sequence.shape[0]
    return steps / elapsed if elapsed > 0 else float("inf"), elapsed, features
