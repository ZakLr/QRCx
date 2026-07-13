import pennylane as qml
import numpy as np


def _pauli_matrix(label: str) -> np.ndarray:
    if label == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    elif label == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    elif label == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(f"Unknown Pauli: {label}")


def _expectation(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(state.conj().T @ operator @ state))


def _kron_n(*matrices: np.ndarray) -> np.ndarray:
    result = np.array([1.0], dtype=np.complex128)
    for m in matrices:
        result = np.kron(result, m)
    return result


def single_body(state_vector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute <X_i>, <Y_i>, <Z_i> for all qubits.

    Args:
        state_vector: State vector of shape (2**n_qubits,).
        n_qubits: Number of qubits.

    Returns:
        Array of shape (3 * n_qubits,) ordered as all X, all Y, all Z.
    """
    assert state_vector.shape == (2 ** n_qubits,), f"Expected state of dim 2^{n_qubits}"
    result = np.zeros(3 * n_qubits, dtype=np.float64)
    eye = np.eye(2, dtype=np.complex128)
    for i in range(n_qubits):
        for p_idx, pauli in enumerate(["X", "Y", "Z"]):
            op = _pauli_matrix(pauli)
            matrices = [eye] * n_qubits
            matrices[i] = op
            full_op = _kron_n(*matrices)
            result[p_idx * n_qubits + i] = _expectation(state_vector, full_op)
    return result


def two_body(state_vector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute <Z_i Z_j>, <X_i X_j>, <Y_i Y_j> for all i < j.

    Args:
        state_vector: State vector of shape (2**n_qubits,).
        n_qubits: Number of qubits.

    Returns:
        Array of shape (3 * C(n_qubits, 2),) ordered as all ZZ, all XX, all YY.
    """
    n_pairs = n_qubits * (n_qubits - 1) // 2
    result = np.zeros(3 * n_pairs, dtype=np.float64)
    eye = np.eye(2, dtype=np.complex128)
    p_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for p_label in ["Z", "X", "Y"]:
                matrices = [eye] * n_qubits
                matrices[i] = _pauli_matrix(p_label)
                matrices[j] = _pauli_matrix(p_label)
                full_op = _kron_n(*matrices)
                result[p_idx] = _expectation(state_vector, full_op)
                p_idx += 1
    return result


def extract_correlators(state_vector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Concatenate single_body and two_body correlators.

    Args:
        state_vector: State vector of shape (2**n_qubits,).
        n_qubits: Number of qubits.

    Returns:
        Array of shape (3 * n_qubits + 3 * C(n_qubits, 2),).
        For the reference 12-qubit configuration this is 234.
    """
    sb = single_body(state_vector, n_qubits)
    tb = two_body(state_vector, n_qubits)
    result = np.concatenate([sb, tb])
    expected = 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2
    assert len(result) == expected, f"Expected {expected} correlators, got {len(result)}"
    return result


def _ptrace_1q(rho: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Reduced single-qubit density matrix via partial trace.

    Avoids ever materialising a 2**n_qubits x 2**n_qubits Pauli operator
    (unlike ``single_body``/``two_body``, which are only cheap for small
    n_qubits): cost is one O(4**n_qubits) reshape+contraction, independent
    of which qubit is traced to.
    """
    dim = 2 ** n_qubits
    P, S = 2 ** qubit, 2 ** (n_qubits - qubit - 1)
    t = rho.reshape(P, 2, S, P, 2, S)
    t = np.transpose(t, (0, 2, 1, 3, 5, 4))  # (P, S, ket_q, P, S, bra_q)
    t = t.reshape(P * S, 2, P * S, 2)
    return np.einsum("EaEb->ab", t)


def _ptrace_2q(rho: np.ndarray, q1: int, q2: int, n_qubits: int) -> np.ndarray:
    """Reduced two-qubit (4x4) density matrix via partial trace, q1 < q2."""
    assert q1 < q2
    dim = 2 ** n_qubits
    A, B, C = 2 ** q1, 2 ** (q2 - q1 - 1), 2 ** (n_qubits - q2 - 1)
    t = rho.reshape(A, 2, B, 2, C, A, 2, B, 2, C)
    # axes: (a_ket, q1_ket, b_ket, q2_ket, c_ket, a_bra, q1_bra, b_bra, q2_bra, c_bra)
    t = np.transpose(t, (0, 2, 4, 5, 7, 9, 1, 3, 6, 8))
    # -> (a,b,c,a',b',c', q1_ket, q2_ket, q1_bra, q2_bra)
    t = t.reshape(A * B * C, A * B * C, 2, 2, 2, 2)
    reduced = np.einsum("EEabcd->abcd", t)
    return reduced.reshape(4, 4)


def single_body_dm(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """Density-matrix analogue of ``single_body``: <X_i>,<Y_i>,<Z_i> via Tr(rho_i P)."""
    assert rho.shape == (2 ** n_qubits, 2 ** n_qubits)
    result = np.zeros(3 * n_qubits, dtype=np.float64)
    for i in range(n_qubits):
        rho_i = _ptrace_1q(rho, i, n_qubits)
        for p_idx, pauli in enumerate(["X", "Y", "Z"]):
            result[p_idx * n_qubits + i] = float(np.real(np.trace(rho_i @ _pauli_matrix(pauli))))
    return result


def two_body_dm(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """Density-matrix analogue of ``two_body``: <Z_iZ_j>,<X_iX_j>,<Y_iY_j> via Tr(rho_ij P⊗P)."""
    n_pairs = n_qubits * (n_qubits - 1) // 2
    result = np.zeros(3 * n_pairs, dtype=np.float64)
    p_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            rho_ij = _ptrace_2q(rho, i, j, n_qubits)
            for p_label in ["Z", "X", "Y"]:
                op = np.kron(_pauli_matrix(p_label), _pauli_matrix(p_label))
                result[p_idx] = float(np.real(np.trace(rho_ij @ op)))
                p_idx += 1
    return result


def extract_correlators_dm(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """Density-matrix analogue of ``extract_correlators``.

    Used by the sequential dissipative reservoir (mixed states from Kraus
    channels), where correlators must be computed from rho = 2**n_qubits x
    2**n_qubits rather than a pure state vector.
    """
    sb = single_body_dm(rho, n_qubits)
    tb = two_body_dm(rho, n_qubits)
    result = np.concatenate([sb, tb])
    expected = 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2
    assert len(result) == expected, f"Expected {expected} correlators, got {len(result)}"
    return result
