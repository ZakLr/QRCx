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
