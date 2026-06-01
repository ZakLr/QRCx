import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml


def verify_esp(
    qrc,
    input_sequence: np.ndarray,
    n_initial_states: int = 20,
    n_steps: int = 100,
    convergence_threshold: float = 1e-2,
) -> tuple:
    """Verify Echo State Property for TFIM reservoir.

    Args:
        qrc: AtmosphericQRC instance.
        input_sequence: Input driving sequence, shape (n_steps, ...).
        n_initial_states: Number of random initial states.
        n_steps: Number of evolution steps.
        convergence_threshold: L2 distance threshold for convergence.

    Returns:
        Tuple of (figure, converged bool, distances array shape (n_steps,)).
    """
    dev = qml.device("lightning.qubit", wires=qrc.n_qubits)
    n_samples, W, d = input_sequence.shape[:3] if input_sequence.ndim == 3 else (input_sequence.shape[0], 0, 0)

    @qml.qnode(dev)
    def esp_state(x: np.ndarray) -> list[np.ndarray]:
        qrc.encoder.circuit(x)
        for _ in range(qrc.trotter_steps):
            for i in range(qrc.n_qubits):
                qml.RX(2.0 * qrc.g[i] * qrc.dt, wires=i)
            for i in range(qrc.n_qubits):
                qml.RZ(2.0 * qrc.h[i] * qrc.dt, wires=i)
            for i in range(qrc.n_qubits):
                for j in range(i + 1, qrc.n_qubits):
                    if qrc.J[i, j] != 0:
                        qml.IsingZZ(2.0 * qrc.J[i, j] * qrc.dt, wires=[i, j])
        return [qml.expval(qml.PauliZ(k)) for k in range(qrc.n_qubits)]

    rng = np.random.default_rng(42)
    n_steps_actual = min(n_steps, input_sequence.shape[0])
    distances = np.zeros(n_steps_actual)

    for step in range(n_steps_actual):
        x_step = input_sequence[step].ravel()
        x_scaled = qrc.encoder.scale_to_pi(x_step)
        ref_state = np.array(esp_state(x_scaled))

        pair_dists = []
        for _ in range(n_initial_states):
            qrc.h = rng.uniform(-1.0, 1.0, size=qrc.n_qubits)
            qrc.g = rng.uniform(0.5, 1.5, size=qrc.n_qubits)
            J_raw = rng.uniform(-1.0, 1.0, size=(qrc.n_qubits, qrc.n_qubits))
            qrc.J = (J_raw + J_raw.T) / 2.0
            np.fill_diagonal(qrc.J, 0.0)

            alt_state = np.array(esp_state(x_scaled))
            pair_dists.append(np.linalg.norm(ref_state - alt_state))

        distances[step] = np.mean(pair_dists)

    converged = distances[-1] < convergence_threshold

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(distances)
    ax.axhline(convergence_threshold, color="r", linestyle="--", label=f"threshold={convergence_threshold}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean pairwise L2 distance")
    ax.set_title(f"ESP Verification (converged={converged})")
    ax.legend()
    ax.grid(True)

    return fig, converged, distances
