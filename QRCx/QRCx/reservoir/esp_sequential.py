"""ESP verification for SequentialDissipativeQRC: trace distance between
density matrices started from different random initial conditions, driven
by the *same* input sequence, must collapse below threshold within
`washout` steps -- this is the sequential-mode analogue of
reservoir/esp.py's statevector L2-distance check, adapted because the
reservoir now carries a genuine density matrix state across steps instead
of re-encoding a fresh window from |0><0| each time.
"""
import numpy as np


def _random_density_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Ginibre-ensemble random density matrix (Hilbert-Schmidt uniform)."""
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    return rho / np.trace(rho).real


def _trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    diff = rho1 - rho2
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def verify_esp_sequential(
    qrc,
    input_sequence: np.ndarray,
    n_initial_states: int = 8,
    n_steps: int = None,
    convergence_threshold: float = 1e-2,
    seed: int = 0,
) -> tuple:
    """Drive `n_initial_states` random initial rho's and the vacuum through
    the same input sequence; report the mean pairwise trace distance to the
    vacuum trajectory at every step.

    Args:
        qrc: SequentialDissipativeQRC instance.
        input_sequence: shape (T, n_features), T >= n_steps.
        n_steps: defaults to qrc.washout.
        convergence_threshold: trace-distance threshold at the final step.

    Returns:
        (converged: bool, distances: np.ndarray shape (n_steps,))
    """
    n_steps = n_steps or qrc.washout
    n_steps = min(n_steps, input_sequence.shape[0])
    rng = np.random.default_rng(seed)
    dim = qrc.ops.dim

    ref_rho = qrc.ops.vacuum()
    alt_rhos = [_random_density_matrix(dim, rng) for _ in range(n_initial_states)]

    distances = np.zeros(n_steps)
    for k in range(n_steps):
        x = input_sequence[k]
        ref_rho = qrc.step(ref_rho, x)
        alt_rhos = [qrc.step(rho, x) for rho in alt_rhos]
        pair_dists = [_trace_distance(ref_rho, rho) for rho in alt_rhos]
        distances[k] = float(np.mean(pair_dists))

    converged = bool(distances[-1] < convergence_threshold)
    return converged, distances
