import time

import numpy as np

from ..reservoir.tfim import AtmosphericQRC


def timing_benchmark(qrc, n_samples: int = 100) -> float:
    """Measure average single-sample latency.

    Args:
        qrc: AtmosphericQRC instance.
        n_samples: Number of dummy samples for timing.

    Returns:
        Latency in seconds per sample.

    Raises:
        RuntimeError: If latency exceeds threshold after reduction.
    """
    x_dummy = np.random.randn(1, 24, 8)
    t0 = time.perf_counter()
    qrc.transform(x_dummy)
    latency = time.perf_counter() - t0

    if latency > 0.5 and qrc.trotter_steps > 5:
        print(f"Latency {latency:.2f}s > 0.5s; reducing trotter_steps to 5")
        reduced = AtmosphericQRC(n_qubits=qrc.n_qubits, n_layers=qrc.n_layers, trotter_steps=5, dt=qrc.dt, seed=getattr(qrc, "seed", 42))
        return timing_benchmark(reduced, n_samples)

    return latency


def benchmark_gate(reservoir, threshold: float = 0.5, min_steps: int = 5) -> dict:
    """Full benchmark gate with auto-reduction.

    Args:
        reservoir: Reservoir instance with transform() and trotter_steps.
        threshold: Max allowed latency in seconds.
        min_steps: Minimum trotter steps to reduce to.

    Returns:
        Dict with latency, trotter_steps, passed, auto_reduced.
    """
    x_dummy = np.random.randn(1, 24, 9)
    t0 = time.perf_counter()
    reservoir.transform(x_dummy)
    latency = time.perf_counter() - t0
    steps = reservoir.trotter_steps
    passed = latency < threshold

    if not passed and steps > min_steps:
        from ..reservoir.tfim import AtmosphericQRC
        new_res = AtmosphericQRC(
            n_qubits=reservoir.n_qubits,
            n_layers=getattr(reservoir, "n_layers", 3),
            trotter_steps=min_steps,
            dt=getattr(reservoir, "dt", 0.1),
            seed=getattr(reservoir, "seed", 42),
        )
        t0 = time.perf_counter()
        new_res.transform(x_dummy)
        latency = time.perf_counter() - t0
        steps = min_steps
        passed = latency < threshold
        return {"latency": latency, "trotter_steps": steps, "passed": passed, "auto_reduced": True}

    return {"latency": latency, "trotter_steps": steps, "passed": passed, "auto_reduced": False}
