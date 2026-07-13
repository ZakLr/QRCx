import numpy as np

from QRCx.reservoir.sequential import SequentialDissipativeQRC
from QRCx.metrics.reservoir_sequential import (
    measure_memory_capacity_sequential,
    measure_ipc_sequential,
)

N_QUBITS = 4  # small for test speed; reference config is 12 (see README)


def test_memory_capacity_returns_nonnegative_finite_value():
    qrc = SequentialDissipativeQRC(n_qubits=N_QUBITS, trotter_steps=3, gamma1=0.05, gamma2=0.02, washout=0)
    result = measure_memory_capacity_sequential(qrc, n_steps=60, max_lag=8)
    assert np.isfinite(result["MC"])
    assert result["MC"] >= 0.0
    assert len(result["per_lag"]) == 8


def test_ipc_linear_plus_nonlinear_equals_total():
    qrc = SequentialDissipativeQRC(n_qubits=N_QUBITS, trotter_steps=3, gamma1=0.05, gamma2=0.02, washout=0)
    result = measure_ipc_sequential(qrc, n_steps=60, max_lag=5)
    assert result["total_ipc"] == result["linear_ipc"] + result["nonlinear_ipc"]
    assert np.isfinite(result["total_ipc"])
