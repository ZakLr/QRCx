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


def test_shuffle_surrogate_thresholding_zeroes_below_null_and_never_exceeds_raw():
    """Dambre 2012 thresholding must only zero out capacities, never inflate
    them, and thresholded totals must be <= the raw (unthresholded) totals."""
    qrc = SequentialDissipativeQRC(n_qubits=N_QUBITS, trotter_steps=3, gamma1=0.05, gamma2=0.02, washout=0)
    mc = measure_memory_capacity_sequential(
        qrc, n_steps=80, max_lag=6, threshold_surrogates=True, n_surrogates=10,
    )
    assert mc["threshold_surrogates"] is True
    assert mc["MC"] <= mc["MC_raw"] + 1e-9
    assert all(v >= 0.0 for v in mc["per_lag"])
    assert len(mc["surrogate_thresholds"]) == len(mc["per_lag"])

    ipc = measure_ipc_sequential(
        qrc, n_steps=80, max_lag=4, threshold_surrogates=True, n_surrogates=10,
    )
    assert ipc["total_ipc"] <= ipc["total_ipc_raw"] + 1e-9
    assert ipc["total_ipc"] == ipc["linear_ipc"] + ipc["nonlinear_ipc"]
