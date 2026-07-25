import numpy as np

from QRCx.reservoir.sequential import SequentialDissipativeQRC
from QRCx.metrics.reservoir_sequential import (
    measure_memory_capacity_sequential,
    measure_ipc_sequential,
    measure_ipc_by_degree,
    drive_iid_gaussian,
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


def test_ipc_by_degree_matches_measure_ipc_sequential_for_degrees_1_and_2():
    """Sprint 5's degree-generalized IPC must reproduce the already-
    validated linear/quadratic split from measure_ipc_sequential exactly
    (same math, same Hermite polynomials for degree 1/2) when given the
    same precomputed (u, features) drive -- only degree 3 is new."""
    qrc = SequentialDissipativeQRC(n_qubits=N_QUBITS, trotter_steps=3, gamma1=0.05, gamma2=0.02, washout=0)
    u, feats = drive_iid_gaussian(qrc, n_steps=100, seed=0)
    old = measure_ipc_sequential(u=u, features=feats, max_lag=8)
    new = measure_ipc_by_degree(u=u, features=feats, max_lag=8, degrees=(1, 2, 3))
    assert new["C"].shape == (3, len(new["lags"]))
    assert new["lags"][0] == 0  # lag=0 included (unlike measure_ipc_sequential, which starts at lag=1)
    # Sum over lags>=1 only, to compare against measure_ipc_sequential (which never includes lag=0)
    lag_ge1 = [i for i, k in enumerate(new["lags"]) if k >= 1]
    assert abs(new["C"][0, lag_ge1].sum() - old["linear_ipc"]) < 1e-9
    assert abs(new["C"][1, lag_ge1].sum() - old["nonlinear_ipc"]) < 1e-9
    assert np.all(new["C"] >= -1e-9)
    assert np.isfinite(new["C"][2].sum())
