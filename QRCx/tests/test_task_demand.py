import numpy as np

from QRCx.metrics.task_demand import compute_demand_profile, legendre_value, _map_to_bounded


def test_legendre_values_match_known_polynomials():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert np.allclose(legendre_value(x, 1), x)
    assert np.allclose(legendre_value(x, 2), 0.5 * (3 * x ** 2 - 1))
    assert np.allclose(legendre_value(x, 3), 0.5 * (5 * x ** 3 - 3 * x))


def test_demand_profile_shape_and_bounds():
    rng = np.random.default_rng(1)
    series = rng.standard_normal(500)
    result = compute_demand_profile(series, horizon=1, delays=range(0, 5), degrees=(1, 2, 3))
    assert result["D"].shape == (3, 5)
    assert np.all(result["D"] >= -1e-9)
    assert np.all(result["D"] <= 1.0 + 1e-9)
    assert result["total_demand"] == result["D"].sum()


def test_demand_profile_detects_planted_autoregressive_signal():
    """y[t] = f(y[t-lag]) via a known degree -- demand should show its
    largest capacity at that (delay, degree) cell (see module docstring:
    exact degree separation isn't guaranteed since the empirical series
    isn't uniformly distributed, but the correct delay must dominate)."""
    rng = np.random.default_rng(2)
    n = 2000
    lag = 4
    y = np.zeros(n)
    y[:lag] = rng.standard_normal(lag) * 0.3
    for t in range(lag, n):
        y[t] = 0.9 * legendre_value(_map_to_bounded(np.array([y[t - lag]])), 2)[0] + 0.05 * rng.standard_normal()

    result = compute_demand_profile(y, horizon=lag, delays=range(0, 8), degrees=(1, 2, 3))
    D = result["D"]
    peak_delay = np.argmax(D.sum(axis=0))
    assert peak_delay == 0  # y[t+lag] = f(y[t]) -> peak at delay=0 relative to the forecast target
