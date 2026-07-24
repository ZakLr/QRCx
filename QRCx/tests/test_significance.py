import numpy as np
import pytest

from QRCx.metrics.significance import (
    diebold_mariano, moving_block_bootstrap, skill_difference_ci,
)


def test_dm_matches_statsmodels_hac_ols_h1():
    """DM test at h=1 (maxlags=0) is exactly the t-stat of an OLS
    intercept-only regression of the loss differential with HAC(0) errors
    (Diebold & Mariano 1995's original equivalence)."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.normal(size=n)
    y_pred_model = y_true + rng.normal(scale=0.5, size=n)
    y_pred_baseline = y_true + rng.normal(scale=1.0, size=n)

    result = diebold_mariano(y_true, y_pred_model, y_pred_baseline, h=1, loss="squared")

    e_model = y_true - y_pred_model
    e_baseline = y_true - y_pred_baseline
    d = e_model ** 2 - e_baseline ** 2
    ols = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": 0, "use_correction": False})

    assert result["dm_stat"] == pytest.approx(float(ols.tvalues[0]), rel=1e-8)
    assert result["p_value"] == pytest.approx(float(ols.pvalues[0]), rel=1e-6)


def test_dm_matches_statsmodels_hac_ols_h6():
    """Same equivalence at h=6 (maxlags=5), verifying the Bartlett-kernel
    HAC correction for h-step-ahead forecast-error autocorrelation."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    n = 400
    y_true = rng.normal(size=n)
    y_pred_model = y_true + rng.normal(scale=0.5, size=n)
    y_pred_baseline = y_true + rng.normal(scale=0.9, size=n)

    result = diebold_mariano(y_true, y_pred_model, y_pred_baseline, h=6, loss="squared")

    e_model = y_true - y_pred_model
    e_baseline = y_true - y_pred_baseline
    d = e_model ** 2 - e_baseline ** 2
    ols = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})

    assert result["dm_stat"] == pytest.approx(float(ols.tvalues[0]), rel=1e-8)
    assert result["maxlags"] == 5


def test_dm_identical_forecasts_gives_zero_stat():
    y_true = np.random.default_rng(2).normal(size=100)
    pred = y_true + 0.3
    result = diebold_mariano(y_true, pred, pred, h=1)
    assert result["dm_stat"] == 0.0
    assert result["p_value"] == pytest.approx(1.0)


def test_dm_much_better_model_gives_significant_negative_stat():
    rng = np.random.default_rng(3)
    n = 500
    y_true = rng.normal(size=n)
    y_pred_model = y_true + rng.normal(scale=0.05, size=n)
    y_pred_baseline = y_true + rng.normal(scale=5.0, size=n)
    result = diebold_mariano(y_true, y_pred_model, y_pred_baseline, h=1, alternative="less")
    assert result["dm_stat"] < -5
    assert result["p_value"] < 0.001


def test_dm_absolute_loss_runs():
    rng = np.random.default_rng(4)
    y_true = rng.normal(size=50)
    result = diebold_mariano(y_true, y_true + 0.1, y_true + 0.2, loss="absolute")
    assert np.isfinite(result["dm_stat"])


def test_moving_block_bootstrap_ci_covers_true_mean_for_iid():
    rng = np.random.default_rng(5)
    n = 2000
    true_mean = 0.7
    data = rng.normal(loc=true_mean, scale=1.0, size=n)

    def stat_fn(idx):
        return float(data[idx].mean())

    result = moving_block_bootstrap(stat_fn, n, n_boot=1000, seed=42)
    assert result["lower"] < true_mean < result["upper"]
    assert result["boot_mean"] == pytest.approx(true_mean, abs=0.05)


def test_moving_block_bootstrap_reproducible_with_seed():
    data = np.arange(100, dtype=float)

    def stat_fn(idx):
        return float(data[idx].mean())

    r1 = moving_block_bootstrap(stat_fn, 100, n_boot=200, seed=7)
    r2 = moving_block_bootstrap(stat_fn, 100, n_boot=200, seed=7)
    assert r1["lower"] == r2["lower"]
    assert r1["upper"] == r2["upper"]


def test_skill_difference_ci_zero_when_models_identical():
    rng = np.random.default_rng(6)
    n = 300
    y_true = rng.normal(size=n)
    y_clim = np.zeros(n)
    pred = y_true + rng.normal(scale=0.3, size=n)

    result = skill_difference_ci(y_true, pred, pred, y_clim, n_boot=500, seed=1)
    assert result["point_estimate"] == pytest.approx(0.0)
    assert result["lower"] <= 0.0 <= result["upper"]


def test_skill_difference_ci_positive_when_model_better():
    rng = np.random.default_rng(7)
    n = 500
    y_true = rng.normal(size=n)
    y_clim = np.zeros(n)
    y_pred_model = y_true + rng.normal(scale=0.1, size=n)
    y_pred_baseline = y_true + rng.normal(scale=2.0, size=n)

    result = skill_difference_ci(y_true, y_pred_model, y_pred_baseline, y_clim, n_boot=1000, seed=2)
    assert result["point_estimate"] > 0
    assert result["lower"] > 0
