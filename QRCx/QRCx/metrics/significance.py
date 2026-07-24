"""Statistical significance testing for forecast comparisons — Sprint 3.

Diebold-Mariano test (Diebold & Mariano 1995) with HAC (Newey-West/Bartlett)
variance correction for h-step-ahead forecasts, and a moving-block bootstrap
(Kunsch 1989) for 95% CIs on skill differences. Every reported skill
difference between models must be accompanied by one of these.
"""
from typing import Callable, Optional

import numpy as np
from scipy import stats


def _hac_long_run_variance(d: np.ndarray, maxlags: int) -> float:
    """Bartlett-kernel (Newey-West) long-run variance of a series, no
    small-sample bias correction (denominator n, not n-maxlags), matching
    statsmodels' cov_hac(..., use_correction=False)."""
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    # Normalize every autocovariance term by n (not n - lag), matching
    # statsmodels' cov_hac meat-matrix convention (sum / nobs throughout).
    gamma0 = np.sum(dc ** 2) / n
    var = gamma0
    for lag in range(1, maxlags + 1):
        w = 1.0 - lag / (maxlags + 1)
        cov = np.sum(dc[lag:] * dc[:-lag]) / n
        var += 2.0 * w * cov
    return var


def diebold_mariano(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    h: int = 1,
    loss: str = "squared",
    alternative: str = "two-sided",
) -> dict:
    """Diebold-Mariano test: does the model's forecast error differ
    significantly from the baseline's, accounting for h-step-ahead
    forecast-error autocorrelation via a Bartlett/Newey-West HAC variance
    with maxlags = h - 1 (standard DM convention).

    Args:
        y_true: Ground truth, shape (n,).
        y_pred_model: Model predictions, shape (n,).
        y_pred_baseline: Baseline predictions, shape (n,).
        h: Forecast horizon (steps ahead) — sets the HAC truncation lag.
        loss: "squared" or "absolute".
        alternative: "two-sided", "less" (model better), or "greater".

    Returns:
        dict with dm_stat, p_value, mean_loss_diff, n, maxlags.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_model = np.asarray(y_pred_model, dtype=float)
    y_pred_baseline = np.asarray(y_pred_baseline, dtype=float)
    assert y_true.shape == y_pred_model.shape == y_pred_baseline.shape
    assert h >= 1

    e_model = y_true - y_pred_model
    e_baseline = y_true - y_pred_baseline

    if loss == "squared":
        d = e_model ** 2 - e_baseline ** 2
    elif loss == "absolute":
        d = np.abs(e_model) - np.abs(e_baseline)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    maxlags = h - 1
    dbar = float(d.mean())
    var = _hac_long_run_variance(d, maxlags)
    se = np.sqrt(var / n)

    if se == 0.0:
        dm_stat = 0.0 if dbar == 0.0 else np.sign(dbar) * np.inf
    else:
        dm_stat = dbar / se

    if alternative == "two-sided":
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    elif alternative == "less":
        p_value = float(stats.norm.cdf(dm_stat))
    elif alternative == "greater":
        p_value = float(1.0 - stats.norm.cdf(dm_stat))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "mean_loss_diff": dbar,
        "n": n,
        "maxlags": maxlags,
    }


def moving_block_bootstrap(
    stat_fn: Callable[[np.ndarray], float],
    n: int,
    block_length: Optional[int] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Moving-block bootstrap (Kunsch 1989) over index positions 0..n-1.

    `stat_fn(idx)` receives a resampled index array of length n (built from
    contiguous blocks, preserving local autocorrelation) and returns a
    scalar statistic. Default block length n**(1/3) (standard rule of thumb).

    Returns:
        dict with lower/upper CI bounds, boot_mean, boot_std, block_length,
        n_boot, and the raw bootstrap replicate array.
    """
    assert n > 1
    if block_length is None:
        block_length = max(1, int(round(n ** (1.0 / 3.0))))
    block_length = min(block_length, n)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length

    replicates = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        replicates[b] = stat_fn(idx)

    alpha = 1.0 - ci
    lower, upper = np.percentile(replicates, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])

    return {
        "lower": float(lower),
        "upper": float(upper),
        "boot_mean": float(replicates.mean()),
        "boot_std": float(replicates.std()),
        "block_length": block_length,
        "n_boot": n_boot,
        "ci": ci,
        "replicates": replicates,
    }


def skill_difference_ci(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_clim: np.ndarray,
    block_length: Optional[int] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """95% moving-block-bootstrap CI for skill(model) - skill(baseline),
    both measured against the same climatological reference y_clim.

    Skill is defined as in QRCx.metrics.forecast.skill_score
    (1 - MSE_forecast / MSE_clim); this bootstraps the paired residuals so
    the ratio-based skill statistic is recomputed on each resample rather
    than assuming a linear/Gaussian statistic.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_model = np.asarray(y_pred_model, dtype=float)
    y_pred_baseline = np.asarray(y_pred_baseline, dtype=float)
    y_clim = np.asarray(y_clim, dtype=float)
    n = len(y_true)
    assert y_pred_model.shape == (n,) and y_pred_baseline.shape == (n,) and y_clim.shape == (n,)

    def skill_diff(idx: np.ndarray) -> float:
        yt, ym, yb, yc = y_true[idx], y_pred_model[idx], y_pred_baseline[idx], y_clim[idx]
        mse_clim = np.mean((yt - yc) ** 2)
        if mse_clim == 0:
            return 0.0
        mse_model = np.mean((yt - ym) ** 2)
        mse_baseline = np.mean((yt - yb) ** 2)
        skill_model = 1.0 - mse_model / mse_clim
        skill_baseline = 1.0 - mse_baseline / mse_clim
        return float(skill_model - skill_baseline)

    result = moving_block_bootstrap(skill_diff, n, block_length=block_length, n_boot=n_boot, ci=ci, seed=seed)
    result["point_estimate"] = skill_diff(np.arange(n))
    return result
