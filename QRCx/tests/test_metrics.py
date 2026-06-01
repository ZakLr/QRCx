import numpy as np
from QRCx.metrics.forecast import rmse, skill_score, vpt
from QRCx.metrics.fsdh import compute_fsdh_curve, fsdh


def test_fsdh_returns_int():
    n = 100
    y_true = np.random.randn(n)
    y_model = y_true + 0.1 * np.random.randn(n)
    y_persist = np.random.randn(n)
    result = fsdh(y_true, y_model, y_persist)
    assert isinstance(result, int)


def test_fsdh_curve_returns_int():
    n, h = 100, 10
    y_true = np.random.randn(n, h)
    y_model = y_true + 0.1 * np.random.randn(n, h)
    y_persist = np.random.randn(n, h)
    result = compute_fsdh_curve(y_true, y_model, y_persist)
    assert isinstance(result, int)


def test_vpt_threshold():
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=50)
    y_pred = y_true.copy()
    assert vpt(y_true, y_pred, threshold=0.4) == 1.0
    y_pred_bad = y_true + 10.0
    assert vpt(y_true, y_pred_bad, threshold=0.4) == 0.0
