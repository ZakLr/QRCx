import numpy as np

from QRCx.readout.dirac3_selector import build_problem, select, select_sa, select_dirac3


def _toy_problem(seed=0):
    rng = np.random.default_rng(seed)
    n, T = 30, 200
    F = rng.normal(size=(T, n))
    true_idx = rng.choice(n, size=5, replace=False)
    y = F[:, true_idx].sum(axis=1) + 0.01 * rng.normal(size=T)
    return F, y, true_idx


def test_build_problem_shapes():
    F, y, _ = _toy_problem()
    c, Q = build_problem(F, y)
    assert c.shape == (F.shape[1],)
    assert Q.shape == (F.shape[1], F.shape[1])
    assert np.allclose(Q, Q.T)


def test_select_sa_returns_k_indices():
    F, y, true_idx = _toy_problem()
    c, Q = build_problem(F, y)
    result = select_sa(c, Q, K=5, seed=1, n_restarts=2, n_iters=200)
    assert result.solver == "sa"
    assert len(result.selected_idx) == 5
    assert result.error is None
    # SA should recover most of the true informative features on this
    # easy, low-noise toy problem
    assert len(set(result.selected_idx.tolist()) & set(true_idx.tolist())) >= 3


def test_select_dirac3_fails_gracefully_without_credentials():
    F, y, _ = _toy_problem()
    c, Q = build_problem(F, y)
    result = select_dirac3(c, Q, K=5, max_retries=1)
    assert result.solver == "dirac3"
    assert result.error is not None
    assert len(result.selected_idx) == 0


def test_select_dispatch():
    F, y, _ = _toy_problem()
    c, Q = build_problem(F, y)
    r = select("sa", c, Q, K=5, n_restarts=1, n_iters=100)
    assert r.solver == "sa"
